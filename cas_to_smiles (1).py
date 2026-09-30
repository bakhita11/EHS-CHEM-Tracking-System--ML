"""
CAS -> SMILES lookup against PubChem, and molecular features for the hazard
classifier.

Why this exists
---------------
The v2 pipeline's features are character n-grams over the chemical name plus
element counts parsed from the molecular formula. Measured on the consolidated
inventory, Random Forest beats a majority-class baseline by about one point of
subset accuracy while scoring 0.99 on its own training data: it memorises and
does not generalise. That is a feature problem, not a tuning problem.

Structural descriptors are the fix. PubChem resolves a CAS number to a SMILES
string, and RDKit turns that into Morgan fingerprints and physicochemical
descriptors, which is the standard representation in cheminformatics and
carries the substructure signal that peroxide-formers and pyrophorics actually
share.

Running it
----------
Two stages, deliberately separate, because the first needs the network and the
second does not:

    python cas_to_smiles.py fetch   --in combined_inventory.xlsx
    python cas_to_smiles.py feature --in combined_inventory.xlsx

`fetch` writes smiles_cache.json incrementally, so an interrupted run resumes
where it stopped rather than starting over. `feature` reads that cache and
writes molecular_features.parquet (or .csv).

In Colab, install the dependency first:

    !pip install rdkit-pypi

Courtesy
--------
PubChem's usage policy caps requests at 5 per second and 400 per minute, and
asks that scripted access identify itself. Both are honoured below. The whole
inventory is roughly 550 lookups, so a cold run takes about three minutes.
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Optional

import pandas as pd

PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
USER_AGENT = ("TAMIU-EHS-inventory-research/1.0 "
              "(academic study; contact bakhita.salman@tamiu.edu)")

MIN_INTERVAL = 0.22        # seconds between requests: under PubChem's 5/sec cap
MAX_RETRIES = 4
CACHE_PATH = "smiles_cache.json"

# PubChem renamed its SMILES properties; older deployments serve the former
# spelling and newer ones the latter. Ask for each in turn rather than assuming.
PROPERTY_SETS = [
    "SMILES,MolecularFormula,MolecularWeight,InChIKey",
    "CanonicalSMILES,MolecularFormula,MolecularWeight,InChIKey",
    "ConnectivitySMILES,MolecularFormula,MolecularWeight,InChIKey",
]
SMILES_KEYS = ("SMILES", "CanonicalSMILES", "ConnectivitySMILES", "IsomericSMILES")

_last_request = [0.0]


def _throttle():
    wait = MIN_INTERVAL - (time.monotonic() - _last_request[0])
    if wait > 0:
        time.sleep(wait)
    _last_request[0] = time.monotonic()


def _get(url: str) -> Optional[dict]:
    """
    One PubChem request, with backoff. Returns parsed JSON, or None for a clean
    "not found" (404). Raises only when the service is unreachable entirely.
    """
    for attempt in range(MAX_RETRIES):
        _throttle()
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code in (400, 404):
                # 404: no such compound. 400: this property spelling is not
                # served by this deployment. Both mean "try the next query".
                return None
            if e.code in (429, 503) and attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)          # PubChem is throttling us
                continue
            if attempt == MAX_RETRIES - 1:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(2 ** attempt)
    return None


def _extract(payload: dict) -> Optional[dict]:
    try:
        props = payload["PropertyTable"]["Properties"][0]
    except (KeyError, IndexError, TypeError):
        return None
    smiles = next((props[k] for k in SMILES_KEYS if props.get(k)), None)
    if not smiles:
        return None
    return {
        "smiles": smiles,
        "pubchem_cid": props.get("CID"),
        "pubchem_formula": props.get("MolecularFormula"),
        "pubchem_mw": props.get("MolecularWeight"),
        "inchikey": props.get("InChIKey"),
    }


def lookup(cas: str, name: str = "") -> dict:
    """
    Resolve one substance to a SMILES string.

    Tries the CAS number first, since it is unambiguous, then falls back to the
    chemical name. The `source` field records which succeeded, because a
    name-resolved match is weaker evidence than a CAS-resolved one and should
    be reviewable later.
    """
    queries = []
    if cas:
        queries.append(("cas", str(cas)))
    # The name column carries NaN for some records, so coerce before testing.
    name = "" if name is None else str(name).strip()
    if name.lower() not in ("", "unknown", "nan", "none"):
        queries.append(("name", name))

    last_error = None
    for source, term in queries:
        quoted = urllib.parse.quote(str(term), safe="")
        for props in PROPERTY_SETS:
            url = f"{PUBCHEM}/compound/name/{quoted}/property/{props}/JSON"
            try:
                payload = _get(url)
            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
                continue                                # try the next spelling
            if payload is None:
                continue                                # not served, or no match
            found = _extract(payload)
            if found:
                found["status"] = "ok"
                found["source"] = source
                return found
    if last_error:
        return {"status": "error", "detail": last_error}
    return {"status": "not_found"}


# ---------------------------------------------------------------------------
# Stage 1 — fetch
# ---------------------------------------------------------------------------
def load_cache(path: str) -> dict:
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def save_cache(cache: dict, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(cache, fh, indent=1)
    os.replace(tmp, path)                               # atomic: no truncated cache


def read_substances(path: str) -> pd.DataFrame:
    """One row per substance, from any of the inventory exports."""
    sheets = pd.ExcelFile(path).sheet_names
    sheet = "inventory" if "inventory" in sheets else 0
    raw = pd.read_excel(path, sheet_name=sheet)
    raw.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in raw.columns]

    def col(*names):
        for n in names:
            if n in raw.columns:
                return raw[n]
        return pd.Series([""] * len(raw))

    def norm_cas(v):
        try:
            return str(int(float(v)))
        except (ValueError, TypeError):
            return re.sub(r"\D", "", str(v)) if pd.notna(v) else ""

    df = pd.DataFrame({
        "cas_key": col("cas_key", "CAS No.", "cas_number").apply(norm_cas),
        "chemical_name": col("Proper Chemical Name", "chemical_name").astype(str).str.strip(),
    })
    df = df[df.cas_key.str.len() >= 5].drop_duplicates("cas_key")
    df["cas_formatted"] = df.cas_key.map(lambda k: f"{k[:-3]}-{k[-3:-1]}-{k[-1]}")
    return df.reset_index(drop=True)


def fetch(args):
    subs = read_substances(args.infile)
    cache = load_cache(args.cache)
    todo = [r for r in subs.itertuples() if r.cas_key not in cache]

    print(f"{len(subs)} distinct substances; {len(cache)} already cached; "
          f"{len(todo)} to fetch.")
    if not todo:
        print("Nothing to do.")
        return
    print(f"At {MIN_INTERVAL:.2f}s between requests this takes about "
          f"{len(todo) * MIN_INTERVAL / 60:.1f} minutes.\n")

    ok = miss = err = 0
    for i, row in enumerate(todo, 1):
        result = lookup(row.cas_formatted, row.chemical_name)
        result["cas_formatted"] = row.cas_formatted
        result["chemical_name"] = row.chemical_name
        cache[row.cas_key] = result

        ok += result["status"] == "ok"
        miss += result["status"] == "not_found"
        err += result["status"] == "error"

        if i % 25 == 0 or i == len(todo):
            save_cache(cache, args.cache)           # survive an interrupted run
            print(f"  {i}/{len(todo)}  resolved {ok}  not found {miss}  errors {err}")

    save_cache(cache, args.cache)
    by_source = {}
    for v in cache.values():
        if v.get("status") == "ok":
            by_source[v.get("source", "?")] = by_source.get(v.get("source", "?"), 0) + 1
    print(f"\nCache written to {args.cache}")
    print(f"  resolved by CAS:  {by_source.get('cas', 0)}")
    print(f"  resolved by name: {by_source.get('name', 0)}   (weaker match — review these)")
    print(f"  unresolved:       {sum(1 for v in cache.values() if v.get('status') != 'ok')}")


# ---------------------------------------------------------------------------
# Stage 2 — features
# ---------------------------------------------------------------------------
# Functional groups chosen because they are what the hazard classes are defined
# by: peroxide-formers carry ethers and allylic positions, pyrophorics carry
# metal-carbon or metal-hydride bonds, oxidisers carry nitro and nitrate groups.
SMARTS = {
    "n_ether":          "[OD2]([#6])[#6]",
    "n_peroxide":       "[OX2][OX2]",
    "n_nitro":          "[$([NX3](=O)=O),$([NX3+](=O)[O-])]",
    "n_nitrate":        "[$([NX3](=[OX1])(=[OX1])O)]",
    "n_hydroxyl":       "[OX2H]",
    "n_carboxyl":       "[CX3](=O)[OX2H1]",
    "n_amine":          "[NX3;H2,H1;!$(NC=O)]",
    "n_halogen":        "[F,Cl,Br,I]",
    "n_aromatic_ring":  "a1aaaaa1",
    "n_alkene":         "[CX3]=[CX3]",
    "n_alkyne":         "[CX2]#[CX2]",
    "n_azide":          "[NX1]~[NX2]~[NX2]",
    "n_metal":          "[Li,Na,K,Mg,Ca,Al,Zn,Fe,Cu,Cr,Mn,Ni,Pb,Hg,Cd,Ba,Sr]",
    "n_sulfur":         "[#16]",
    "n_phosphorus":     "[#15]",
}


def features(args):
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import Descriptors, rdFingerprintGenerator
    except ImportError:
        sys.exit("RDKit is not installed. Run:  pip install rdkit-pypi")
    RDLogger.DisableLog("rdApp.*")                  # parse warnings are expected

    cache = load_cache(args.cache)
    if not cache:
        sys.exit(f"No cache at {args.cache}. Run the fetch stage first.")

    patterns = {k: Chem.MolFromSmarts(v) for k, v in SMARTS.items()}
    bad = [k for k, v in patterns.items() if v is None]
    if bad:
        sys.exit(f"Invalid SMARTS patterns: {bad}")

    morgan = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=args.bits)

    rows, unparsed = [], []
    for cas_key, rec in sorted(cache.items()):
        if rec.get("status") != "ok":
            continue
        mol = Chem.MolFromSmiles(rec["smiles"])
        if mol is None:
            unparsed.append((cas_key, rec.get("chemical_name", ""), rec["smiles"]))
            continue

        row = {
            "cas_key": cas_key,
            "cas_formatted": rec.get("cas_formatted", ""),
            "chemical_name": rec.get("chemical_name", ""),
            "smiles": rec["smiles"],
            "inchikey": rec.get("inchikey"),
            "resolved_by": rec.get("source", ""),
            # Physicochemical descriptors
            "mol_weight": Descriptors.MolWt(mol),
            "log_p": Descriptors.MolLogP(mol),
            "tpsa": Descriptors.TPSA(mol),
            "n_heavy_atoms": mol.GetNumHeavyAtoms(),
            "n_rings": Descriptors.RingCount(mol),
            "n_aromatic_rings": Descriptors.NumAromaticRings(mol),
            "n_rotatable_bonds": Descriptors.NumRotatableBonds(mol),
            "n_h_donors": Descriptors.NumHDonors(mol),
            "n_h_acceptors": Descriptors.NumHAcceptors(mol),
            "fraction_csp3": Descriptors.FractionCSP3(mol),
            "n_radical_electrons": Descriptors.NumRadicalElectrons(mol),
            "n_valence_electrons": Descriptors.NumValenceElectrons(mol),
        }
        for key, patt in patterns.items():
            row[key] = len(mol.GetSubstructMatches(patt))

        fp = morgan.GetFingerprintAsNumPy(mol)
        for bit in range(args.bits):
            row[f"fp_{bit}"] = int(fp[bit])
        rows.append(row)

    feat = pd.DataFrame(rows)
    out = args.out
    if out.endswith(".parquet"):
        try:
            feat.to_parquet(out, index=False)
        except Exception:
            out = out.replace(".parquet", ".csv")
            feat.to_csv(out, index=False)
    else:
        feat.to_csv(out, index=False)

    dense = [c for c in feat.columns if not c.startswith("fp_")]
    print(f"Featurized {len(feat)} substances -> {out}")
    print(f"  {len(dense) - 6} descriptors + {args.bits} fingerprint bits")
    print(f"  fingerprint bits that are ever set: "
          f"{int((feat[[c for c in feat.columns if c.startswith('fp_')]].sum() > 0).sum())}")
    if unparsed:
        print(f"\n  {len(unparsed)} SMILES strings RDKit could not parse:")
        for cas, name, smi in unparsed[:10]:
            print(f"    {cas}  {name[:40]:40} {smi[:50]}")


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = ap.add_subparsers(dest="stage", required=True)

    f = sub.add_parser("fetch", help="resolve CAS numbers to SMILES via PubChem")
    f.add_argument("--in", dest="infile", default="combined_inventory.xlsx")
    f.add_argument("--cache", default=CACHE_PATH)
    f.set_defaults(func=fetch)

    g = sub.add_parser("feature", help="build RDKit features from the cache")
    g.add_argument("--in", dest="infile", default="combined_inventory.xlsx")
    g.add_argument("--cache", default=CACHE_PATH)
    g.add_argument("--out", default="molecular_features.parquet")
    g.add_argument("--bits", type=int, default=1024,
                   help="Morgan fingerprint length (default 1024)")
    g.set_defaults(func=features)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__" and "ipykernel" not in sys.modules:
    main()
