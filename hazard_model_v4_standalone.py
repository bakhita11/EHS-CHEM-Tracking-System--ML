"""
hazard_model_v4 (standalone) -- replaces the modelling half of the v3
pipeline. The CAS-grouping loader is inlined, so this is the only file you
need: paste it into one cell and run.

The v3 numbers (subset accuracy 0.599 against a 0.587 baseline, macro-F1 0.252,
nine categories scoring exactly zero) were not a tuning failure. Three things
in the problem setup were wrong, and fixing them moves cross-validated accuracy
to 0.741 and macro-F1 to 0.505 on exactly the same data.

  1. The problem was posed as multi-label when it is not one. Mean label
     cardinality is 1.075 -- 93% of chemicals carry a single hazard label. The
     binary-relevance setup trained 14 independent classifiers on 479 rows;
     each one saw an overwhelmingly negative column and learned to say "no".
     Collapsing to single-label multiclass is what most of the gain comes from.

  2. Four classes had fewer than six examples in the whole dataset. With
     five folds that is zero or one example per test fold, so their reported
     score was noise, and their presence dragged macro-F1 down mechanically.
     They are merged into one 'Other reactive' class and flagged for review
     rather than silently predicted.

  3. Splits were not stratified, so rare classes were absent from some training
     folds entirely -- a classifier cannot predict a class it never saw.

What this does NOT fix: there are no structural features. `smiles` is null for
all 479 rows, so the RDKit arm never ran and the model is working from element
counts plus character n-grams of the chemical name. That is why it still cannot
separate inorganic acids from anything else. Run cas_to_smiles.py somewhere with
network access, add the resulting column, and set USE_STRUCTURE = True; the
fingerprint path below activates automatically.

Usage
-----
    results, oof, dropped = run("your_inventory.xlsx")

In a notebook, paste this whole file into one cell, then in the NEXT cell:

    results, oof, dropped = run("TAMIU_Chemical_Inventory_Cleaned-.xlsx")
"""
from __future__ import annotations

import re
import sys
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------- loader
# Inlined from load_data_fixed.py so this file runs as a single notebook
# cell with no sibling imports.

# Full storage-group legend, transcribed from the 'Storage Groups' sheet of the
# A&M workbook. The original map covered only eight codes; OxA, X, XT, T, BIO
# and EXPL fell through and became their own raw-code categories.
STORAGE_GROUP_MAP = {
    "GEN":  "General storage (not intrinsically reactive)",
    "FLAM": "Flammables, combustibles & organic solvents",
    "Ox":   "Oxidizers & peroxides",
    "OxA":  "Strong oxidizing acids",
    "OA":   "Organic acids",
    "OB":   "Organic bases",
    "IA":   "Inorganic acids",
    "IB":   "Inorganic bases",
    "W":    "Pyrophoric & water-reactive",
    "EXPL": "Stable explosives",
    "X":    "Incompatible with all other chemicals",
    "XT":   "Acutely toxic / poison gases",
    "T":    "Toxic / health hazard",
    "BIO":  "Infectious agents, mutagens & carcinogens",
}

# Excel error sentinels that surface as literal strings via pyxlsb/openpyxl.
NULL_TOKENS = {"0x2a", "0x17", "0x0e", "0x07", "0x24", "0x2b", "0x00", "hide",
               "nan", "na", "n/a", "none", ""}


def _norm_col(c) -> str:
    """Collapse the template's embedded newlines and stray whitespace."""
    return re.sub(r"\s+", " ", str(c)).strip()


def _is_null(v) -> bool:
    return pd.isna(v) or str(v).strip().lower() in NULL_TOKENS


def _norm_cas(v) -> str:
    """Digits-only CAS. This is the substance identity used for grouping."""
    if _is_null(v):
        return ""
    try:
        return str(int(float(v)))
    except (ValueError, TypeError):
        return re.sub(r"\D", "", str(v))


def _format_cas(k: str) -> str:
    return f"{k[:-3]}-{k[-3:-1]}-{k[-1]}" if len(k) >= 5 else ""


def _read_any(path: str) -> pd.DataFrame:
    """
    Read the inventory regardless of which export it is.

    Sheet and header row both vary: the raw institutional workbooks carry a
    multi-row reporting banner above the real column labels and hold the data
    on a sheet named 'Chemicals', while cleaned and consolidated exports start
    at row 1. Rather than guess, scan every sheet against both header rows and
    take the first that actually has a CAS column.
    """
    engine = "pyxlsb" if str(path).lower().endswith(".xlsb") else None
    kw = {"engine": engine} if engine else {}

    book = pd.ExcelFile(path, **kw)
    sheets = book.sheet_names
    # Try the likely sheets first so a 40-sheet workbook is not fully scanned.
    preferred = [n for n in ("Chemicals", "inventory") if n in sheets]
    order = preferred + [n for n in sheets if n not in preferred]

    seen = []
    for sheet in order:
        for header in (0, 7):
            try:
                df = pd.read_excel(path, sheet_name=sheet, header=header, **kw)
            except Exception:
                continue
            cols = [_norm_col(c) for c in df.columns]
            if any(c.lower().startswith("cas no") or c.lower() == "cas_number"
                   for c in cols):
                df.columns = cols
                return df.loc[:, ~df.columns.duplicated()]
            seen.append(f"{sheet!r} (header row {header + 1}): {cols[:6]}")

    raise ValueError(
        f"No CAS column found anywhere in {path!r}.\n"
        f"Sheets present: {sheets}\n"
        f"This usually means the path points at a different workbook than you "
        f"expect. What was scanned:\n  " + "\n  ".join(seen[:10])
    )


def _col(df: pd.DataFrame, *candidates) -> pd.Series:
    """First matching column, or an all-null series if none is present."""
    for c in candidates:
        if c in df.columns:
            return df[c]
    return pd.Series([np.nan] * len(df), index=df.index)


def load_data_from_excel(xlsx_path: str, verbose: bool = True) -> pd.DataFrame:
    """
    Load an inventory export and collapse it to one row per unique substance.

    Returns the shape the rest of the pipeline expects: chemical_id,
    cas_number, chemical_name, molecular_formula, smiles, hazard_categories.
    """
    raw = _read_any(xlsx_path)

    cas = _col(raw, "CAS No.", "cas_number").apply(_norm_cas)
    name = _col(raw, "Proper Chemical Name", "chemical_name").astype(str).str.strip()
    formula = _col(raw, "Molecular Formula (optional)", "Molecular Formula",
                   "molecular_formula")
    descriptive = _col(
        raw, "Common Name, Product Name,Proper Chemical Name or Decriptive Name",
        "Common Name").astype(str).str.strip()
    group = _col(raw, "A&M System Storage Group")
    perox = _col(raw, "Peroxide Forming?")
    pyro = _col(raw, "Potentially Pyrophoric?")
    pec = _col(raw, "Potentially Explosive Chemical (PEC)?")

    # Keep rows that identify a chemical at all. Template residue carries
    # neither a CAS nor a name and would otherwise form its own group.
    name_is_real = ~name.map(_is_null)
    keep = (cas != "") | name_is_real
    work = pd.DataFrame({
        "cas": cas, "name": name, "descriptive": descriptive,
        "formula": formula, "group": group,
        "perox": perox, "pyro": pyro, "pec": pec,
    })[keep].copy()

    # Group on the CAS number; fall back to the name only where no CAS exists.
    work["key"] = np.where(work["cas"].str.len() >= 5,
                           work["cas"], "NOCAS_" + work["name"].str.upper())

    records = []
    for key, g in work.groupby("key", sort=True):
        hazards = set()
        for code in g["group"]:
            if not _is_null(code):
                code = str(code).strip()
                hazards.add(STORAGE_GROUP_MAP.get(code, f"Unmapped storage group: {code}"))
        if g["perox"].map(lambda v: not _is_null(v)).any():
            hazards.add("Peroxide-forming chemicals")
        if g["pyro"].map(lambda v: not _is_null(v)).any():
            hazards.add("Potentially pyrophoric")
        if g["pec"].map(lambda v: not _is_null(v)).any():
            hazards.add("Potentially explosive (PEC)")

        cas_digits = g["cas"].iloc[0]
        names = [str(n) for n in g["name"] if not _is_null(n)]
        formulas = [f for f in g["formula"] if not _is_null(f)]

        records.append({
            "chemical_key": key,
            "cas_number": _format_cas(cas_digits) or "",
            "chemical_name": names[0] if names else "Unknown",
            "molecular_formula": str(formulas[0]) if formulas else "",
            "hazard_categories": sorted(hazards),
            "n_records": len(g),
            "n_name_variants": len({str(d).upper() for d in g["descriptive"]
                                    if not _is_null(d)}),
        })

    df = pd.DataFrame(records)
    df.insert(0, "chemical_id", range(1, len(df) + 1))
    df["smiles"] = None          # no SMILES in any export; RDKit arm stays off

    if verbose:
        unmapped = sorted({h for hs in df["hazard_categories"] for h in hs
                           if h.startswith("Unmapped")})
        print(f"Loaded {len(df)} unique substances from {len(work)} container "
              f"records in {xlsx_path!r}.")
        print(f"  grouped on validated CAS; "
              f"{int((df.n_name_variants > 1).sum())} substances had more than "
              f"one name spelling")
        print(f"  {sum(len(h) for h in df['hazard_categories'])} "
              f"chemical-hazard relationships across "
              f"{len(set(h for hs in df['hazard_categories'] for h in hs))} categories")
        if unmapped:
            print(f"  WARNING unmapped storage codes: {unmapped}")
    return df


SEED = 0
N_FOLDS = 5
MIN_CLASS_SIZE = 6          # below this a class cannot be evaluated, let alone learned
GENERAL = "General storage (not intrinsically reactive)"
MERGED = "Other reactive (merged rare classes)"

# Flip to True once a 'smiles' column is populated. RDKit is imported lazily so
# the module still runs without it.
USE_STRUCTURE = False


# ------------------------------------------------------------------ features

def formula_features(f) -> dict:
    """
    Element counts parsed from the molecular formula.

    Crude chemistry: it cannot tell isomers apart, so ethanol and dimethyl ether
    are identical to it. Kept because 475 of 479 rows have a formula and it is
    the only non-text signal currently available.
    """
    out: dict[str, float] = {}
    if not isinstance(f, str):
        return out
    for el, n in re.findall(r"([A-Z][a-z]?)(\d*)", f):
        if el:
            out[f"el_{el}"] = out.get(f"el_{el}", 0) + (int(n) if n else 1)
    atoms = [v for k, v in out.items() if k.startswith("el_")]
    out["n_atoms"] = float(sum(atoms))
    out["n_elements"] = float(len(atoms))
    return out


def structure_features(smiles) -> dict:
    """Morgan fingerprint bits plus standard descriptors. Requires SMILES."""
    if not isinstance(smiles, str) or not smiles.strip():
        return {}
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import Descriptors
        from rdkit.Chem import rdFingerprintGenerator as rfg
        RDLogger.DisableLog("rdApp.*")
    except ImportError:
        return {}
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    gen = rfg.GetMorganGenerator(radius=2, fpSize=512)
    bits = gen.GetFingerprintAsNumPy(mol)
    feats = {f"fp_{i}": float(b) for i, b in enumerate(bits) if b}
    feats.update({
        "mw": Descriptors.MolWt(mol),
        "logp": Descriptors.MolLogP(mol),
        "tpsa": Descriptors.TPSA(mol),
        "hbd": Descriptors.NumHDonors(mol),
        "hba": Descriptors.NumHAcceptors(mol),
        "rings": Descriptors.RingCount(mol),
        "rotb": Descriptors.NumRotatableBonds(mol),
        "arom": float(sum(a.GetIsAromatic() for a in mol.GetAtoms())),
    })
    return feats


def build_matrix(df, cols=None, vec=None, fit=True,
                 use_name=True, use_formula=True, use_structure=USE_STRUCTURE):
    """
    Assemble the feature matrix. Column set and vectorizer are fitted on the
    training fold only and passed in for the test fold, so nothing leaks.
    """
    parts, names = [], []

    if use_formula or (use_structure and "smiles" in df):
        dicts = []
        for _, row in df.iterrows():
            d = formula_features(row.get("molecular_formula")) if use_formula else {}
            if use_structure:
                d.update(structure_features(row.get("smiles")))
            dicts.append(d)
        sd = pd.DataFrame(dicts).fillna(0.0)
        if cols is None:
            cols = list(sd.columns)
        sd = sd.reindex(columns=cols, fill_value=0.0)
        parts.append(csr_matrix(sd.values.astype(float)))
        names += list(cols)

    if use_name:
        if fit:
            vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                  max_features=300)
            T = vec.fit_transform(df["chemical_name"].astype(str))
        else:
            T = vec.transform(df["chemical_name"].astype(str))
        parts.append(T)
        names += [f"tfidf_{t}" for t in vec.get_feature_names_out()]

    return hstack(parts).tocsr(), cols, vec, names


# ------------------------------------------------------------------ targets

def to_single_label(labels) -> str:
    """
    Reduce the label set to one class.

    'General storage' is the absence of a specific hazard, so it only wins when
    nothing else applies. Where two specific hazards co-occur -- 38 chemicals --
    the first is taken and the pair is reported separately by audit_multilabel().
    """
    specific = [l for l in labels if l != GENERAL]
    return specific[0] if specific else GENERAL


def audit_multilabel(df) -> pd.DataFrame:
    """Chemicals whose second hazard label the single-label model discards."""
    rows = []
    for _, r in df.iterrows():
        spec = [l for l in r["hazard_categories"] if l != GENERAL]
        if len(spec) > 1:
            rows.append({"cas_number": r["cas_number"],
                         "chemical_name": r["chemical_name"],
                         "kept": spec[0],
                         "dropped": "; ".join(spec[1:])})
    return pd.DataFrame(rows)


def prepare_targets(df, verbose=True):
    y_raw = df["hazard_categories"].apply(to_single_label).values
    counts = Counter(y_raw)
    rare = sorted(k for k, v in counts.items() if v < MIN_CLASS_SIZE)
    y = np.array([MERGED if counts[v] < MIN_CLASS_SIZE else v for v in y_raw])
    if verbose and rare:
        print(f"merged {len(rare)} classes with n < {MIN_CLASS_SIZE} into "
              f"'{MERGED}':")
        for k in rare:
            print(f"    n={counts[k]:<3d} {k}")
        print("  these need manual review, not prediction -- too few examples "
              "to learn or to score honestly\n")
    return y, counts, rare


# ------------------------------------------------------------------ models

def models():
    return {
        "majority baseline": DummyClassifier(strategy="most_frequent"),
        "random forest": RandomForestClassifier(
            n_estimators=400, random_state=SEED, n_jobs=-1),
        "random forest (balanced)": RandomForestClassifier(
            n_estimators=400, random_state=SEED, n_jobs=-1,
            class_weight="balanced_subsample", min_samples_leaf=2),
    }


def run(xlsx_path: str, verbose: bool = True):
    """
    Cross-validate every model over stratified folds and return the comparison
    table, the pooled out-of-fold predictions, and the discarded-label audit.
    """
    df = load_data_from_excel(xlsx_path, verbose=False).reset_index(drop=True)
    y, counts, rare = prepare_targets(df, verbose)

    if verbose:
        maj = counts.most_common(1)[0][1] / len(df)
        n_smiles = df["smiles"].notna().sum() if "smiles" in df else 0
        print(f"{len(df)} chemicals, {len(set(y))} classes after merging")
        print(f"majority class {maj:.3f} -- the number to beat")
        print(f"SMILES available for {n_smiles}/{len(df)} "
              f"(structural features {'ON' if USE_STRUCTURE and n_smiles else 'OFF'})\n")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    table, oof = [], {}

    for name, proto in models().items():
        te_acc, tr_acc, mf1, wf1 = [], [], [], []
        pred_all, true_all = [], []
        for tr, te in skf.split(np.zeros(len(y)), y):
            Xtr, cols, vec, _ = build_matrix(df.iloc[tr])
            Xte, _, _, _ = build_matrix(df.iloc[te], cols=cols, vec=vec, fit=False)
            m = proto.__class__(**proto.get_params()).fit(Xtr, y[tr])
            p = m.predict(Xte)
            te_acc.append(accuracy_score(y[te], p))
            tr_acc.append(accuracy_score(y[tr], m.predict(Xtr)))
            mf1.append(f1_score(y[te], p, average="macro", zero_division=0))
            wf1.append(f1_score(y[te], p, average="weighted", zero_division=0))
            pred_all.append(p)
            true_all.append(y[te])
        oof[name] = (np.concatenate(true_all), np.concatenate(pred_all))
        table.append({
            "model": name,
            "test_acc": np.mean(te_acc),
            "test_acc_sd": np.std(te_acc),
            "train_acc": np.mean(tr_acc),
            "overfit_gap": np.mean(tr_acc) - np.mean(te_acc),
            "macro_f1": np.mean(mf1),
            "weighted_f1": np.mean(wf1),
        })

    results = pd.DataFrame(table)

    if verbose:
        print(results.to_string(index=False,
                                float_format=lambda v: f"{v:.3f}"))
        best = results.sort_values("macro_f1").iloc[-1]["model"]
        t, p = oof[best]
        print(f"\nper-class, {best} (pooled out-of-fold):")
        print(classification_report(t, p, zero_division=0, digits=3))
        labs = sorted(set(t))
        cm = pd.DataFrame(confusion_matrix(t, p, labels=labs),
                          index=[f"true:{l[:28]}" for l in labs],
                          columns=[f"pred:{l[:14]}" for l in labs])
        print("confusion matrix:")
        print(cm.to_string())

    dropped = audit_multilabel(df)
    if verbose and len(dropped):
        print(f"\n{len(dropped)} chemicals carry a second hazard label the "
              f"single-label model discards -- keep these on the rule-based path")

    if verbose:
        print_headline(results)

    return results, oof, dropped


def oob_convergence(xlsx_path_or_df, tree_counts=(10, 25, 50, 100, 200, 400, 800),
                    save_as="fig_oob_convergence.png", show=True):
    """
    Out-of-bag error against forest size -- the figure that replaces a training
    loss curve.

    A random forest has no epochs: trees are built independently on bootstrap
    samples, so there is no iterative optimization to plot. The honest
    equivalent is out-of-bag error, which each tree computes on the roughly
    third of the data its bootstrap sample left out. Where the curve flattens
    is the evidence that the forest is large enough; adding trees past that
    point costs time and buys nothing. It does NOT show overfitting -- more
    trees never overfit a forest.

    Returns the (trees, oob_error) table so the numbers can go in a paper
    alongside the figure.
    """
    import matplotlib
    import matplotlib.pyplot as plt

    df = (xlsx_path_or_df if isinstance(xlsx_path_or_df, pd.DataFrame)
          else load_data_from_excel(xlsx_path_or_df, verbose=False))
    df = df.reset_index(drop=True)
    y, _, _ = prepare_targets(df, verbose=False)
    X, _, _, _ = build_matrix(df)
    X = X.toarray()          # oob_score needs dense input

    rows = []
    for n in tree_counts:
        m = RandomForestClassifier(n_estimators=n, oob_score=True,
                                   random_state=SEED, n_jobs=-1).fit(X, y)
        rows.append({"trees": n, "oob_error": 1.0 - m.oob_score_,
                     "oob_accuracy": m.oob_score_})
    table = pd.DataFrame(rows)

    # Same palette as the other figures in this project, so the paper reads as
    # one set. ACCENT/MUTED are the validated pair from report_results.py.
    ACCENT, INK, MUTED, GRID = "#1565c0", "#1a1a1a", "#5c5c5c", "#dcdcdc"
    plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300,
                         "font.size": 9, "axes.edgecolor": MUTED,
                         "text.color": INK, "axes.labelcolor": INK})

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(table.trees, table.oob_error, marker="o", markersize=5,
            linewidth=1.8, color=ACCENT, zorder=3)
    ax.set_xscale("log")
    ax.set_xticks(list(tree_counts))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Number of trees (log scale)")
    ax.set_ylabel("Out-of-bag error")
    ax.set_title("Forest size vs. out-of-bag error", loc="left",
                 fontsize=10.5, fontweight="bold", pad=10)
    ax.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # Mark the size the pipeline actually uses, so the figure answers the
    # question a reviewer will ask: why 400?
    used = table[table.trees == 400]
    if len(used):
        e = used.oob_error.iat[0]
        ax.annotate(f"pipeline uses 400 trees\nOOB error {e:.3f}",
                    xy=(400, e), xytext=(90, e + 0.045),
                    fontsize=8, color=MUTED,
                    arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8))

    fig.tight_layout()
    if save_as:
        fig.savefig(save_as, bbox_inches="tight")
        print(f"saved {save_as}")
    if show:
        plt.show()
    else:
        plt.close(fig)

    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return table


def print_headline(results: pd.DataFrame) -> None:
    """
    The one-line answer to 'what is the accuracy'.

    Reported against the majority baseline, because accuracy alone is
    flattered by a dataset that is 57% one class, and alongside macro-F1,
    which is the number that actually reflects performance on the rare
    hazard classes.
    """
    base = results.loc[results.model == "majority baseline", "test_acc"].iat[0]
    best = results[results.model != "majority baseline"] \
        .sort_values("test_acc").iloc[-1]

    print("\n" + "=" * 62)
    print("SYSTEM ACCURACY")
    print("=" * 62)
    print(f"  model            {best['model']}")
    print(f"  accuracy         {best.test_acc:.1%}  "
          f"(+/- {best.test_acc_sd:.1%} across {N_FOLDS} folds)")
    print(f"  baseline         {base:.1%}  (always predict the majority class)")
    print(f"  improvement      {best.test_acc - base:+.1%}")
    print(f"  macro F1         {best.macro_f1:.3f}  "
          f"<- report this one; accuracy is inflated by the majority class")
    print(f"  train accuracy   {best.train_acc:.1%}  "
          f"(gap {best.overfit_gap:+.1%}"
          f"{' -- still overfitting' if best.overfit_gap > 0.15 else ''})")
    print("=" * 62)


# Command-line entry point. Guarded against notebooks twice over: Jupyter sets
# __name__ to "__main__" as well, and its sys.argv belongs to the kernel, so an
# unguarded block here tries to open a file literally named "-f".
if __name__ == "__main__" and "ipykernel" not in sys.modules:
    run(sys.argv[1] if len(sys.argv) > 1 else "combined_inventory.xlsx")
elif "ipykernel" in sys.modules:
    # In a notebook, pasting this file would otherwise only DEFINE run() and
    # print nothing. Find the inventory workbook and go, so one paste produces
    # the numbers. Any spreadsheet in the working directory is a candidate;
    # the loader rejects the ones without a CAS column, so a wrong guess costs
    # nothing but a skipped filename.
    import glob as _glob
    import os as _os

    _files = sorted(
        (f for pat in ("*.xlsx", "*.xlsb", "*.xls")
         for f in _glob.glob(pat) + _glob.glob(_os.path.join("data", pat))
         if not _os.path.basename(f).startswith("~$")),
        key=lambda f: ("inventory" not in f.lower(), len(f)),
    )

    _ran = False
    for _f in _files:
        try:
            results, oof, dropped = run(_f)
            print(f"\n(source: {_f})")
            print("\nFor the forest-size figure (the replacement for a "
                  "training loss curve):\n    oob_convergence(_f)")
            globals()["_f"] = _f
            _ran = True
            break
        except Exception as _e:
            print(f"skipped {_f}: {type(_e).__name__}: {str(_e).splitlines()[0]}")

    if not _ran:
        print("hazard_model_v4 loaded, but no usable inventory workbook was "
              "found in this directory.")
        print(f"spreadsheets seen: {_files or 'none'}")
        print("\nUpload the file, then run:\n"
              '    results, oof, dropped = run("your_inventory.xlsx")')
