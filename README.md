# Hazard-Class Prediction from Institutional Inventory Records

Code accompanying **"How Far Does Chemical Identity Go? Bounding Hazard-Class Prediction
from Institutional Inventory Records"** (Salman, Khasawneh, Yassin & Rodriguez).

The companion repository, [EHS-CHEM-Tracking-System](https://github.com/bakhita11/EHS-CHEM-Tracking-System),
holds the information architecture and the deterministic hazard cross-walk. **This**
repository holds the supervised-learning work: whether hazard class can be *predicted* for a
substance the institutional reference library does not cover.

---

## The problem this code is built around

Institutional inventories carry hazard fields for the substances they already list. Those
fields are populated by registry-number lookup against a reference library — in the inventory
studied here, agreeing with it at 99.7% and 99.3%.

That has a consequence most evaluations miss. **If the labels are a deterministic function of
the CAS number, any model given the CAS number reproduces the lookup.** It can score near
1.0 while having learned nothing about hazard. This is label leakage, it is invisible to
ordinary cross-validation, and it must be excluded by construction rather than checked for
afterwards.

So the task is posed narrowly: predict hazard class from **chemical identity alone** — the name
and, where recorded, the molecular formula. No CAS number, no CAS validity flag, no field
derived from the reference library.

---

## Headline result

| Configuration | Held-out acc. | Macro F₁ | Training acc. | Gap |
|---|---|---|---|---|
| Majority-class baseline | 0.566 | 0.080 | 0.566 | −0.000 |
| Name *n*-grams only | 0.697 | 0.387 | 0.980 | 0.283 |
| Molecular formula only | 0.672 | 0.318 | 0.875 | 0.203 |
| Name + formula | **0.741** | 0.470 | 0.997 | 0.256 |
| Name + formula, class-balanced | 0.714 | **0.505** | 0.920 | 0.206 |

Five-fold stratified cross-validation, 479 substances, 9 classes after merging four with
fewer than six members.

**Read this as a floor, not a capability.** Two of nine classes are not recovered at all, and
the 25.6-point separation between training and held-out accuracy — combined with out-of-bag
error that settles by 200 trees — puts the deficit in the *representation*, not in model
capacity. What distinguishes an oxidizer from a flammable is the arrangement of atoms, not
their counts, and no institutional export records structure.

---

## Requirements

```bash
pip install pandas numpy scikit-learn openpyxl pyxlsb matplotlib
pip install rdkit          # only for the structural-feature path
```

## Quickstart

```bash
python hazard_model_v4.py inventory.xlsx
```

Or paste `hazard_model_v4_standalone.py` into a Jupyter or Colab cell — it inlines the loader,
finds the workbook itself, and prints the report with no arguments.

```python
results, oof, dropped = run("inventory.xlsx")
oob_convergence("inventory.xlsx")      # forest size vs out-of-bag error
```

`results` is the comparison table, `oof` the pooled out-of-fold predictions per model, and
`dropped` the 30 substances whose secondary hazard label the single-label reduction sets
aside.

---

## Four decisions that drive the result

**The problem is not multi-label.** Mean label cardinality is 1.075 and 91.4% of substances
carry exactly one label. Under binary relevance, fourteen classifiers are each fitted to a
column that is overwhelmingly negative on 479 rows, and each minimizes its loss by predicting
the negative class throughout. Collapsing to single-label multiclass is where most of the
gain over the original pipeline came from.

**Records are grouped by validated CAS number.** Grouping on a key that concatenates CAS with
a free-text name splits one substance across several groups — zinc appears under eight name
variants — putting near-identical copies in both training and test folds.

**Four classes were merged, not deleted.** They held fewer than six substances each; one held
a single substance. Across five folds such a class contributes zero or one instance per
held-out fold, so any per-class score for it is noise, and its presence depresses macro-F₁
mechanically. They are retained as one explicit *Other reactive* residual and flagged for
manual review rather than prediction.

**Splits are stratified.** Without it, a class of nine substances is absent from some training
folds entirely and cannot be predicted at all.

---

## Files

| File | Purpose |
|---|---|
| `hazard_model_v4.py` | The pipeline: loading, label construction, features, cross-validation, out-of-bag convergence. |
| `hazard_model_v4_standalone.py` | The same with the loader inlined — one file, paste into a notebook cell. |
| `cas_to_smiles.py` | Two-stage PubChem lookup resolving CAS numbers to SMILES, with an incremental cache and SMARTS substructure matching. Needed for the structural-feature path. |
| `report_results.py` | Figures and the results summary table. |

---

## Structural features

The structural path is written but **not yet exercised**, because no institutional export
carries a SMILES representation. To run it:

1. `python cas_to_smiles.py` — resolves CAS numbers against PubChem (rate-limited to 5
   requests/second; the cache makes it resumable).
2. Add the resulting `smiles` column to the input.
3. Set `USE_STRUCTURE = True` in `hazard_model_v4.py`.

Morgan fingerprints (radius 2) and RDKit descriptors then enter the feature matrix
automatically. This is the experiment the paper's bound is designed to be measured against.

Two cautions for anyone extending this. Once structures exist, a **scaffold split** should be
reported alongside the random split — the gap between them is itself a result. And a
screening application wants **calibrated abstention** rather than a forced assignment: an
unflagged pyrophoric is a materially worse error than a redundant manual review.

---

## Data

The inventory is **not included.** It is an institutional Environmental Health and Safety
record giving the identity, quantity and physical storage location of hazardous materials in
a working university laboratory; publishing container-level locations of pyrophoric,
peroxide-forming and acutely toxic materials is a security consideration the institution
manages through role-based access.

Derived aggregate results are complete in the manuscript. For the underlying records, contact
the corresponding author; requests are handled in coordination with the institution's EHS
office.

### Expected columns

`CAS No.`, `Proper Chemical Name`, `Molecular Formula (optional)`,
`A&M System Storage Group`, `Peroxide Forming?`, `Potentially Pyrophoric?`,
`Potentially Explosive Chemical (PEC)?`

The loader scans every sheet for a CAS column rather than assuming a layout, so raw and
cleaned exports both work.

---

## Limitations

- 479 substances, from one institution. Small for supervised learning, and the hazard
  distribution is specific to this setting.
- No structural representation, which is the paper's central finding about the data rather
  than an oversight.
- Labels originate in a single reference library, so they cannot be validated against
  themselves. Any accuracy claim requires an external source.
- The two unrecovered classes and the four merged rare classes belong on a manual path at
  this sample size. A multi-institution corpus is the only route to them.

---

## Citation

```bibtex
@article{Salman2026HazardPrediction,
  author  = {Salman, Bakhita and Khasawneh, Mahmoud and Yassin, Muneeb and Rodriguez, Cristian},
  title   = {How Far Does Chemical Identity Go? Bounding Hazard-Class Prediction
             from Institutional Inventory Records},
  year    = {2026},
  note    = {Manuscript under review}
}
```

---

 
