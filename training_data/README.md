# LINGGE · L1000 Training-Set Preparation Guide
_Date: 2025-09-30 04:27_

This document explains **exactly** how the training dataset was produced from the LINCS L1000 data for the `lingge` project.  
It consolidates what’s in your notebooks (`00`–`04`) and the newest linear code we executed together to generate a **ready-to-train** bundle under `Processed_data/training_data`.

---

## 0) Goal & Outputs

**Goal:** Turn raw L1000 GCTX matrices (Phase I/II) + curated metadata into a clean training set with:
- **Targets**: 978-gene L1000 Level-5 signatures (`Y`)
- **Molecular features**: MoLFormer embeddings from `smiles_canonical` (`X_mol`)
- **Context features**: Dose (`dose_uM`, `log10(dose_uM+ε)`) and Cell (`cell_id`→index)
- **(Optional) Scaffold annotation** using Bemis–Murcko (from canonical SMILES) — saved for later splitting

**Final bundle (symlinked/copied):**
```
/ShangGaoAIProjects/Lingge/LINCS/data/Processed_data/training_data
├── Y_landmark_train_catalog_canonical.mmap          # memmap: (N, 978) float32
├── Y_landmark_train_catalog_canonical_with_scaffolds.index.parquet  # catalog + scaffold
├── X_mol_molformer_768d.npy                         # memmap file (N, 768) float32
├── X_mol_molformer_768d_l2.npy                      # memmap file (N, 768) float32 (L2 normalized) — optional
├── X_context_dose.npy                               # npy: (N, 2) float32  [dose_uM, log10_dose]
├── X_context_cell_idx.npy                           # npy: (N,) int32
├── cell_to_idx.json                                 # mapping: cell_id → idx
├── molformer_unique_embeddings.npy                  # npy: (n_unique_smiles, 768) float32
├── uniq_smiles_with_emb_row.parquet                 # map: smiles_canonical → row in unique embeddings
├── landmark_idx_phase1.npy                          # indices of 978 genes in Phase I
├── landmark_idx_phase2_in_p2coords.npy              # same landmarks in Phase II coordinates
├── landmark_gene_ids_phase1.txt                     # the 978 landmark gene IDs (Phase I order)
└── manifest.json                                    # machine-readable metadata (paths, formats, shapes)
```

We validated the final pack:
```
Shapes OK? True True True True
Y mean/std: 0.0024 / 1.3626
X_mol mean/std: -0.0021 / 0.5804
Dose head:
[[0.37037  -0.43136]
 [1.11111   0.04576]
 [10.       1.     ]]
Cell idx head: [5 5 5 5 5 5 5 5 5 5]
```

---

## 1) Inputs, Folders & Notebooks

**Raw & metadata (given by you):**
```
/ShangGaoAIProjects/Lingge/LINCS/data
├── GSE92742/GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx
├── GSE70138/GSE70138_Broad_LINCS_Level5_COMPZ_n118050x12328_2017_03_06.gctx
└── Processed_data
    ├── l1000_signatures_metadata_canonical.parquet
    └── ...
```

**Your notebooks (we referenced them; no re-standardization was re-run):**
```
/mnt/data
├── 00_make_l1000_metadata.ipynb
├── 01_RDKit_preprocessing.ipynb          # (already did canonicalization etc.)
├── 02_MoLFormer_download.ipynb
├── 03_L1000_main_data_preprocessing.ipynb
└── 04_MoLFormer_prcess_selected_molecular.ipynb
```

---

## 2) Phase I & II: Indexing & Consistency Checks

- Read GCTX shapes:
  - **Phase I (GSE92742)**: `(473,647 signatures, 12,328 genes)`
  - **Phase II (GSE70138)**: `(118,050 signatures, 12,328 genes)`
- Built & saved indices:
  - `GSE92742_gene_index.parquet`, `GSE92742_sig_index.parquet`
  - `GSE70138_gene_index.parquet`, `GSE70138_sig_index.parquet`
  - `gene_index_union.parquet` (pos_92742 / pos_70138 mapping)
- Confirmed the **12,328-gene set** aligns across phases (intersection = union = 12,328).

Artifacts (already present):
```
Processed_data/L1000gctx_process/
├── GSE92742_gene_index.parquet
├── GSE92742_sig_index.parquet
├── GSE70138_gene_index.parquet
├── GSE70138_sig_index.parquet
└── gene_index_union.parquet
```

---

## 3) Landmark Gene Index (978) & Cross-Phase Mapping

From Phase I `gene_info` (with `pr_is_lm==1`), we formed the **978** landmark row positions in **Phase I** order, then mapped them into **Phase II** coordinates using `gene_index_union.parquet`.

Artifacts:
- `landmark_idx_phase1.npy`
- `phase2_row_reorder_index.npy` (size 12,328)
- `landmark_idx_phase2_in_p2coords.npy`
- `landmark_gene_ids_phase1.txt`

---

## 4) Extract Y (978) from GCTX (h5py-safe batching)

Key practical details:
- HDF5 datasets are stored as **(signatures, genes)**; we index **rows by signatures** (using `col_pos` from catalog) and **columns by gene indices** (landmarks 978).
- **h5py constraints**:
  - No 2D fancy indexing → slice by rows first, then select columns in-memory.
  - Fancy indices must be **sorted** → batch-wise sort `sig_rows`, fetch, then un-sort to match catalog row order.

Output:
- `Y_landmark_train_catalog_canonical.mmap` — a **memmap** float32 array `(N, 978)`

Global stats (z-scores, clipped ±10):
```
mean ≈ 0.0024, std ≈ 1.3626, min ≈ -10, max ≈ 10
```

---

## 5) MoLFormer Embeddings (from canonical SMILES)

- Model: `ibm/MoLFormer-XL-both-10epochs` from Hugging Face
- Settings to avoid NaNs: **FP32**, `trust_remote_code=True`, **no autocast**
- Encode **unique** `smiles_canonical` only; then map back to each row.
- **QC (Step 16.5)**: all finite; ~25 vectors with higher norms but within IQR thresholds.

Artifacts:
- `molformer_unique_embeddings.npy` (shape: `n_unique_smiles × 768`)
- `uniq_smiles_with_emb_row.parquet` (map: smiles → row)
- `X_mol_molformer_768d.npy` (memmap `(N,768)`), and optional `_l2.npy`

Notes:
- We also saved an **L2-normalized** version for cosine-similarity style usage, but the recommended baseline is **raw + train-only z-score** later (after you pick a split).

---

## 6) Context Features: Dose & Cell

- **Dose**: continuous 2D `[[dose_uM, log10(dose_uM+ε)]]` → `X_context_dose.npy` `(N,2)`
- **Cell**: categorical index via mapping `cell_id → int` → `X_context_cell_idx.npy` `(N,)` and `cell_to_idx.json`

These are aligned **row-by-row** with the catalog and `Y`.

---

## 7) Scaffold Annotation (No re-standardization)

- Using your **existing canonical SMILES**, compute **Bemis–Murcko scaffold** (RDKit) and append as a new column `scaffold` to the catalog.
- Saved:
  - `Y_landmark_train_catalog_canonical_with_scaffolds.index.parquet`

> We did **not** redo RDKit standardization. This is only an annotation for any **future** scaffold splits.

---

## 8) Packaging for Training (+ `manifest.json` with formats)

We created a clean, self-describing bundle under:
```
/ShangGaoAIProjects/Lingge/LINCS/data/Processed_data/training_data
```
and wrote a **manifest** that declares **paths**, **formats**, and **shapes** to load everything uniformly.

### Manifest highlights
```json
{{
  "paths": {{
    "catalog": "Y_landmark_train_catalog_canonical_with_scaffolds.index.parquet",
    "Y_mmap": "Y_landmark_train_catalog_canonical.mmap",
    "X_mol_raw": "X_mol_molformer_768d.npy",
    "X_mol_l2": "X_mol_molformer_768d_l2.npy",
    "X_dose": "X_context_dose.npy",
    "X_cell_idx": "X_context_cell_idx.npy",
    "cell_to_idx": "cell_to_idx.json",
    "mol_unique_emb": "molformer_unique_embeddings.npy",
    "uniq_smiles_map": "uniq_smiles_with_emb_row.parquet",
    "landmark_idx_p1": "landmark_idx_phase1.npy",
    "landmark_idx_p2": "landmark_idx_phase2_in_p2coords.npy",
    "landmark_ids": "landmark_gene_ids_phase1.txt"
  }},
  "formats": {{
    "Y_mmap": "memmap",
    "X_mol_raw": "memmap",
    "X_mol_l2": "memmap",
    "X_dose": "npy",
    "X_cell_idx": "npy",
    "catalog": "parquet",
    "cell_to_idx": "json",
    "mol_unique_emb": "npy",
    "uniq_smiles_map": "parquet",
    "landmark_idx_p1": "npy",
    "landmark_idx_p2": "npy",
    "landmark_ids": "txt"
  }},
  "shapes": {{
    "N_rows": N,
    "Y": [N, 978],
    "X_mol": [N, 768],
    "X_dose": [N, 2],
    "X_cell_idx": [N]
  }}
}
```

---

## 9) Loading Example (memmap-aware)

```python
import os, json, numpy as np, pandas as pd

ROOT = "/ShangGaoAIProjects/Lingge/LINCS/data/Processed_data/training_data"
mani = json.load(open(f"{ROOT}/manifest.json"))

N = mani["shapes"]["N_rows"]; D_y = mani["shapes"]["Y"][1]; D_mol = mani["shapes"]["X_mol"][1]

# memmap loaders per manifest
Y = np.memmap(f"{ROOT}/{mani['paths']['Y_mmap']}", mode="r", dtype="float32", shape=(N, D_y))
X_mol = np.memmap(f"{ROOT}/{mani['paths']['X_mol_raw']}", mode="r", dtype="float32", shape=(N, D_mol))

# standard npy/parquet/json
X_dose = np.load(f"{ROOT}/{mani['paths']['X_dose']}")
X_cell = np.load(f"{ROOT}/{mani['paths']['X_cell_idx']}")
CAT    = pd.read_parquet(f"{ROOT}/{mani['paths']['catalog']}")

print(Y.shape, X_mol.shape, X_dose.shape, X_cell.shape, len(CAT))
```

---

## 10) Troubleshooting & Gotchas

- **memmap vs .npy**: files created via `np.memmap(..., mode="w+")` are **raw binaries** (no `.npy` header).  
  → Always load with `np.memmap` and the **right shape/dtype** (we declare them in `manifest.json`).  
  If you prefer `.npy`, convert once via `np.save(dest, np.asarray(np.memmap(...)))`.

- **h5py indexing**:
  - Avoid 2D fancy indexing;
  - Sort row indices within each batch before selection; then invert the permutation to restore original order.

- **MoLFormer NaN/Inf**:
  - Use **FP32**, `trust_remote_code=True`, **no autocast**; batch encode uniques, **QC** with finite checks;
  - If a batch fails on GPU, retry on CPU; drop or zero-fill rare pathological SMILES if any.

- **Normalization choices**:
  - Keep **raw** Mol embeddings for maximum information (vector norm may be informative).  
  - Optionally also keep **L2** for cosine-based methods.  
  - For training, compute **train-only** `(μ,σ)` z-score **after** you choose a split (to avoid leakage).

---

## 11) Reproducibility Notes

- **Paths** are absolute and consistently used in code cells we ran.  
- **Randomness**: where applicable (e.g., prospective scaffold split), set a fixed seed (e.g., 42).  
- **Environment**: `rdkit`, `h5py`, `torch`, `transformers (trust_remote_code)`; FP32 encoding for MoLFormer.

---

## 12) Next Steps (optional)

- Create `scaffold_split_indices.json` from the scaffolded catalog (80/10/10 or task-specific).  
- Compute **train-only** z-score stats for `X_mol` and/or `Y` using **train split only**.  
- Train a minimal model: `MolProj(768→d) ⊕ CellEmbedding ⊕ DoseMLP → 978` with MSE (+ optional -Pearson).

---

## Appendix · Key Artifacts Index

- **Catalogs**:
  - `Y_landmark_train_catalog_canonical.index.parquet`
  - `Y_landmark_train_catalog_canonical_with_scaffolds.index.parquet`

- **Targets**:
  - `Y_landmark_train_catalog_canonical.mmap` (N×978)

- **Molecular features**:
  - `molformer_unique_embeddings.npy`, `uniq_smiles_with_emb_row.parquet`
  - `X_mol_molformer_768d.npy` (memmap), `X_mol_molformer_768d_l2.npy` (memmap)

- **Context features**:
  - `X_context_dose.npy` (N×2), `X_context_cell_idx.npy` (N,), `cell_to_idx.json`

- **Landmark gene indices**:
  - `landmark_idx_phase1.npy`, `landmark_idx_phase2_in_p2coords.npy`, `landmark_gene_ids_phase1.txt`

- **Bundle**:
  - `Processed_data/training_data/manifest.json`
