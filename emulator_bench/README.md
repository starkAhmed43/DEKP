# emulator_bench

This folder adds an emulator-bench style retraining workflow to `DEKP` without rewriting the original repo entrypoints.

## What It Adds

- Explicit train/val/test split training for EMULaToR baseline directories
- One-time cached protein, ligand, and structure payloads under the split root
- Auto AMP selection: `bf16` on Ampere/Hopper-class CUDA, otherwise `fp16`
- Optuna tuning for optimizer-side hyperparameters only
- Parallel multi-GPU Optuna workers against one shared study
- Parallel multi-GPU retraining from the best Optuna trial
- Stable output files for later aggregation and comparison

## Expected Split Root

Default base dir:

- `~/github/EMULaToR/data/processed/baselines/DEKP`

Expected split layout:

- `random_splits/train.parquet`
- `random_splits/val.parquet`
- `random_splits/test.parquet`
- `enzyme_sequence_splits/threshold_x/train.parquet`
- `substrate_splits/threshold_x/train.parquet`

## Inputs The Bench Expects

Minimum required tabular columns:

- `sequence` or `Sequence`
- `smiles` or `Smiles`
- `log10_value` or `Label` or `value`

Protein identity columns, if present:

- `uniprot_id`
- `UniprotID`
- `protein_id`

Structure identity columns, if present:

- `structure_id`
- `pdb_id`
- otherwise the protein id is reused

Optional lookup columns:

- `CID` for legacy `trfm.pkl` / `molformer.pkl`
- `pdb_type`
- `pdb_source`
- `pdbs`

## What Gets Cached

Protein cache:

- normalized protein sequence
- cached protein token ids for the trainable CNN branch
- cached ProtT5 pooled embedding when `t5` is enabled

Ligand cache:

- raw SMILES
- cached SMILES token ids for the trainable CNN branch
- cached SMILES Transformer embedding when `trfm` is enabled
- optional cached MolFormer embedding when loaded from legacy features

Structure cache:

- protein structure graph used by DEKP’s graph encoder
- optional cached PST embedding when loaded from legacy features
- optional cached DSSP mean feature vector

Shared tokenizer cache:

- fixed amino-acid token vocab for sequence CNN input
- split-derived SMILES token vocab for molecule CNN input

Everything is written once under:

- `<base_dir>/embeddings`

The manifest lives at:

- `<base_dir>/embeddings/manifest.json`

## Feature Sources

Default DEKP feature set for the bench:

- `trfm,t5`

The intended DEKP setup in this bench is:

- ProtT5 for protein sequence embeddings
- SMILES Transformer for substrate embeddings
- DEKP's graph encoder fed by protein graphs built from PDB structures

Feature resolution order:

1. Reuse legacy DEKP-style pickles from `--legacy_feature_dir`
2. Compute supported features directly when possible

Directly computed by this bench:

- `t5` via `Rostlab/prot_t5_xl_uniref50`
- `trfm` via the SMILES Transformer weights path
- structure graph from PDB files
- `dssp` from PDB files when requested

Optional legacy-only extras if you explicitly ask for them:

- `pst`
- `molformer`

Those are not required for the default DEKP bench path anymore.

## Main Speedups

- expensive pretrained embeddings are cached once per unique entity instead of once per split or once per epoch
- tokenization is cached once instead of being redone in every dataset item fetch
- structure graphs are cached once instead of being rebuilt for every retrain
- AMP is automatic and consistent across cache building and training
- the graph encoder layer stack is instantiated correctly per layer in the bench model
- training outputs are explicit and split-aware, which makes Optuna and multi-GPU fan-out cheap to orchestrate

## Main Commands

See `commands.txt` in this folder for copy-ready examples.

## Notes

- The bench does not mutate the original `DEKP/pretrain.py` or `DEKP/fine_tune.py` workflow.
- Cache building reads the full `km_kinetic_params_3d.parquet` file by default, then deduplicates sequence, SMILES, and PDB identities once for the whole dataset.
- If you provide the three PDB root directories, the bench can build the DEKP protein graphs directly.
- `--experimental_pdb_dir` is used when `pdb_type == experimental`, resolving from the `pdbs` column.
- `--alphafold_pdb_dir` is used when `pdb_type == predicted` and `pdb_source == AlphaFold`, resolving `AF-<pdbs>-F1-model_v*.pdb` and choosing the highest version.
- `--esm3_pdb_dir` is used when `pdb_type == predicted` and `pdb_source` is empty or `nan`, resolving `ESM3-open-small-<pdbs>.pdb`.
- If you explicitly request `pst` or `molformer` without a usable legacy feature directory, the cache builder will stop with a clear error.
- Multi-GPU retraining here means multiple independent single-GPU retrains running in parallel, which is usually the highest-throughput way to sweep many split/seed jobs.
