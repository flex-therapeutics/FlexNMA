# FlexNMA

CPU protein backbone sampling with internal-coordinate normal modes, pair-distance correction, soft region restraints and calibrated ensemble spread. No neural-network weights or GPU are required.

## Install and sample

Use Python 3.10 or newer:

```bash
git clone https://github.com/flex-therapeutics/FlexNMA.git
cd FlexNMA
pip install .
flexnma --input protein.pdb --output ensemble.pdb --samples 60
```

Provide one continuous protein chain with standard residues and complete N/CA/C/O atoms. Output is a multi-model backbone PDB; side chains are not sampled.

## Control spread and restrain a region

```bash
flexnma --input protein.pdb --output ensemble.pdb --samples 60 \
  --spread 2.0 --restrain A:20-25,A:40 --seed 2026
```

- `--spread 2.0`: target a mean aligned pairwise Cα RMSD of 2 Å. If omitted, spread is predicted from the input elastic-network spectrum using the bundled calibration.
- `--restrain A:20-25,A:40`: softly restrain chain A residues 20–25 and 40, using input PDB numbering. Restraints do not fix coordinates exactly.
- `--seed`: control reproducibility. Sampling uses one CPU thread.

Calibration uses eight probe conformations; finite-ensemble spread can differ from the target. Extreme spread targets may exceed the available torsional range. This repository contains sampling code and calibration only.
