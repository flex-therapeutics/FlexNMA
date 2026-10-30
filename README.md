# FlexNMA

CPU protein backbone sampling using internal-coordinate normal modes, pair-distance correction, soft region restraints, and calibrated ensemble spread. No trained neural network or GPU is required.

## Install and sample

```bash
pip install .
flexnma --input input.pdb --output outputs/ensemble.pdb --samples 60
```

Provide one continuous protein chain with standard residues and complete N/CA/C/O atoms. Output is a multi-model backbone PDB; side chains are not sampled. Backbone oxygen atoms follow their residue frames.

By default, FlexNMA predicts a spread target from the input elastic-network spectrum using the bundled calibration. Use `--spread 2.0` to request a mean aligned pairwise Cα RMSD of 2 Å, or `--restrain A:20-25,A:40` to restrain a region. Restraints are soft, not exact coordinate fixing. Calibration uses eight probe conformations; the realized spread of a finite ensemble can differ from the target. Extreme targets may exceed the available torsional amplitude range.

Use `--seed` for reproducibility. Sampling uses one CPU thread. This repository contains the sampler and its required calibration, without benchmark or analysis pipelines.
