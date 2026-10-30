import argparse
from threadpoolctl import threadpool_limits
from .sampling import sample, write_pdb


def main():
    p = argparse.ArgumentParser(description="FlexNMA backbone ensemble sampling")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--samples", type=int, default=60)
    p.add_argument(
        "--spread",
        type=float,
        help="Target mean aligned pairwise C-alpha RMSD (angstrom); default: structure-predicted",
    )
    p.add_argument(
        "--restrain",
        default="",
        help="Softly restrained PDB residues, e.g. A:20-25,A:40",
    )
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    with threadpool_limits(limits=1):
        models, residues, target = sample(
            a.input, a.samples, a.spread, a.restrain, a.seed
        )
    write_pdb(a.output, models, residues)
    print(f"Wrote {len(models)} samples; requested spread {target:.3f} angstrom")


if __name__ == "__main__":
    main()
