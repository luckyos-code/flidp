import argparse

from experiments import run_emnist, run_svhn

parser = argparse.ArgumentParser(
    prog="flidp",
)

parser.add_argument("--dir", type=str, required=True)
parser.add_argument("--dataset", choices=["emnist", "svhn"], required=True)
parser.add_argument("--budgets", nargs='*', type=float, required=True)
parser.add_argument("--ratios", nargs='*', type=float, required=True)

def check_args(args):
    if not len(args.budgets) == len(args.ratios):
        raise Exception("budgets and ratios must have the same lengths")
    if not sum(args.ratios) == 1.0:
        raise Exception("ratios must sum up to 1.0")
    return args


def main():
    args = check_args(parser.parse_args())
    if args.dataset == "emnist":        
        run_emnist(args.dir, args.budgets, args.ratios)
    elif args.dataset == "svhn":
        run_svhn(args.dir, args.budgets, args.ratios)
    else:
        raise NotImplementedError(f"Currently there is no implementation for {args.dataset}")

if __name__ == "__main__":
    main()