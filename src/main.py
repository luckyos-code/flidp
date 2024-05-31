import argparse

from experiments import run_emnist, run_svhn

parser = argparse.ArgumentParser(
    prog="flidp",
)

parser.add_argument("--dir", type=str, required=True)
parser.add_argument("--dataset", choices=["emnist", "svhn"], required=True)
parser.add_argument("--budgets", nargs='*', type=float, default=[])
parser.add_argument("--ratios", nargs='*', type=float, default=[])

def infer_dp_level(budgets):
    if not budgets:
        return 'nodp'
    if len(budgets) == 1:
        return 'dp'
    else:
        return 'idp'

def check_args(args):
    if not len(args.budgets) == len(args.ratios):
        raise Exception("budgets and ratios must have the same lengths")
    if len(args.ratios) > 0 and not sum(args.ratios) == 1.0:
        raise Exception("ratios must sum up to 1.0")
    return args


def main():
    args = parser.parse_args()
    args = check_args(args)
    dp_level = infer_dp_level(args.budgets)
    print(f"dp level was set to {dp_level}.")
    if args.dataset == "emnist":        
        run_emnist(args.dir, args.budgets, args.ratios, dp_level)
    elif args.dataset == "svhn":
        run_svhn(args.dir, args.budgets, args.ratios, dp_level)
    else:
        raise NotImplementedError(f"Currently there is no implementation for {args.dataset}")

if __name__ == "__main__":
    main()