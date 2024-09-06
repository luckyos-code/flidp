import argparse

from experiments import run_emnist, run_svhn, run_cifar100, run_cifar10
from helpers import setup_tff_runtime

parser = argparse.ArgumentParser(
    prog="flidp",
)

parser.add_argument("--save-dir", type=str, required=True)
parser.add_argument("--dataset", choices=["emnist", "svhn", "cifar10", "cifar100"], required=True)
parser.add_argument("--make-iid", action='store_true', default=False)
parser.add_argument("--model", choices=["simple-cnn", "lucasnet", "efficientnet", "mobilenet"], required=True)
parser.add_argument("--budgets", nargs='*', type=float, default=[])
parser.add_argument("--ratios", nargs='*', type=float, default=[])
parser.add_argument("--rounds", type=int, required=True)
parser.add_argument("--clients-per-round", type=int, required=True)
parser.add_argument("--local-epochs", type=int, required=True)
parser.add_argument("--batch-size", type=int, required=True)
parser.add_argument("--client-lr", type=float, required=True)
parser.add_argument("--server-lr", type=float, required=True)

def infer_dp_level(budgets):
    if not budgets:
        return 'nodp'
    if len(budgets) == 1:
        return 'dp'
    else:
        return 'idp'

def check_args(args):
    if not len(args.budgets) == len(args.ratios):
        raise Exception(f"budgets ({args.budgets}) and ratios ({args.ratios}) must have the same lengths")
    if len(args.ratios) > 0 and not sum(args.ratios) == 1.0:
        raise Exception(f"ratios must sum up to 1.0 (actual sum: {sum(args.ratios)})")
    return args


def main():
    args = parser.parse_args()
    args = check_args(args)
    dp_level = infer_dp_level(args.budgets)
    print(f"dp level was set to {dp_level}.")
    args = vars(args)
    
    dataset = args.pop("dataset")
    
    # setup GPU devices etc
    # setup_tff_runtime()
    
    if dataset == "emnist":        
        run_emnist(**args, dp_level=dp_level)
    elif dataset == "svhn":
        run_svhn(**args, dp_level=dp_level)
    elif dataset == "cifar100":
        run_cifar100(**args, dp_level=dp_level)
    elif dataset == "cifar10":
        run_cifar10(**args, dp_level=dp_level)
    else:
        raise NotImplementedError(f"Currently there is no implementation for {dataset}")

if __name__ == "__main__":
    main()
