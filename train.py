import argparse
from trainer import run_experiment
from config import EXPERIMENTS

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=list(EXPERIMENTS.keys()) + ["all"])
    parser.add_argument("-g", "--generations", type=int, default=50)
    parser.add_argument("-n", "--trials", type=int, default=1)
    args = parser.parse_args()

    experiments = list(EXPERIMENTS.keys()) if args.experiment == "all" else [args.experiment]

    for exp in experiments:
        for trial in range(args.trials):
            print(f"\n=== {exp} (trial {trial + 1}/{args.trials}) ===")
            run_experiment(exp, args.generations, trial_id=trial)
