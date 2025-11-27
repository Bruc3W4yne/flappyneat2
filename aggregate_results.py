import os
import csv
import glob
from collections import defaultdict

def load_csv(filepath):
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        return list(reader)

def aggregate_results(results_dir="results"):
    if not os.path.exists(results_dir):
        print(f"No results directory found: {results_dir}")
        return

    files = glob.glob(os.path.join(results_dir, "*_trial*.csv"))
    if not files:
        print("No trial CSV files found")
        return

    experiments = defaultdict(list)
    for f in files:
        basename = os.path.basename(f)
        exp_name = basename.rsplit("_trial", 1)[0]
        experiments[exp_name].append(f)

    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Experiment':<15} {'Trials':<8} {'Final Best':<20} {'Final Avg':<20}")
    print("-" * 70)

    summary_data = []
    for exp_name, trial_files in sorted(experiments.items()):
        final_bests = []
        final_avgs = []
        for filepath in trial_files:
            data = load_csv(filepath)
            if data:
                last_row = data[-1]
                final_bests.append(float(last_row["best_fitness"]))
                final_avgs.append(float(last_row["avg_fitness"]))

        if final_bests:
            mean_best = sum(final_bests) / len(final_bests)
            std_best = (sum((x - mean_best)**2 for x in final_bests) / len(final_bests)) ** 0.5
            mean_avg = sum(final_avgs) / len(final_avgs)
            std_avg = (sum((x - mean_avg)**2 for x in final_avgs) / len(final_avgs)) ** 0.5

            best_str = f"{mean_best:.2f} ± {std_best:.2f}"
            avg_str = f"{mean_avg:.2f} ± {std_avg:.2f}"
            print(f"{exp_name:<15} {len(trial_files):<8} {best_str:<20} {avg_str:<20}")
            summary_data.append([exp_name, len(trial_files), mean_best, std_best, mean_avg, std_avg])

    print("=" * 70)

    summary_path = os.path.join(results_dir, "summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["experiment", "trials", "mean_best", "std_best", "mean_avg", "std_avg"])
        writer.writerows(summary_data)
    print(f"\nSaved to {summary_path}")


if __name__ == "__main__":
    aggregate_results()
