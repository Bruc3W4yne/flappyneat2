import os
import csv
import glob
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = "results"
FIGURES_DIR = "results/figures"

ARCHITECTURES = ["neat_ff", "neat_rnn", "static_ff", "static_rnn"]
PARAM_SETS = ["low", "medium", "high"]

ARCH_COLORS = {
    "neat_ff": "#1f77b4",
    "neat_rnn": "#ff7f0e",
    "static_ff": "#2ca02c",
    "static_rnn": "#d62728"
}

PARAM_STYLES = {
    "low": "--",
    "medium": "-.",
    "high": "-"
}


def load_training_data():
    data = defaultdict(list)
    csv_files = glob.glob(os.path.join(RESULTS_DIR, "*_trial*.csv"))

    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        if filename == "test_results.csv" or filename == "summary.csv":
            continue

        parts = filename.replace(".csv", "").rsplit("_trial", 1)
        if len(parts) != 2:
            continue
        exp_name = parts[0]

        generations, best_fitness, avg_fitness = [], [], []
        with open(csv_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                generations.append(int(row["generation"]))
                best_fitness.append(float(row["best_fitness"]))
                avg_fitness.append(float(row["avg_fitness"]))

        if generations:
            data[exp_name].append({
                "generations": generations,
                "best_fitness": best_fitness,
                "avg_fitness": avg_fitness
            })

    return data


def compute_mean_std(trials_data, key="best_fitness"):
    if not trials_data:
        return [], [], []

    max_gen = max(len(t["generations"]) for t in trials_data)
    all_values = []

    for gen in range(max_gen):
        gen_values = []
        for trial in trials_data:
            if gen < len(trial[key]):
                gen_values.append(trial[key][gen])
        all_values.append(gen_values)

    generations = list(range(max_gen))
    means = [np.mean(v) if v else 0 for v in all_values]
    stds = [np.std(v) if len(v) > 1 else 0 for v in all_values]

    return generations, means, stds


def plot_learning_curves(data, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, arch in enumerate(ARCHITECTURES):
        ax = axes[idx]
        ax.set_title(f"{arch.replace('_', ' ').upper()}", fontsize=12, fontweight='bold')

        for param in PARAM_SETS:
            exp_name = f"{arch}_{param}"
            if exp_name not in data:
                continue

            gens, means, stds = compute_mean_std(data[exp_name])
            if not gens:
                continue

            means, stds = np.array(means), np.array(stds)
            color = ARCH_COLORS[arch]
            style = PARAM_STYLES[param]

            ax.plot(gens, means, style, color=color, label=param, linewidth=2)
            if len(data[exp_name]) > 1:
                ax.fill_between(gens, means - stds, means + stds, color=color, alpha=0.2)

        ax.set_xlabel("Generation")
        ax.set_ylabel("Best Fitness (Pipes Passed)")
        ax.legend(title="Mutation Rate")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_architecture_comparison(data, output_path):
    fig, ax = plt.subplots(figsize=(12, 6))

    for arch in ARCHITECTURES:
        all_trials = []
        for param in PARAM_SETS:
            exp_name = f"{arch}_{param}"
            if exp_name in data:
                all_trials.extend(data[exp_name])

        if not all_trials:
            continue

        gens, means, stds = compute_mean_std(all_trials)
        if not gens:
            continue

        means, stds = np.array(means), np.array(stds)
        color = ARCH_COLORS[arch]

        label = arch.replace("_", " ").upper()
        ax.plot(gens, means, color=color, label=label, linewidth=2)
        ax.fill_between(gens, means - stds, means + stds, color=color, alpha=0.2)

    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Best Fitness (Pipes Passed)", fontsize=12)
    ax.set_title("Architecture Comparison (All Parameter Sets)", fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_parameter_sensitivity(data, output_path):
    fig, ax = plt.subplots(figsize=(12, 6))

    param_colors = {"low": "#3498db", "medium": "#f39c12", "high": "#e74c3c"}

    for param in PARAM_SETS:
        all_trials = []
        for arch in ARCHITECTURES:
            exp_name = f"{arch}_{param}"
            if exp_name in data:
                all_trials.extend(data[exp_name])

        if not all_trials:
            continue

        gens, means, stds = compute_mean_std(all_trials)
        if not gens:
            continue

        means, stds = np.array(means), np.array(stds)
        color = param_colors[param]

        ax.plot(gens, means, color=color, label=f"{param.upper()} mutation", linewidth=2)
        ax.fill_between(gens, means - stds, means + stds, color=color, alpha=0.2)

    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Best Fitness (Pipes Passed)", fontsize=12)
    ax.set_title("Parameter Sensitivity (All Architectures)", fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_final_performance(data, output_path):
    experiments = []
    final_means = []
    final_stds = []
    colors = []

    for arch in ARCHITECTURES:
        for param in PARAM_SETS:
            exp_name = f"{arch}_{param}"
            if exp_name not in data or not data[exp_name]:
                continue

            final_values = []
            for trial in data[exp_name]:
                if trial["best_fitness"]:
                    final_values.append(trial["best_fitness"][-1])

            if final_values:
                experiments.append(exp_name)
                final_means.append(np.mean(final_values))
                final_stds.append(np.std(final_values) if len(final_values) > 1 else 0)
                colors.append(ARCH_COLORS[arch])

    if not experiments:
        print("No data for final performance plot")
        return

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(experiments))

    bars = ax.bar(x, final_means, yerr=final_stds, capsize=3, color=colors, alpha=0.8, edgecolor='black')

    ax.set_xlabel("Experiment", fontsize=12)
    ax.set_ylabel("Final Best Fitness", fontsize=12)
    ax.set_title("Final Performance Comparison", fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(experiments, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=ARCH_COLORS[arch], label=arch.replace("_", " ").upper())
                       for arch in ARCHITECTURES]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("Loading training data...")
    data = load_training_data()

    if not data:
        print("No training CSV files found in results/")
        print("Run experiments first: python trainer.py --run-all")
        return

    print(f"Found {len(data)} experiments with data")
    for exp, trials in sorted(data.items()):
        print(f"  {exp}: {len(trials)} trial(s)")

    print("\nGenerating plots...")
    plot_learning_curves(data, os.path.join(FIGURES_DIR, "learning_curves.png"))
    plot_architecture_comparison(data, os.path.join(FIGURES_DIR, "architecture_comparison.png"))
    plot_parameter_sensitivity(data, os.path.join(FIGURES_DIR, "parameter_sensitivity.png"))
    plot_final_performance(data, os.path.join(FIGURES_DIR, "final_performance.png"))

    print(f"\nAll figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
