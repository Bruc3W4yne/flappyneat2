import os
import csv
import glob
from collections import defaultdict
import math

ARCHITECTURES = ["neat_ff", "neat_rnn", "static_ff", "static_rnn"]
PARAM_SETS = ["low", "medium", "high"]


def load_csv(filepath):
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        return list(reader)


def mean(values):
    return sum(values) / len(values) if values else 0


def std(values):
    if len(values) < 2:
        return 0
    m = mean(values)
    return (sum((x - m) ** 2 for x in values) / len(values)) ** 0.5


def cohens_d(group1, group2):
    if not group1 or not group2:
        return 0
    n1, n2 = len(group1), len(group2)
    m1, m2 = mean(group1), mean(group2)
    s1, s2 = std(group1), std(group2)
    pooled_std = math.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2)) if n1 + n2 > 2 else 1
    return (m1 - m2) / pooled_std if pooled_std > 0 else 0


def mann_whitney_u(x, y):
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return None, None

    all_values = [(v, 0, i) for i, v in enumerate(x)] + [(v, 1, i) for i, v in enumerate(y)]
    all_values.sort()

    ranks = {}
    i = 0
    while i < len(all_values):
        j = i
        while j < len(all_values) and all_values[j][0] == all_values[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2
        for k in range(i, j):
            group, idx = all_values[k][1], all_values[k][2]
            ranks[(group, idx)] = avg_rank
        i = j

    r1 = sum(ranks[(0, i)] for i in range(nx))
    u1 = r1 - nx * (nx + 1) / 2

    mu = nx * ny / 2
    sigma = math.sqrt(nx * ny * (nx + ny + 1) / 12) if nx + ny > 1 else 1
    z = (u1 - mu) / sigma if sigma > 0 else 0

    p = 2 * (1 - _norm_cdf(abs(z)))
    return u1, p


def _norm_cdf(x):
    return (1 + math.erf(x / math.sqrt(2))) / 2


def kruskal_wallis(groups):
    all_values = []
    for i, group in enumerate(groups):
        for v in group:
            all_values.append((v, i))
    all_values.sort()

    N = len(all_values)
    if N < 2:
        return None, None

    ranks = []
    i = 0
    while i < N:
        j = i
        while j < N and all_values[j][0] == all_values[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2
        for k in range(i, j):
            ranks.append((avg_rank, all_values[k][1]))
        i = j

    group_ranks = defaultdict(list)
    for rank, group_idx in ranks:
        group_ranks[group_idx].append(rank)

    k = len(groups)
    H = 0
    for i, group in enumerate(groups):
        ni = len(group)
        if ni > 0:
            ri_mean = mean(group_ranks[i])
            H += ni * (ri_mean - (N + 1) / 2) ** 2

    H = 12 / (N * (N + 1)) * H if N > 1 else 0

    df = k - 1
    p = 1 - _chi2_cdf(H, df) if df > 0 else 1
    return H, p


def _chi2_cdf(x, df):
    if x <= 0:
        return 0
    return _incomplete_gamma(df / 2, x / 2) / math.gamma(df / 2)


def _incomplete_gamma(a, x, iterations=100):
    if x == 0:
        return 0
    if x < 0:
        return 0

    if x < a + 1:
        term = 1 / a
        total = term
        for n in range(1, iterations):
            term *= x / (a + n)
            total += term
            if abs(term) < 1e-10:
                break
        return total * math.exp(-x + a * math.log(x) - math.lgamma(a))
    else:
        return math.gamma(a) - _incomplete_gamma_upper(a, x, iterations)


def _incomplete_gamma_upper(a, x, iterations=100):
    f = 1 + x - a
    c = 1 / 1e-30
    d = 1 / f
    h = d

    for i in range(1, iterations):
        an = -i * (i - a)
        bn = f + 2 * i
        d = bn + an * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = bn + an / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-10:
            break

    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def aggregate_results(results_dir="results"):
    if not os.path.exists(results_dir):
        print(f"No results directory found: {results_dir}")
        return

    files = glob.glob(os.path.join(results_dir, "*_trial*.csv"))
    files = [f for f in files if "test_results" not in f and "summary" not in f]
    if not files:
        print("No trial CSV files found")
        return

    experiments = defaultdict(list)
    for f in files:
        basename = os.path.basename(f)
        exp_name = basename.rsplit("_trial", 1)[0]
        experiments[exp_name].append(f)

    exp_final_bests = {}
    print("=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Experiment':<20} {'Trials':<8} {'Final Best':<22} {'Final Avg':<22}")
    print("-" * 80)

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
            exp_final_bests[exp_name] = final_bests
            mean_best = mean(final_bests)
            std_best = std(final_bests)
            mean_avg = mean(final_avgs)
            std_avg = std(final_avgs)

            best_str = f"{mean_best:.2f} ± {std_best:.2f}"
            avg_str = f"{mean_avg:.2f} ± {std_avg:.2f}"
            print(f"{exp_name:<20} {len(trial_files):<8} {best_str:<22} {avg_str:<22}")
            summary_data.append([exp_name, len(trial_files), mean_best, std_best, mean_avg, std_avg])

    print("=" * 80)

    summary_path = os.path.join(results_dir, "summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["experiment", "trials", "mean_best", "std_best", "mean_avg", "std_avg"])
        writer.writerows(summary_data)
    print(f"\nSaved to {summary_path}")

    print("\n" + "=" * 80)
    print("STATISTICAL ANALYSIS")
    print("=" * 80)

    print("\n--- Architecture Comparison (Kruskal-Wallis) ---")
    arch_groups = {}
    for arch in ARCHITECTURES:
        arch_data = []
        for param in PARAM_SETS:
            exp_name = f"{arch}_{param}"
            if exp_name in exp_final_bests:
                arch_data.extend(exp_final_bests[exp_name])
        if arch_data:
            arch_groups[arch] = arch_data
            print(f"{arch}: n={len(arch_data)}, mean={mean(arch_data):.2f} ± {std(arch_data):.2f}")

    if len(arch_groups) >= 2:
        groups = [arch_groups[a] for a in ARCHITECTURES if a in arch_groups]
        H, p = kruskal_wallis(groups)
        if H is not None:
            print(f"\nKruskal-Wallis H = {H:.3f}, p = {p:.4f}")
            print("Significant difference" if p < 0.05 else "No significant difference")

    print("\n--- Parameter Set Comparison (Kruskal-Wallis) ---")
    param_groups = {}
    for param in PARAM_SETS:
        param_data = []
        for arch in ARCHITECTURES:
            exp_name = f"{arch}_{param}"
            if exp_name in exp_final_bests:
                param_data.extend(exp_final_bests[exp_name])
        if param_data:
            param_groups[param] = param_data
            print(f"{param}: n={len(param_data)}, mean={mean(param_data):.2f} ± {std(param_data):.2f}")

    if len(param_groups) >= 2:
        groups = [param_groups[p] for p in PARAM_SETS if p in param_groups]
        H, p = kruskal_wallis(groups)
        if H is not None:
            print(f"\nKruskal-Wallis H = {H:.3f}, p = {p:.4f}")
            print("Significant difference" if p < 0.05 else "No significant difference")

    print("\n--- Pairwise Comparisons (Mann-Whitney U, Cohen's d) ---")
    comparisons = [
        ("neat_ff", "static_ff", "NEAT FF vs Static FF"),
        ("neat_rnn", "static_rnn", "NEAT RNN vs Static RNN"),
        ("neat_ff", "neat_rnn", "NEAT FF vs NEAT RNN"),
        ("static_ff", "static_rnn", "Static FF vs Static RNN"),
    ]

    for arch1, arch2, label in comparisons:
        if arch1 in arch_groups and arch2 in arch_groups:
            g1, g2 = arch_groups[arch1], arch_groups[arch2]
            u, p = mann_whitney_u(g1, g2)
            d = cohens_d(g1, g2)
            sig = "*" if p and p < 0.05 else ""
            effect = "large" if abs(d) > 0.8 else "medium" if abs(d) > 0.5 else "small"
            print(f"{label}: U={u:.1f}, p={p:.4f}{sig}, d={d:.2f} ({effect})")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    aggregate_results()
