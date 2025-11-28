import os
import glob
import pickle
import matplotlib.pyplot as plt
import numpy as np

NETWORKS_DIR = "results/networks"

INPUT_LABELS = ["vel", "dist", "y_off"]
OUTPUT_LABELS = ["jump"]


def load_winner(winner_path):
    with open(winner_path, "rb") as f:
        return pickle.load(f)


def export_neat_network(winner, config, output_path):
    try:
        import graphviz
    except ImportError:
        print("graphviz not installed. Install with: pip install graphviz")
        return False

    dot = graphviz.Digraph(format='png')
    dot.attr(rankdir='LR', splines='true', nodesep='0.5', ranksep='1.5')

    input_nodes = list(config.genome_config.input_keys)
    output_nodes = list(config.genome_config.output_keys)
    hidden_nodes = [k for k in winner.nodes.keys() if k not in input_nodes + output_nodes]

    with dot.subgraph(name='cluster_inputs') as c:
        c.attr(label='Inputs', style='dashed')
        for i, node in enumerate(input_nodes):
            label = INPUT_LABELS[i] if i < len(INPUT_LABELS) else f"in_{node}"
            c.node(str(node), label, shape='circle', style='filled', fillcolor='lightblue')

    with dot.subgraph(name='cluster_outputs') as c:
        c.attr(label='Outputs', style='dashed')
        for i, node in enumerate(output_nodes):
            label = OUTPUT_LABELS[i] if i < len(OUTPUT_LABELS) else f"out_{node}"
            c.node(str(node), label, shape='circle', style='filled', fillcolor='lightyellow')

    if hidden_nodes:
        with dot.subgraph(name='cluster_hidden') as c:
            c.attr(label='Hidden', style='dashed')
            for node in hidden_nodes:
                c.node(str(node), f"h{node}", shape='circle', style='filled', fillcolor='lightgray')

    for (src, dst), conn in winner.connections.items():
        if not conn.enabled:
            continue
        weight = conn.weight
        color = 'green' if weight > 0 else 'red'
        penwidth = str(min(3.0, max(0.5, abs(weight))))
        dot.edge(str(src), str(dst), color=color, penwidth=penwidth)

    try:
        dot.render(output_path.replace('.png', ''), cleanup=True)
        return True
    except Exception as e:
        print(f"Error rendering {output_path}: {e}")
        return False


def export_static_network(genome_data, output_path):
    hidden_layers = genome_data["hidden_layers"]
    is_recurrent = genome_data["is_recurrent"]
    weights = genome_data["genome"]

    layer_sizes = [3] + hidden_layers + [1]
    num_layers = len(layer_sizes)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(-0.5, num_layers - 0.5)
    max_nodes = max(layer_sizes)
    ax.set_ylim(-0.5, max_nodes + 0.5)
    ax.axis('off')

    node_positions = {}
    for layer_idx, size in enumerate(layer_sizes):
        y_offset = (max_nodes - size) / 2
        for node_idx in range(size):
            x = layer_idx
            y = y_offset + node_idx
            node_positions[(layer_idx, node_idx)] = (x, y)

    weight_idx = 0
    for layer in range(num_layers - 1):
        in_size = layer_sizes[layer]
        out_size = layer_sizes[layer + 1]
        for out_idx in range(out_size):
            for in_idx in range(in_size):
                if weight_idx < len(weights):
                    w = weights[weight_idx]
                    weight_idx += 1
                    x1, y1 = node_positions[(layer, in_idx)]
                    x2, y2 = node_positions[(layer + 1, out_idx)]
                    color = 'green' if w > 0 else 'red'
                    alpha = min(1.0, abs(w) / 5)
                    ax.plot([x1, x2], [y1, y2], color=color, alpha=alpha, linewidth=0.5)

    layer_labels = [INPUT_LABELS, *[[f"h{i}" for i in range(s)] for s in hidden_layers], OUTPUT_LABELS]
    layer_colors = ['lightblue'] + ['lightgray'] * len(hidden_layers) + ['lightyellow']

    for layer_idx, size in enumerate(layer_sizes):
        color = layer_colors[layer_idx]
        labels = layer_labels[layer_idx] if layer_idx < len(layer_labels) else []
        for node_idx in range(size):
            x, y = node_positions[(layer_idx, node_idx)]
            circle = plt.Circle((x, y), 0.15, color=color, ec='black', zorder=10)
            ax.add_patch(circle)
            label = labels[node_idx] if node_idx < len(labels) else f"{node_idx}"
            ax.text(x, y, label, ha='center', va='center', fontsize=8, zorder=11)

    title = f"Static {'RNN' if is_recurrent else 'FF'}: {layer_sizes}"
    ax.set_title(title, fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    return True


def plot_weight_distribution(genome_data, output_path):
    if isinstance(genome_data, tuple):
        winner, config = genome_data
        weights = [c.weight for c in winner.connections.values() if c.enabled]
        biases = [n.bias for n in winner.nodes.values()]
        title = "NEAT Network"
    elif isinstance(genome_data, dict):
        weights = genome_data["genome"]
        biases = []
        title = "Static GA Network"
    else:
        return False

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    if weights:
        axes[0].hist(weights, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        axes[0].axvline(0, color='red', linestyle='--', alpha=0.5)
        axes[0].set_xlabel("Weight Value")
        axes[0].set_ylabel("Frequency")
        axes[0].set_title(f"{title} - Weight Distribution")
        axes[0].text(0.95, 0.95, f"n={len(weights)}\nmean={np.mean(weights):.2f}\nstd={np.std(weights):.2f}",
                     transform=axes[0].transAxes, ha='right', va='top', fontsize=9,
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    if biases:
        axes[1].hist(biases, bins=20, color='coral', edgecolor='black', alpha=0.7)
        axes[1].axvline(0, color='red', linestyle='--', alpha=0.5)
        axes[1].set_xlabel("Bias Value")
        axes[1].set_ylabel("Frequency")
        axes[1].set_title(f"{title} - Bias Distribution")
    else:
        axes[1].text(0.5, 0.5, "No separate biases\n(embedded in weights)",
                     ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title("Bias Distribution")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    return True


def get_network_stats(genome_data):
    stats = {}
    if isinstance(genome_data, tuple):
        winner, config = genome_data
        stats["type"] = "NEAT"
        stats["nodes"] = len(winner.nodes)
        stats["connections"] = sum(1 for c in winner.connections.values() if c.enabled)
        stats["disabled"] = sum(1 for c in winner.connections.values() if not c.enabled)
        weights = [c.weight for c in winner.connections.values() if c.enabled]
        stats["weight_mean"] = np.mean(weights) if weights else 0
        stats["weight_std"] = np.std(weights) if weights else 0
    elif isinstance(genome_data, dict):
        stats["type"] = "Static GA"
        stats["hidden_layers"] = genome_data["hidden_layers"]
        stats["is_recurrent"] = genome_data["is_recurrent"]
        stats["parameters"] = len(genome_data["genome"])
        stats["weight_mean"] = np.mean(genome_data["genome"])
        stats["weight_std"] = np.std(genome_data["genome"])
    return stats


def main():
    os.makedirs(NETWORKS_DIR, exist_ok=True)

    winner_files = glob.glob("winner_*_trial*.pkl")
    if not winner_files:
        print("No winner files found (winner_*_trial*.pkl)")
        print("Run experiments first: python trainer.py --run-all")
        return

    print(f"Found {len(winner_files)} winner files")

    stats_all = []
    for winner_path in sorted(winner_files):
        exp_name = winner_path.replace("winner_", "").replace(".pkl", "")
        print(f"\nProcessing: {exp_name}")

        data = load_winner(winner_path)
        stats = get_network_stats(data)
        stats["experiment"] = exp_name
        stats_all.append(stats)

        if isinstance(data, tuple):
            winner, config = data
            network_path = os.path.join(NETWORKS_DIR, f"{exp_name}_topology.png")
            if export_neat_network(winner, config, network_path):
                print(f"  Saved topology: {network_path}")

            dist_path = os.path.join(NETWORKS_DIR, f"{exp_name}_weights.png")
            if plot_weight_distribution(data, dist_path):
                print(f"  Saved weights: {dist_path}")

            print(f"  Stats: {stats['nodes']} nodes, {stats['connections']} connections")

        elif isinstance(data, dict):
            network_path = os.path.join(NETWORKS_DIR, f"{exp_name}_topology.png")
            if export_static_network(data, network_path):
                print(f"  Saved topology: {network_path}")

            dist_path = os.path.join(NETWORKS_DIR, f"{exp_name}_weights.png")
            if plot_weight_distribution(data, dist_path):
                print(f"  Saved weights: {dist_path}")

            print(f"  Stats: {stats['parameters']} parameters, layers={stats['hidden_layers']}")

    print("\n" + "=" * 60)
    print("NETWORK STATISTICS SUMMARY")
    print("=" * 60)

    neat_stats = [s for s in stats_all if s.get("type") == "NEAT"]
    if neat_stats:
        print("\nNEAT Networks:")
        for s in neat_stats:
            print(f"  {s['experiment']}: {s['nodes']} nodes, {s['connections']} conn")

    ga_stats = [s for s in stats_all if s.get("type") == "Static GA"]
    if ga_stats:
        print("\nStatic GA Networks:")
        for s in ga_stats:
            rnn = " (RNN)" if s.get("is_recurrent") else ""
            print(f"  {s['experiment']}: {s['parameters']} params{rnn}")

    print(f"\nAll exports saved to {NETWORKS_DIR}/")


if __name__ == "__main__":
    main()
