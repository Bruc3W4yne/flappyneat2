import pickle
from trainer import build_neat_config, create_network
from game import FlappyGame, network_to_action

# Input node labels: vel=velocity, dist=horizontal distance, y_off=vertical offset to gap
INPUT_LABELS = {-1: "vel", -2: "dist", -3: "y_off"}


def analyze_network(winner_path, experiment_name="neat_ff"):
    config = build_neat_config(experiment_name)
    is_recurrent = not config.genome_config.feed_forward
    with open(winner_path, "rb") as f:
        winner = pickle.load(f)

    print("=" * 50)
    print(f"NETWORK STRUCTURE ({'RNN' if is_recurrent else 'FF'})")
    print("=" * 50)
    print(f"Nodes: {len(winner.nodes)}")
    print(f"Connections: {len(winner.connections)}")

    print("\nConnections (enabled only):")
    for key, conn in winner.connections.items():
        if conn.enabled:
            src, dst = key
            src_name = INPUT_LABELS.get(src, f"h{src}")
            dst_name = {0: "jump"}.get(dst, f"h{dst}")
            print(f"  {src_name} -> {dst_name}: weight = {conn.weight:.3f}")

    print("\nNode biases:")
    for key, node in winner.nodes.items():
        name = {0: "output"}.get(key, f"h{key}")
        print(f"  {name}: bias = {node.bias:.3f}")

    net = create_network(winner, config, is_recurrent)

    print("\n" + "=" * 50)
    print("BEHAVIOR ANALYSIS")
    print("=" * 50)

    print("\nOutput for different observations (tanh activation, >0 means jump):")
    print(f"{'vel':>6} {'dist':>6} {'y_off':>6} -> {'output':>8} {'action':>8}")
    print("-" * 45)

    test_cases = [
        (0.0, 0.5, 0.2), (0.0, 0.5, 0.0), (0.0, 0.5, -0.2),
        (0.5, 0.5, 0.1), (-0.5, 0.5, 0.1), (1.0, 0.5, -0.1),
        (0.0, 0.1, 0.05), (0.0, 0.9, 0.05),
    ]
    for vel, dist, y_off in test_cases:
        output = net.activate((vel, dist, y_off))
        action = "JUMP" if network_to_action(output) else "fall"
        print(f"{vel:>6.2f} {dist:>6.2f} {y_off:>6.2f} -> {output[0]:>8.3f} {action:>8}")

    print("\n" + "=" * 50)
    print("LEARNED DECISION BOUNDARY")
    print("=" * 50)

    print("\nFinding y_off threshold where action changes (vel=0, dist=0.5):")
    for y_off in [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]:
        output = net.activate((0.0, 0.5, y_off))
        action = "JUMP" if network_to_action(output) else "fall"
        print(f"  y_off={y_off:+.2f}: output={output[0]:+.3f} -> {action}")

    print("\nSearching for exact decision boundary...")
    for y_int in range(-30, 30):
        y_off = y_int / 100.0
        output = net.activate((0.0, 0.5, y_off))
        if abs(output[0]) < 0.1:
            print(f"  Near boundary at y_off={y_off:.2f}: output={output[0]:.4f}")


def test_on_seeds(winner_path, experiment_name="neat_ff"):
    config = build_neat_config(experiment_name)
    is_recurrent = not config.genome_config.feed_forward
    with open(winner_path, "rb") as f:
        winner = pickle.load(f)

    print("\n" + "=" * 50)
    print("TESTING ON VARIOUS SEEDS")
    print("=" * 50)

    scores = []
    for seed in range(50):
        net = create_network(winner, config, is_recurrent)
        game = FlappyGame(seed=seed)
        game.reset()
        for frame in range(50000):
            obs = game.get_observation()
            output = net.activate(obs)
            action = network_to_action(output)
            _, _, done, score = game.step(action)
            if done:
                break
        scores.append(score)

    print(f"Scores on seeds 0-49: min={min(scores)}, max={max(scores)}, avg={sum(scores)/len(scores):.1f}")
    print(f"Score distribution: {sorted(scores)[:10]}...{sorted(scores)[-10:]}")

    print("\nEval seeds [42, 123, 456]:")
    for seed in [42, 123, 456]:
        net = create_network(winner, config, is_recurrent)
        game = FlappyGame(seed=seed)
        game.reset()
        for frame in range(50000):
            obs = game.get_observation()
            output = net.activate(obs)
            action = network_to_action(output)
            _, _, done, score = game.step(action)
            if done:
                print(f"  Seed {seed}: score={score}, died at frame {frame}")
                break
        else:
            print(f"  Seed {seed}: score={score}, SURVIVED!")


if __name__ == "__main__":
    import sys
    winner_path = sys.argv[1] if len(sys.argv) > 1 else "winner_neat_ff.pkl"
    experiment = sys.argv[2] if len(sys.argv) > 2 else "neat_ff"
    try:
        analyze_network(winner_path, experiment)
        test_on_seeds(winner_path, experiment)
    except FileNotFoundError:
        print(f"Winner file not found: {winner_path}")
        print("Usage: python analyze_winner.py <winner.pkl> [experiment_name]")
