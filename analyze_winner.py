import pickle
import neat
from trainer import build_neat_config
from game import FlappyGame


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
            src_name = {-1: "vel", -2: "i", -3: "j"}.get(src, f"h{src}")
            dst_name = {0: "output"}.get(dst, f"h{dst}")
            print(f"  {src_name} -> {dst_name}: weight = {conn.weight:.3f}")

    print("\nNode biases:")
    for key, node in winner.nodes.items():
        name = {0: "output"}.get(key, f"h{key}")
        print(f"  {name}: bias = {node.bias:.3f}")

    if is_recurrent:
        net = neat.nn.RecurrentNetwork.create(winner, config)
    else:
        net = neat.nn.FeedForwardNetwork.create(winner, config)

    print("\n" + "=" * 50)
    print("BEHAVIOR ANALYSIS")
    print("=" * 50)

    print("\nOutput for different observations (tanh activation, >0 means jump):")
    print(f"{'vel':>6} {'i':>6} {'j':>6} -> {'output':>8} {'action':>8}")
    print("-" * 45)

    test_cases = [
        (0.0, 0.5, 0.2), (0.0, 0.5, 0.0), (0.0, 0.5, -0.2),
        (0.5, 0.5, 0.1), (-0.5, 0.5, 0.1), (1.0, 0.5, -0.1),
        (0.0, 0.1, 0.05), (0.0, 0.9, 0.05),
    ]
    for vel, i, j in test_cases:
        output = net.activate((vel, i, j))
        action = "JUMP" if output[0] > 0 else "fall"
        print(f"{vel:>6.2f} {i:>6.2f} {j:>6.2f} -> {output[0]:>8.3f} {action:>8}")

    print("\n" + "=" * 50)
    print("LEARNED DECISION BOUNDARY")
    print("=" * 50)

    print("\nFinding j threshold where action changes (vel=0, i=0.5):")
    for j in [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]:
        output = net.activate((0.0, 0.5, j))
        action = "JUMP" if output[0] > 0 else "fall"
        print(f"  j={j:+.2f}: output={output[0]:+.3f} -> {action}")

    print("\nSearching for exact decision boundary...")
    for j_int in range(-30, 30):
        j = j_int / 100.0
        output = net.activate((0.0, 0.5, j))
        if abs(output[0]) < 0.1:
            print(f"  Near boundary at j={j:.2f}: output={output[0]:.4f}")


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
        if is_recurrent:
            net = neat.nn.RecurrentNetwork.create(winner, config)
        else:
            net = neat.nn.FeedForwardNetwork.create(winner, config)
        game = FlappyGame(seed=seed)
        game.reset()
        for frame in range(50000):
            obs = game.get_observation()
            output = net.activate(obs)
            action = output[0] > 0
            _, _, done, score = game.step(action)
            if done:
                break
        scores.append(score)

    print(f"Scores on seeds 0-49: min={min(scores)}, max={max(scores)}, avg={sum(scores)/len(scores):.1f}")
    print(f"Score distribution: {sorted(scores)[:10]}...{sorted(scores)[-10:]}")

    print("\nEval seeds [42, 123, 456]:")
    for seed in [42, 123, 456]:
        if is_recurrent:
            net = neat.nn.RecurrentNetwork.create(winner, config)
        else:
            net = neat.nn.FeedForwardNetwork.create(winner, config)
        game = FlappyGame(seed=seed)
        game.reset()
        for frame in range(50000):
            obs = game.get_observation()
            output = net.activate(obs)
            action = output[0] > 0
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
