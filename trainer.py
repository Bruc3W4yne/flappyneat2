import os
import csv
import copy
import pickle
import random
import tempfile
import math
import neat

from game import FlappyGame, play_game
from config import BASE_CONFIG, EXPERIMENTS, SHARED_CONFIG, GA_CONFIG, TRAIN_SEEDS, TEST_SEEDS, MAX_FRAMES_EVAL, PARAM_SETS


class CSVReporter(neat.reporting.BaseReporter):
    def __init__(self, filename):
        self.filename = filename
        self.generation = 0
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else ".", exist_ok=True)
        with open(filename, "w", newline="") as f:
            csv.writer(f).writerow(["generation", "best_fitness", "avg_fitness", "min_fitness"])

    def post_evaluate(self, config, population, species, best_genome):
        fitnesses = [g.fitness for g in population.values()]
        with open(self.filename, "a", newline="") as f:
            csv.writer(f).writerow([self.generation, max(fitnesses), sum(fitnesses)/len(fitnesses), min(fitnesses)])
        self.generation += 1


def dict_to_config_file(config_dict):
    lines = []
    for section, params in config_dict.items():
        lines.append(f"[{section}]")
        for key, value in params.items():
            lines.append(f"{key} = {value}")
        lines.append("")
    return "\n".join(lines)


def build_neat_config(experiment_name):
    exp = EXPERIMENTS[experiment_name]
    config_dict = copy.deepcopy(BASE_CONFIG)
    config_dict["DefaultGenome"]["feed_forward"] = str(exp["feed_forward"])

    param_set_name = exp.get("param_set", "high")
    param_set = PARAM_SETS[param_set_name]
    config_dict["DefaultGenome"]["weight_mutate_rate"] = str(param_set["weight_mutate_rate"])

    if not exp.get("use_neat", True):
        config_dict["DefaultGenome"]["node_add_prob"] = "0.0"
        config_dict["DefaultGenome"]["node_delete_prob"] = "0.0"
        config_dict["DefaultGenome"]["conn_add_prob"] = "0.0"
        config_dict["DefaultGenome"]["conn_delete_prob"] = "0.0"
        config_dict["DefaultGenome"]["enabled_mutate_rate"] = "0.0"
    else:
        config_dict["DefaultGenome"]["node_add_prob"] = str(param_set["node_add_prob"])
        config_dict["DefaultGenome"]["conn_add_prob"] = str(param_set["conn_add_prob"])

    config_text = dict_to_config_file(config_dict)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".cfg", delete=False) as f:
        f.write(config_text)
        config_path = f.name
    neat_config = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path
    )
    os.unlink(config_path)
    return neat_config


def create_network(genome, config, is_recurrent=None):
    if is_recurrent is None:
        is_recurrent = not config.genome_config.feed_forward
    if is_recurrent:
        return neat.nn.RecurrentNetwork.create(genome, config)
    return neat.nn.FeedForwardNetwork.create(genome, config)


def evaluate_genome_neat(genome, config, seeds, max_frames=MAX_FRAMES_EVAL):
    is_recurrent = not config.genome_config.feed_forward
    total_score = 0
    for seed in seeds:
        net = create_network(genome, config, is_recurrent)
        game = FlappyGame(seed=seed)
        score = play_game(net, game, max_frames=max_frames)
        total_score += score
    return total_score / len(seeds)


def make_eval_genomes(seeds, max_frames):
    def eval_genomes(genomes, config):
        for genome_id, genome in genomes:
            genome.fitness = evaluate_genome_neat(genome, config, seeds, max_frames)
    return eval_genomes


def run_neat(experiment_name, generations=None, seeds=None, verbose=True, trial_id=0):
    if generations is None:
        generations = SHARED_CONFIG["generations"]
    if seeds is None:
        seeds = TRAIN_SEEDS
    config = build_neat_config(experiment_name)
    population = neat.Population(config)
    if verbose:
        population.add_reporter(neat.StdOutReporter(True))
        population.add_reporter(neat.StatisticsReporter())
    os.makedirs("results", exist_ok=True)
    population.add_reporter(CSVReporter(f"results/{experiment_name}_trial{trial_id}.csv"))
    eval_fn = make_eval_genomes(seeds, max_frames=MAX_FRAMES_EVAL)
    winner = population.run(eval_fn, generations)
    return winner, config


class StaticNetwork:
    def __init__(self, weights, hidden_layers, is_recurrent=False):
        self.hidden_layers = hidden_layers
        self.is_recurrent = is_recurrent
        self.weights = weights
        self.layer_sizes = [3] + hidden_layers + [1]
        self._parse_weights()
        self.hidden_state = [0.0] * hidden_layers[0] if is_recurrent and hidden_layers else None

    def _parse_weights(self):
        self.layer_weights = []
        self.layer_biases = []
        idx = 0
        for i in range(len(self.layer_sizes) - 1):
            in_size = self.layer_sizes[i]
            out_size = self.layer_sizes[i + 1]
            layer_w = []
            for _ in range(out_size):
                layer_w.append(self.weights[idx:idx + in_size])
                idx += in_size
            self.layer_weights.append(layer_w)
            self.layer_biases.append(self.weights[idx:idx + out_size])
            idx += out_size
        self.recurrent_weights = None
        if self.is_recurrent and self.hidden_layers:
            hidden_size = self.hidden_layers[0]
            self.recurrent_weights = []
            for _ in range(hidden_size):
                self.recurrent_weights.append(self.weights[idx:idx + hidden_size])
                idx += hidden_size

    def activate(self, inputs):
        x = list(inputs)
        for layer_idx, (weights, biases) in enumerate(zip(self.layer_weights, self.layer_biases)):
            new_x = []
            for neuron_idx, (w, b) in enumerate(zip(weights, biases)):
                total = sum(xi * wi for xi, wi in zip(x, w)) + b
                if layer_idx == 0 and self.is_recurrent and self.hidden_state and self.recurrent_weights:
                    for h_idx, h_val in enumerate(self.hidden_state):
                        total += h_val * self.recurrent_weights[neuron_idx][h_idx]
                new_x.append(math.tanh(total))
            x = new_x
            if layer_idx == 0 and self.is_recurrent and self.hidden_layers:
                self.hidden_state = x.copy()
        return x

    def reset(self):
        if self.is_recurrent and self.hidden_layers:
            self.hidden_state = [0.0] * self.hidden_layers[0]


def calculate_genome_size(hidden_layers, is_recurrent=False):
    layer_sizes = [3] + hidden_layers + [1]
    total = 0
    for i in range(len(layer_sizes) - 1):
        total += layer_sizes[i] * layer_sizes[i + 1] + layer_sizes[i + 1]
    if is_recurrent and hidden_layers:
        total += hidden_layers[0] ** 2
    return total


def random_genome(size):
    mean, stdev = GA_CONFIG["weight_init_mean"], GA_CONFIG["weight_init_stdev"]
    return [random.gauss(mean, stdev) for _ in range(size)]


def uniform_crossover(p1, p2):
    return [random.choice([a, b]) for a, b in zip(p1, p2)]


def single_point_crossover(p1, p2):
    point = random.randint(1, len(p1) - 1)
    return p1[:point] + p2[point:]


def mutate(genome, weight_mutate_rate=None):
    rate = weight_mutate_rate if weight_mutate_rate is not None else SHARED_CONFIG["weight_mutate_rate"]
    power = SHARED_CONFIG["weight_mutate_power"]
    replace = SHARED_CONFIG["weight_replace_rate"]
    w_min, w_max = GA_CONFIG["weight_min"], GA_CONFIG["weight_max"]
    new_genome = []
    for gene in genome:
        if random.random() < rate:
            if random.random() < replace:
                gene = random.gauss(GA_CONFIG["weight_init_mean"], GA_CONFIG["weight_init_stdev"])
            else:
                gene += random.gauss(0, power)
            gene = max(w_min, min(w_max, gene))
        new_genome.append(gene)
    return new_genome


def evaluate_genome_ga(genome, hidden_layers, is_recurrent, seeds, max_frames=MAX_FRAMES_EVAL):
    total_score = 0
    for seed in seeds:
        net = StaticNetwork(genome, hidden_layers, is_recurrent)
        game = FlappyGame(seed=seed)
        score = play_game(net, game, max_frames=max_frames)
        total_score += score
    return total_score / len(seeds)


def run_static_ga(experiment_name, generations=None, seeds=None, verbose=True, trial_id=0):
    if generations is None:
        generations = SHARED_CONFIG["generations"]
    if seeds is None:
        seeds = TRAIN_SEEDS
    exp = EXPERIMENTS[experiment_name]
    hidden_layers = exp["hidden_layers"]
    is_recurrent = not exp["feed_forward"]
    pop_size = SHARED_CONFIG["pop_size"]
    elitism = SHARED_CONFIG["elitism"]
    survival_threshold = GA_CONFIG["survival_threshold"]
    crossover_fn = uniform_crossover if GA_CONFIG["crossover_type"] == "uniform" else single_point_crossover
    genome_size = calculate_genome_size(hidden_layers, is_recurrent)

    param_set_name = exp.get("param_set", "high")
    param_set = PARAM_SETS[param_set_name]
    weight_mutate_rate = param_set["weight_mutate_rate"]

    if verbose:
        print(f"Static GA: {experiment_name} | layers={hidden_layers} | mutate_rate={weight_mutate_rate} | pop={pop_size}")

    os.makedirs("results", exist_ok=True)
    csv_path = f"results/{experiment_name}_trial{trial_id}.csv"
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["generation", "best_fitness", "avg_fitness", "min_fitness"])

    population = [random_genome(genome_size) for _ in range(pop_size)]
    best_genome, best_fitness = None, -float('inf')

    for gen in range(generations):
        fitness_scores = [evaluate_genome_ga(g, hidden_layers, is_recurrent, seeds) for g in population]
        ranked = sorted(zip(population, fitness_scores), key=lambda x: -x[1])
        if ranked[0][1] > best_fitness:
            best_fitness = ranked[0][1]
            best_genome = ranked[0][0].copy()
        avg = sum(fitness_scores) / len(fitness_scores)
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([gen, ranked[0][1], avg, ranked[-1][1]])
        if verbose:
            print(f"Gen {gen:3d} | Best: {ranked[0][1]:7.2f} | Avg: {avg:7.2f}")
        new_population = [ranked[i][0].copy() for i in range(elitism)]
        parent_pool = [g for g, _ in ranked[:max(2, int(pop_size * survival_threshold))]]
        while len(new_population) < pop_size:
            p1, p2 = random.sample(parent_pool, 2)
            child = mutate(crossover_fn(p1, p2), weight_mutate_rate)
            new_population.append(child)
        population = new_population

    if verbose:
        print(f"Training complete! Best fitness: {best_fitness:.2f}")
    return best_genome, hidden_layers, is_recurrent, best_fitness


def run_experiment(experiment_name, generations=None, verbose=True, trial_id=0):
    if generations is None:
        generations = SHARED_CONFIG["generations"]
    exp = EXPERIMENTS[experiment_name]
    if exp.get("use_neat", True):
        winner, config = run_neat(experiment_name, generations, TRAIN_SEEDS, verbose, trial_id)
        output_path = f"winner_{experiment_name}_trial{trial_id}.pkl"
        with open(output_path, "wb") as f:
            pickle.dump((winner, config), f)
        if verbose:
            print(f"Saved to {output_path} | fitness: {winner.fitness:.2f}")
        return winner
    else:
        best_genome, hidden_layers, is_recurrent, best_fitness = run_static_ga(
            experiment_name, generations, TRAIN_SEEDS, verbose, trial_id
        )
        output_path = f"winner_{experiment_name}_trial{trial_id}.pkl"
        with open(output_path, "wb") as f:
            pickle.dump({"genome": best_genome, "hidden_layers": hidden_layers,
                        "is_recurrent": is_recurrent, "fitness": best_fitness}, f)
        if verbose:
            print(f"Saved to {output_path}")
        return best_genome


def evaluate_winner(winner_path, seeds=None, verbose=True, log_csv=True):
    if seeds is None:
        seeds = TEST_SEEDS
    with open(winner_path, "rb") as f:
        data = pickle.load(f)

    per_seed_scores = []
    if isinstance(data, tuple):
        winner, config = data
        is_recurrent = not config.genome_config.feed_forward
        for seed in seeds:
            net = create_network(winner, config, is_recurrent)
            game = FlappyGame(seed=seed)
            score = play_game(net, game, max_frames=MAX_FRAMES_EVAL)
            per_seed_scores.append(score)
    elif isinstance(data, dict):
        for seed in seeds:
            net = StaticNetwork(data["genome"], data["hidden_layers"], data["is_recurrent"])
            game = FlappyGame(seed=seed)
            score = play_game(net, game, max_frames=MAX_FRAMES_EVAL)
            per_seed_scores.append(score)
    else:
        raise ValueError(f"Unknown winner format in {winner_path}")

    fitness = sum(per_seed_scores) / len(per_seed_scores)
    std_dev = (sum((s - fitness) ** 2 for s in per_seed_scores) / len(per_seed_scores)) ** 0.5

    if verbose:
        print(f"Test fitness ({len(seeds)} seeds): {fitness:.2f} ± {std_dev:.2f}")
        print(f"  Min: {min(per_seed_scores)}, Max: {max(per_seed_scores)}")

    if log_csv:
        import re
        match = re.search(r"winner_(.+)_trial(\d+)\.pkl", winner_path)
        if match:
            exp_name, trial_id = match.group(1), int(match.group(2))
            csv_path = "results/test_results.csv"
            os.makedirs("results", exist_ok=True)
            write_header = not os.path.exists(csv_path)
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(["experiment", "trial_id", "mean_fitness", "std_dev", "min", "max", "per_seed_scores"])
                writer.writerow([exp_name, trial_id, f"{fitness:.2f}", f"{std_dev:.2f}",
                                min(per_seed_scores), max(per_seed_scores), per_seed_scores])

    return fitness


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", nargs="?", choices=list(EXPERIMENTS.keys()))
    parser.add_argument("-g", "--generations", type=int, default=None)
    parser.add_argument("-n", "--trials", type=int, default=1, help="Number of trials to run")
    parser.add_argument("-t", "--test", action="store_true")
    parser.add_argument("--run-all", action="store_true", help="Run all 12 experiments sequentially")
    args = parser.parse_args()

    if args.run_all:
        for exp_name in EXPERIMENTS.keys():
            for trial in range(args.trials):
                print(f"\n{'='*60}")
                print(f"Running experiment: {exp_name} (trial {trial + 1}/{args.trials})")
                print(f"{'='*60}\n")
                run_experiment(exp_name, args.generations, verbose=True, trial_id=trial)
                if args.test:
                    evaluate_winner(f"winner_{exp_name}_trial{trial}.pkl", TEST_SEEDS, verbose=True)
    elif args.experiment:
        for trial in range(args.trials):
            if args.trials > 1:
                print(f"\n--- Trial {trial + 1}/{args.trials} ---\n")
            run_experiment(args.experiment, args.generations, verbose=True, trial_id=trial)
            if args.test:
                evaluate_winner(f"winner_{args.experiment}_trial{trial}.pkl", TEST_SEEDS, verbose=True)
    else:
        parser.error("Either specify an experiment name or use --run-all")
