TRAIN_SEEDS = [1, 7, 13, 22, 31, 45, 58, 67, 79, 88]
TEST_SEEDS = [2, 14, 29, 38, 47, 56, 63, 74, 85, 91]
GAME_MODE = "standard"  # "standard" or "oscillating"

SHARED_CONFIG = {
    "pop_size": 50,
    "generations": 50,
    "elitism": 2,
    "weight_mutate_rate": 0.8,
    "weight_mutate_power": 0.5,
    "weight_replace_rate": 0.1,
}

GA_CONFIG = {
    "crossover_type": "uniform",
    "survival_threshold": 0.2,
    "weight_init_mean": 0.0,
    "weight_init_stdev": 0.5,
    "weight_min": -30.0,
    "weight_max": 30.0,
}

BASE_CONFIG = {
    "NEAT": {
        "fitness_criterion": "max",
        "fitness_threshold": "1000",
        "pop_size": str(SHARED_CONFIG["pop_size"]),
        "reset_on_extinction": "False",
        "no_fitness_termination": "True",
    },
    "DefaultGenome": {
        "num_inputs": "3",
        "num_outputs": "1",
        "num_hidden": "0",
        "feed_forward": "True",
        "initial_connection": "full",
        "activation_default": "tanh",
        "activation_mutate_rate": "0.0",
        "activation_options": "tanh",
        "aggregation_default": "sum",
        "aggregation_mutate_rate": "0.0",
        "aggregation_options": "sum",
        "node_add_prob": "0.4",
        "node_delete_prob": "0.1",
        "conn_add_prob": "0.7",
        "conn_delete_prob": "0.3",
        "enabled_default": "True",
        "enabled_mutate_rate": "0.01",
        "enabled_rate_to_true_add": "0.0",
        "enabled_rate_to_false_add": "0.0",
        "single_structural_mutation": "False",
        "structural_mutation_surer": "default",
        "compatibility_disjoint_coefficient": "1.0",
        "compatibility_weight_coefficient": "0.5",
        "weight_init_type": "gaussian",
        "weight_init_mean": "0.0",
        "weight_init_stdev": "0.5",
        "weight_max_value": "30.0",
        "weight_min_value": "-30.0",
        "weight_mutate_power": str(SHARED_CONFIG["weight_mutate_power"]),
        "weight_mutate_rate": str(SHARED_CONFIG["weight_mutate_rate"]),
        "weight_replace_rate": str(SHARED_CONFIG["weight_replace_rate"]),
        "bias_init_type": "gaussian",
        "bias_init_mean": "0.0",
        "bias_init_stdev": "0.5",
        "bias_max_value": "30.0",
        "bias_min_value": "-30.0",
        "bias_mutate_power": "0.5",
        "bias_mutate_rate": "0.7",
        "bias_replace_rate": "0.1",
        "response_init_type": "gaussian",
        "response_init_mean": "1.0",
        "response_init_stdev": "0.0",
        "response_max_value": "30.0",
        "response_min_value": "-30.0",
        "response_mutate_power": "0.0",
        "response_mutate_rate": "0.0",
        "response_replace_rate": "0.0",
    },
    "DefaultSpeciesSet": {
        "compatibility_threshold": "2.0",
    },
    "DefaultStagnation": {
        "species_fitness_func": "max",
        "max_stagnation": "20",
        "species_elitism": str(SHARED_CONFIG["elitism"]),
    },
    "DefaultReproduction": {
        "elitism": str(SHARED_CONFIG["elitism"]),
        "survival_threshold": "0.2",
        "min_species_size": "2",
    },
}

EXPERIMENTS = {
    "neat_ff": {"feed_forward": True, "use_neat": True},
    "neat_rnn": {"feed_forward": False, "use_neat": True},
    "static_ff": {"feed_forward": True, "hidden_layers": [10, 10], "use_neat": False},
    "static_rnn": {"feed_forward": False, "hidden_layers": [10], "use_neat": False},
}
