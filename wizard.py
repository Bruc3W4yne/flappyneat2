"""
Wizard - Visual training dashboard with 3 panels:
1. Left: Swarm visualization (all birds playing with real sprites)
2. Top Right: Fitness graph
3. Bottom Right: Network topology
"""

import os
import copy
import random
import tempfile
import threading
import time
from queue import Queue, Empty

import pygame
import neat

from game import (SwarmGame, Bird, Pipe, SCREEN_WIDTH, SCREEN_HEIGHT,
                  FLOOR_Y, BIRD_X, BIRD_RX, BIRD_RY, PIPE_WIDTH, PIPE_GAP)
from config import BASE_CONFIG, EXPERIMENTS, SHARED_CONFIG, TRAIN_SEEDS
from trainer import StaticNetwork, calculate_genome_size, random_genome, mutate, uniform_crossover

# Window dimensions
WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 900
GAME_PANEL_WIDTH = int(WINDOW_WIDTH * 0.6)
RIGHT_PANEL_WIDTH = WINDOW_WIDTH - GAME_PANEL_WIDTH
RIGHT_PANEL_HEIGHT = WINDOW_HEIGHT // 2

# Assets path
ASSETS_PATH = os.path.join(os.path.dirname(__file__), "..", "imgs")

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
DARK_GRAY = (40, 40, 50)
LIGHT_GRAY = (100, 100, 110)
GREEN = (50, 205, 50)
DARK_GREEN = (0, 100, 0)
RED = (220, 50, 50)
BLUE = (70, 130, 180)
YELLOW = (255, 215, 0)
ORANGE = (255, 140, 0)
SKY_BLUE = (135, 206, 235)
PIPE_GREEN = (80, 180, 80)


class Wizard:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("NEAT Flappy Bird Training Wizard")
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 36)
        self.font_score = pygame.font.Font(None, 48)

        # Load sprites
        self._load_sprites()

        # Panel surfaces
        self.game_surface = pygame.Surface((GAME_PANEL_WIDTH, WINDOW_HEIGHT))
        self.graph_surface = pygame.Surface((RIGHT_PANEL_WIDTH, RIGHT_PANEL_HEIGHT))
        self.network_surface = pygame.Surface((RIGHT_PANEL_WIDTH, RIGHT_PANEL_HEIGHT))

        # State
        self.experiment = None
        self.generation = 0
        self.fitness_history = []  # [(gen, best, avg, min), ...]
        self.current_genomes = []
        self.current_fitnesses = []
        self.current_best_genome = None
        self.neat_config = None
        self.is_neat = True
        self.is_recurrent = False
        self.hidden_layers = []

        # Training thread communication
        self.training_queue = Queue()
        self.training_thread = None
        self.training_active = False
        self.training_done = False
        self.paused = False
        self.speed = 1  # 1=normal, 0=max speed

        # Visualization state
        self.swarm_game = None
        self.networks = []
        self.vis_frame = 0
        self.vis_generation = 0  # Which generation we're currently visualizing
        self.death_time = None  # Time when all birds died (for restart delay)
        self.restart_delay = 1.5  # Seconds to wait before restarting

    def _load_sprites(self):
        """Load game sprites."""
        try:
            self.bg_img = pygame.transform.scale(
                pygame.image.load(os.path.join(ASSETS_PATH, "bg.png")).convert(),
                (SCREEN_WIDTH, SCREEN_HEIGHT))
            self.bird_img = pygame.transform.scale2x(
                pygame.image.load(os.path.join(ASSETS_PATH, "bird1.png")).convert_alpha())
            self.pipe_img = pygame.transform.scale2x(
                pygame.image.load(os.path.join(ASSETS_PATH, "pipe.png")).convert_alpha())
            self.base_img = pygame.transform.scale2x(
                pygame.image.load(os.path.join(ASSETS_PATH, "base.png")).convert_alpha())
            self.sprites_loaded = True
        except Exception as e:
            print(f"Warning: Could not load sprites from {ASSETS_PATH}: {e}")
            print("Falling back to simple shapes.")
            self.sprites_loaded = False

    def run(self):
        """Main loop."""
        running = True
        self.show_experiment_selection()

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif event.key == pygame.K_s:
                        self.speed = 0 if self.speed == 1 else 1
                    elif event.key == pygame.K_ESCAPE:
                        running = False

            # Check for training updates
            self._process_training_queue()

            # Auto-restart visualization when all birds die (with delay)
            if self.swarm_game and self.swarm_game.all_dead and not self.paused:
                if self.death_time is None:
                    self.death_time = time.time()
                elif time.time() - self.death_time >= self.restart_delay:
                    self._reset_swarm_visualization()
                    self.death_time = None

            # Draw all panels
            self.draw_game_panel()
            self.draw_graph_panel()
            self.draw_network_panel()

            # Blit panels to screen
            self.screen.blit(self.game_surface, (0, 0))
            self.screen.blit(self.graph_surface, (GAME_PANEL_WIDTH, 0))
            self.screen.blit(self.network_surface, (GAME_PANEL_WIDTH, RIGHT_PANEL_HEIGHT))

            # Draw panel borders
            pygame.draw.line(self.screen, WHITE, (GAME_PANEL_WIDTH, 0), (GAME_PANEL_WIDTH, WINDOW_HEIGHT), 2)
            pygame.draw.line(self.screen, WHITE, (GAME_PANEL_WIDTH, RIGHT_PANEL_HEIGHT),
                           (WINDOW_WIDTH, RIGHT_PANEL_HEIGHT), 2)

            pygame.display.flip()
            self.clock.tick(60 if self.speed == 1 else 0)

        self.training_active = False
        pygame.quit()

    def show_experiment_selection(self):
        """Show experiment selection screen."""
        experiments = list(EXPERIMENTS.keys())
        selected = 0

        while True:
            self.screen.fill(DARK_GRAY)
            title = self.font_large.render("Select Experiment", True, WHITE)
            self.screen.blit(title, (WINDOW_WIDTH // 2 - title.get_width() // 2, 100))

            for i, exp in enumerate(experiments):
                color = YELLOW if i == selected else WHITE
                exp_info = EXPERIMENTS[exp]
                ff_rnn = "FF" if exp_info["feed_forward"] else "RNN"
                neat_ga = "NEAT" if exp_info.get("use_neat", True) else "Static GA"
                text = self.font.render(f"{exp} ({neat_ga}, {ff_rnn})", True, color)
                self.screen.blit(text, (WINDOW_WIDTH // 2 - text.get_width() // 2, 200 + i * 50))

            instructions = self.font.render("UP/DOWN to select, ENTER to start", True, LIGHT_GRAY)
            self.screen.blit(instructions, (WINDOW_WIDTH // 2 - instructions.get_width() // 2, 450))

            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        selected = (selected - 1) % len(experiments)
                    elif event.key == pygame.K_DOWN:
                        selected = (selected + 1) % len(experiments)
                    elif event.key == pygame.K_RETURN:
                        self.start_experiment(experiments[selected])
                        return

    def start_experiment(self, experiment_name):
        """Start training for selected experiment."""
        self.experiment = experiment_name
        exp = EXPERIMENTS[experiment_name]
        self.is_neat = exp.get("use_neat", True)
        self.is_recurrent = not exp["feed_forward"]
        self.hidden_layers = exp.get("hidden_layers", [])

        # Build config
        if self.is_neat:
            self.neat_config = self._build_neat_config(experiment_name)

        # Start training thread
        self.training_active = True
        self.training_done = False
        self.training_thread = threading.Thread(target=self._training_loop, daemon=True)
        self.training_thread.start()

    def _build_neat_config(self, experiment_name):
        """Build NEAT config from experiment."""
        exp = EXPERIMENTS[experiment_name]
        config_dict = copy.deepcopy(BASE_CONFIG)
        config_dict["DefaultGenome"]["feed_forward"] = str(exp["feed_forward"])
        if not exp.get("use_neat", True):
            config_dict["DefaultGenome"]["node_add_prob"] = "0.0"
            config_dict["DefaultGenome"]["node_delete_prob"] = "0.0"
            config_dict["DefaultGenome"]["conn_add_prob"] = "0.0"
            config_dict["DefaultGenome"]["conn_delete_prob"] = "0.0"
        config_text = self._dict_to_config_file(config_dict)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".cfg", delete=False) as f:
            f.write(config_text)
            config_path = f.name
        neat_config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                  neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
        os.unlink(config_path)
        return neat_config

    def _dict_to_config_file(self, config_dict):
        """Convert config dict to NEAT config file format."""
        lines = []
        for section, params in config_dict.items():
            lines.append(f"[{section}]")
            for key, value in params.items():
                lines.append(f"{key} = {value}")
            lines.append("")
        return "\n".join(lines)

    def _training_loop(self):
        """Training loop running in separate thread."""
        if self.is_neat:
            self._run_neat_training()
        else:
            self._run_ga_training()

    def _run_neat_training(self):
        """Run NEAT training."""
        population = neat.Population(self.neat_config)
        generations = SHARED_CONFIG["generations"]
        gen_count = [0]

        def eval_genomes(genomes, config):
            if not self.training_active:
                return

            # Create networks for visualization
            networks = []
            for genome_id, genome in genomes:
                if self.is_recurrent:
                    net = neat.nn.RecurrentNetwork.create(genome, config)
                else:
                    net = neat.nn.FeedForwardNetwork.create(genome, config)
                networks.append((genome, net))

            # Evaluate on training seeds
            for genome, net in networks:
                total_score = 0
                for seed in TRAIN_SEEDS:
                    from game import FlappyGame, play_game
                    game = FlappyGame(seed=seed)
                    score = play_game(net, game, max_frames=5000)
                    total_score += score
                genome.fitness = total_score / len(TRAIN_SEEDS)

            # Send to visualization
            fitnesses = [g.fitness for _, g in genomes]
            best_idx = fitnesses.index(max(fitnesses))
            self.training_queue.put({
                "type": "generation",
                "generation": gen_count[0],
                "genomes": [g for _, g in genomes],
                "fitnesses": fitnesses,
                "best_genome": genomes[best_idx][1],
                "networks": [n for _, n in networks]
            })
            gen_count[0] += 1

        for _ in range(generations):
            if not self.training_active:
                break
            population.run(eval_genomes, 1)

        self.training_queue.put({"type": "done"})

    def _run_ga_training(self):
        """Run static GA training."""
        pop_size = SHARED_CONFIG["pop_size"]
        generations = SHARED_CONFIG["generations"]
        elitism = SHARED_CONFIG["elitism"]
        genome_size = calculate_genome_size(self.hidden_layers, self.is_recurrent)

        population = [random_genome(genome_size) for _ in range(pop_size)]

        for gen in range(generations):
            if not self.training_active:
                break

            # Evaluate population
            fitnesses = []
            networks = []
            for genome in population:
                net = StaticNetwork(genome, self.hidden_layers, self.is_recurrent)
                networks.append(net)
                total_score = 0
                for seed in TRAIN_SEEDS:
                    from game import FlappyGame
                    game = FlappyGame(seed=seed)
                    game.reset()
                    for _ in range(5000):
                        obs = game.get_observation()
                        output = net.activate(obs)
                        action = output[0] > 0
                        _, _, done, _ = game.step(action)
                        if done:
                            break
                    total_score += game.score
                    if self.is_recurrent:
                        net.reset()
                fitnesses.append(total_score / len(TRAIN_SEEDS))

            # Send to visualization
            best_idx = fitnesses.index(max(fitnesses))
            self.training_queue.put({
                "type": "generation",
                "generation": gen,
                "genomes": population.copy(),
                "fitnesses": fitnesses,
                "best_genome": population[best_idx],
                "networks": [StaticNetwork(g, self.hidden_layers, self.is_recurrent) for g in population]
            })

            # Selection and reproduction
            ranked = sorted(zip(population, fitnesses), key=lambda x: -x[1])
            new_population = [list(ranked[i][0]) for i in range(elitism)]
            parent_pool = [g for g, _ in ranked[:max(2, int(pop_size * 0.2))]]
            while len(new_population) < pop_size:
                p1, p2 = random.sample(parent_pool, 2)
                child = mutate(uniform_crossover(p1, p2))
                new_population.append(child)
            population = new_population

        self.training_queue.put({"type": "done"})

    def _process_training_queue(self):
        """Process messages from training thread."""
        try:
            while True:
                msg = self.training_queue.get_nowait()
                if msg["type"] == "generation":
                    self.generation = msg["generation"]
                    self.current_genomes = msg["genomes"]
                    self.current_fitnesses = msg["fitnesses"]
                    self.current_best_genome = msg["best_genome"]
                    self.networks = msg["networks"]

                    # Update fitness history
                    best = max(self.current_fitnesses)
                    avg = sum(self.current_fitnesses) / len(self.current_fitnesses)
                    min_fit = min(self.current_fitnesses)
                    self.fitness_history.append((self.generation, best, avg, min_fit))

                    # Reset swarm for new generation visualization
                    self._reset_swarm_visualization()

                elif msg["type"] == "done":
                    self.training_active = False
                    self.training_done = True
        except Empty:
            pass

    def _reset_swarm_visualization(self):
        """Reset swarm game for visualizing current generation."""
        if self.current_genomes and self.networks:
            self.swarm_game = SwarmGame(len(self.current_genomes), seed=TRAIN_SEEDS[0])
            self.vis_frame = 0
            self.vis_generation = self.generation
            self.death_time = None  # Reset death timer

    def draw_game_panel(self):
        """Draw the swarm visualization panel with sprites."""
        # Calculate scaling to fit game in panel
        scale_x = GAME_PANEL_WIDTH / SCREEN_WIDTH
        scale_y = WINDOW_HEIGHT / SCREEN_HEIGHT
        scale = min(scale_x, scale_y)
        offset_x = (GAME_PANEL_WIDTH - SCREEN_WIDTH * scale) / 2
        offset_y = (WINDOW_HEIGHT - SCREEN_HEIGHT * scale) / 2

        # Draw background
        if self.sprites_loaded:
            scaled_bg = pygame.transform.scale(self.bg_img,
                (int(SCREEN_WIDTH * scale), int(SCREEN_HEIGHT * scale)))
            self.game_surface.fill(DARK_GRAY)
            self.game_surface.blit(scaled_bg, (offset_x, offset_y))
        else:
            self.game_surface.fill(SKY_BLUE)

        if not self.swarm_game or not self.networks:
            # Show waiting message
            text = self.font_large.render("Waiting for training...", True, WHITE)
            self.game_surface.blit(text, (GAME_PANEL_WIDTH // 2 - text.get_width() // 2,
                                          WINDOW_HEIGHT // 2))
            return

        # Step simulation if not paused and birds alive
        if not self.paused and not self.swarm_game.all_dead:
            observations = self.swarm_game.get_observations()
            actions = []
            for i, (obs, net) in enumerate(zip(observations, self.networks)):
                if self.swarm_game.alive[i]:
                    output = net.activate(obs)
                    action = output[0] > 0 if isinstance(output, (list, tuple)) else output > 0
                    actions.append(action)
                else:
                    actions.append(False)
            self.swarm_game.step(actions)
            self.vis_frame += 1

        # Draw pipes (only those within visible game area)
        if self.sprites_loaded:
            scaled_pipe = pygame.transform.scale(self.pipe_img,
                (int(self.pipe_img.get_width() * scale), int(self.pipe_img.get_height() * scale)))
            for pipe in self.swarm_game.pipes:
                # Skip pipes outside visible game area
                if pipe.x > SCREEN_WIDTH or pipe.x + PIPE_WIDTH < 0:
                    continue
                pipe_x = int(pipe.x * scale + offset_x)
                # Top pipe (flipped)
                pipe_top = pygame.transform.flip(scaled_pipe, False, True)
                top_y = int(pipe.gap_top * scale + offset_y) - pipe_top.get_height()
                self.game_surface.blit(pipe_top, (pipe_x, top_y))
                # Bottom pipe
                bot_y = int(pipe.gap_bottom * scale + offset_y)
                self.game_surface.blit(scaled_pipe, (pipe_x, bot_y))
        else:
            for pipe in self.swarm_game.pipes:
                # Skip pipes outside visible game area
                if pipe.x > SCREEN_WIDTH or pipe.x + PIPE_WIDTH < 0:
                    continue
                pipe_x = int(pipe.x * scale + offset_x)
                pipe_w = int(PIPE_WIDTH * scale)
                top_h = int(pipe.gap_top * scale)
                pygame.draw.rect(self.game_surface, PIPE_GREEN,
                               (pipe_x, int(offset_y), pipe_w, top_h))
                bot_y = int(pipe.gap_bottom * scale + offset_y)
                bot_h = int((FLOOR_Y - pipe.gap_bottom) * scale)
                pygame.draw.rect(self.game_surface, PIPE_GREEN,
                               (pipe_x, bot_y, pipe_w, bot_h))

        # Draw birds with transparency based on fitness
        if self.current_fitnesses:
            min_fit = min(self.current_fitnesses)
            max_fit = max(self.current_fitnesses)

            # Sort by fitness so better ones draw on top
            bird_data = list(zip(self.swarm_game.birds, self.swarm_game.alive,
                                self.current_fitnesses))
            bird_data.sort(key=lambda x: x[2])

            for bird, alive, fitness in bird_data:
                if not alive:
                    continue

                # Calculate alpha based on fitness
                if max_fit > min_fit:
                    normalized = (fitness - min_fit) / (max_fit - min_fit)
                else:
                    normalized = 1.0
                alpha = int(50 + normalized * 205)

                # Bird position
                bx = int(bird.x * scale + offset_x)
                by = int(bird.y * scale + offset_y)

                if self.sprites_loaded:
                    # Scale and rotate bird sprite
                    scaled_bird = pygame.transform.scale(self.bird_img,
                        (int(self.bird_img.get_width() * scale),
                         int(self.bird_img.get_height() * scale)))
                    rotated = pygame.transform.rotate(scaled_bird, -bird.vel * 3)
                    # Apply alpha
                    rotated.set_alpha(alpha)
                    self.game_surface.blit(rotated, (bx, by))
                else:
                    # Fallback: draw ellipse
                    rx = int(BIRD_RX * scale)
                    ry = int(BIRD_RY * scale)
                    cx = bx + int(40 * scale)
                    cy = by + int(30 * scale)
                    bird_surf = pygame.Surface((rx * 2, ry * 2), pygame.SRCALPHA)
                    color = (*YELLOW[:3], alpha)
                    pygame.draw.ellipse(bird_surf, color, (0, 0, rx * 2, ry * 2))
                    self.game_surface.blit(bird_surf, (cx - rx, cy - ry))

        # Draw base/floor
        if self.sprites_loaded:
            scaled_base = pygame.transform.scale(self.base_img,
                (int(SCREEN_WIDTH * scale), int(self.base_img.get_height() * scale)))
            base_y = int(FLOOR_Y * scale + offset_y)
            self.game_surface.blit(scaled_base, (offset_x, base_y))
        else:
            floor_y = int(FLOOR_Y * scale + offset_y)
            pygame.draw.rect(self.game_surface, DARK_GREEN,
                            (0, floor_y, GAME_PANEL_WIDTH, WINDOW_HEIGHT - floor_y))

        # Draw score
        best_score = max(self.swarm_game.scores) if self.swarm_game.scores else 0
        score_text = self.font_score.render(str(best_score), True, WHITE)
        self.game_surface.blit(score_text,
            (GAME_PANEL_WIDTH // 2 - score_text.get_width() // 2, int(50 * scale + offset_y)))

        # Draw info overlay
        alive_count = sum(self.swarm_game.alive)
        total_count = len(self.swarm_game.alive)
        status = "PAUSED" if self.paused else ("DONE" if self.training_done else "Training")

        info_lines = [
            f"Gen: {self.generation} | Visualizing: {self.vis_generation}",
            f"Alive: {alive_count}/{total_count}",
            f"Status: {status} | Speed: {'Normal' if self.speed == 1 else 'Max'}",
            f"[SPACE] Pause | [S] Speed | [ESC] Quit"
        ]
        for i, line in enumerate(info_lines):
            text = self.font.render(line, True, WHITE)
            # Draw shadow for readability
            shadow = self.font.render(line, True, BLACK)
            self.game_surface.blit(shadow, (12, 12 + i * 22))
            self.game_surface.blit(text, (10, 10 + i * 22))

    def draw_graph_panel(self):
        """Draw the fitness graph panel."""
        self.graph_surface.fill(DARK_GRAY)

        # Title
        title = self.font.render("Fitness Over Generations", True, WHITE)
        self.graph_surface.blit(title, (RIGHT_PANEL_WIDTH // 2 - title.get_width() // 2, 10))

        if not self.fitness_history:
            text = self.font.render("No data yet...", True, LIGHT_GRAY)
            self.graph_surface.blit(text, (RIGHT_PANEL_WIDTH // 2 - text.get_width() // 2,
                                          RIGHT_PANEL_HEIGHT // 2))
            return

        # Graph area
        margin = 50
        graph_x = margin
        graph_y = 40
        graph_w = RIGHT_PANEL_WIDTH - margin * 2
        graph_h = RIGHT_PANEL_HEIGHT - margin - graph_y

        # Draw axes
        pygame.draw.line(self.graph_surface, WHITE,
                        (graph_x, graph_y + graph_h), (graph_x + graph_w, graph_y + graph_h), 1)
        pygame.draw.line(self.graph_surface, WHITE,
                        (graph_x, graph_y), (graph_x, graph_y + graph_h), 1)

        # Get data ranges
        gens = [h[0] for h in self.fitness_history]
        bests = [h[1] for h in self.fitness_history]
        avgs = [h[2] for h in self.fitness_history]

        max_gen = max(max(gens), 1) if gens else 1  # Ensure at least 1 to avoid div by zero
        max_fit = max(max(bests), max(avgs)) if bests else 1
        max_fit = max(max_fit, 1)

        # Draw lines
        def plot_line(data, color):
            if len(data) < 2:
                return
            points = []
            for gen, val in zip(gens, data):
                x = graph_x + (gen / max_gen) * graph_w
                y = graph_y + graph_h - (val / max_fit) * graph_h
                points.append((x, y))
            if len(points) >= 2:
                pygame.draw.lines(self.graph_surface, color, False, points, 2)

        plot_line(bests, GREEN)
        plot_line(avgs, BLUE)

        # Draw current values
        if bests:
            current_best = bests[-1]
            current_avg = avgs[-1]
            stats_text = f"Best: {current_best:.1f} | Avg: {current_avg:.1f}"
            stats_render = self.font.render(stats_text, True, WHITE)
            self.graph_surface.blit(stats_render, (graph_x, graph_y + graph_h + 25))

        # Legend
        pygame.draw.line(self.graph_surface, GREEN, (graph_x, graph_y - 15),
                        (graph_x + 20, graph_y - 15), 2)
        best_label = self.font.render("Best", True, GREEN)
        self.graph_surface.blit(best_label, (graph_x + 25, graph_y - 22))

        pygame.draw.line(self.graph_surface, BLUE, (graph_x + 80, graph_y - 15),
                        (graph_x + 100, graph_y - 15), 2)
        avg_label = self.font.render("Avg", True, BLUE)
        self.graph_surface.blit(avg_label, (graph_x + 105, graph_y - 22))

        # Axis labels
        y_label = self.font.render(f"{max_fit:.0f}", True, WHITE)
        self.graph_surface.blit(y_label, (5, graph_y))
        x_label = self.font.render(f"Gen {max_gen}", True, WHITE)
        self.graph_surface.blit(x_label, (graph_x + graph_w - 50, graph_y + graph_h + 5))

    def draw_network_panel(self):
        """Draw the network topology panel."""
        self.network_surface.fill(DARK_GRAY)

        # Title
        title = self.font.render("Network Topology (Best Genome)", True, WHITE)
        self.network_surface.blit(title, (RIGHT_PANEL_WIDTH // 2 - title.get_width() // 2, 10))

        if self.current_best_genome is None:
            text = self.font.render("No genome yet...", True, LIGHT_GRAY)
            self.network_surface.blit(text, (RIGHT_PANEL_WIDTH // 2 - text.get_width() // 2,
                                            RIGHT_PANEL_HEIGHT // 2))
            return

        if self.is_neat:
            self._draw_neat_network()
        else:
            self._draw_static_network()

    def _draw_neat_network(self):
        """Draw NEAT genome network."""
        genome = self.current_best_genome
        margin = 60
        width = RIGHT_PANEL_WIDTH - margin * 2
        height = RIGHT_PANEL_HEIGHT - margin * 2 - 20
        cx = RIGHT_PANEL_WIDTH // 2
        cy = RIGHT_PANEL_HEIGHT // 2 + 10

        # Get nodes by layer
        input_nodes = [-1, -2, -3]
        output_nodes = [0]
        hidden_nodes = [k for k in genome.nodes.keys() if k not in input_nodes + output_nodes]

        # Compute layers
        if hidden_nodes:
            num_layers = 3
        else:
            num_layers = 2

        # Node positions
        node_pos = {}
        layer_x = lambda l: margin + (l / (num_layers - 1)) * width if num_layers > 1 else cx

        # Input layer
        input_labels = ["vel", "dist", "gap"]
        for i, node in enumerate(input_nodes):
            y = cy - height // 4 + (i - 1) * (height // 3)
            node_pos[node] = (layer_x(0), y, input_labels[i])

        # Hidden layer
        for i, node in enumerate(hidden_nodes):
            y = cy - (len(hidden_nodes) - 1) * 30 // 2 + i * 30
            node_pos[node] = (layer_x(1), y, f"h{node}")

        # Output layer
        node_pos[0] = (layer_x(num_layers - 1), cy, "jump")

        # Draw connections
        for (src, dst), conn in genome.connections.items():
            if not conn.enabled:
                continue
            if src not in node_pos or dst not in node_pos:
                continue

            x1, y1, _ = node_pos[src]
            x2, y2, _ = node_pos[dst]

            # Color based on weight sign
            weight = conn.weight
            if weight > 0:
                color = GREEN
            else:
                color = RED

            # Thickness based on weight magnitude
            thickness = max(1, min(5, int(abs(weight))))

            pygame.draw.line(self.network_surface, color, (x1, y1), (x2, y2), thickness)

        # Draw nodes
        for node, (x, y, label) in node_pos.items():
            radius = 20
            if node in input_nodes:
                color = BLUE
            elif node == 0:
                color = ORANGE
            else:
                color = LIGHT_GRAY

            pygame.draw.circle(self.network_surface, color, (int(x), int(y)), radius)
            pygame.draw.circle(self.network_surface, WHITE, (int(x), int(y)), radius, 2)

            # Label
            text = self.font.render(label, True, WHITE)
            self.network_surface.blit(text, (x - text.get_width() // 2, y - text.get_height() // 2))

        # Node count info
        info = f"Nodes: {len(genome.nodes)} | Connections: {sum(1 for c in genome.connections.values() if c.enabled)}"
        info_text = self.font.render(info, True, LIGHT_GRAY)
        self.network_surface.blit(info_text, (margin, RIGHT_PANEL_HEIGHT - 30))

    def _draw_static_network(self):
        """Draw static GA network topology."""
        margin = 60
        width = RIGHT_PANEL_WIDTH - margin * 2
        height = RIGHT_PANEL_HEIGHT - margin * 2 - 20
        cx = RIGHT_PANEL_WIDTH // 2
        cy = RIGHT_PANEL_HEIGHT // 2 + 10

        # Layer sizes
        layer_sizes = [3] + self.hidden_layers + [1]
        num_layers = len(layer_sizes)

        # Node positions
        node_pos = {}
        layer_labels = [["vel", "dist", "gap"]]
        for i, size in enumerate(self.hidden_layers):
            layer_labels.append([f"h{j}" for j in range(size)])
        layer_labels.append(["jump"])

        for layer_idx, (size, labels) in enumerate(zip(layer_sizes, layer_labels)):
            x = margin + (layer_idx / (num_layers - 1)) * width if num_layers > 1 else cx
            for node_idx in range(size):
                y = cy - (size - 1) * 30 // 2 + node_idx * 30
                node_pos[(layer_idx, node_idx)] = (x, y, labels[node_idx])

        # Draw connections (simplified - just show structure)
        for l in range(num_layers - 1):
            for i in range(layer_sizes[l]):
                for j in range(layer_sizes[l + 1]):
                    x1, y1, _ = node_pos[(l, i)]
                    x2, y2, _ = node_pos[(l + 1, j)]
                    pygame.draw.line(self.network_surface, LIGHT_GRAY, (x1, y1), (x2, y2), 1)

        # Draw nodes
        for (layer_idx, node_idx), (x, y, label) in node_pos.items():
            radius = 20
            if layer_idx == 0:
                color = BLUE
            elif layer_idx == num_layers - 1:
                color = ORANGE
            else:
                color = LIGHT_GRAY

            pygame.draw.circle(self.network_surface, color, (int(x), int(y)), radius)
            pygame.draw.circle(self.network_surface, WHITE, (int(x), int(y)), radius, 2)

            text = self.font.render(label, True, WHITE)
            self.network_surface.blit(text, (x - text.get_width() // 2, y - text.get_height() // 2))

        # Network info
        total_params = sum(layer_sizes[i] * layer_sizes[i+1] + layer_sizes[i+1]
                         for i in range(len(layer_sizes) - 1))
        info = f"Layers: {layer_sizes} | Params: {total_params}"
        info_text = self.font.render(info, True, LIGHT_GRAY)
        self.network_surface.blit(info_text, (margin, RIGHT_PANEL_HEIGHT - 30))


def main():
    wizard = Wizard()
    wizard.run()


if __name__ == "__main__":
    main()
