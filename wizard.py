"""
Wizard - Visual training dashboard with 3 panels:
1. Left: Swarm visualization (all birds playing - THIS IS THE EVALUATION)
2. Top Right: Fitness graph
3. Bottom Right: Network topology

Key architectural change: The visualization IS the fitness evaluation.
Birds play visually, their scores determine fitness, then evolution happens.
"""

import os
import csv
import copy
import random
import tempfile
import time

import pygame
import neat

import game as game_module
from game import (SwarmGame, SCREEN_WIDTH, SCREEN_HEIGHT, FLOOR_Y,
                  BIRD_X, BIRD_RX, BIRD_RY, PIPE_WIDTH, PIPE_GAP, network_to_action)
from config import BASE_CONFIG, EXPERIMENTS, SHARED_CONFIG, GA_CONFIG, MAX_FRAMES
from trainer import (StaticNetwork, calculate_genome_size, random_genome,
                     mutate, uniform_crossover, create_network)

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
        self.max_generations = SHARED_CONFIG["generations"]
        self.fitness_history = []  # [(gen, best, avg, min), ...]
        self.training_done = False
        self.paused = False
        self.speed = 1  # 1=normal, 0=max speed

        # NEAT/GA specific state
        self.neat_config = None
        self.neat_population = None
        self.is_neat = True
        self.is_recurrent = False
        self.hidden_layers = []

        # Current generation state
        self.genomes = []  # List of (genome_id, genome) for NEAT or just genomes for GA
        self.networks = []
        self.current_best_genome = None

        # Swarm game state (VISUALIZATION = EVALUATION)
        self.swarm_game = None
        self.vis_frame = 0
        self.max_frames = MAX_FRAMES

        # Death pause tracking
        self.death_time = None
        self.death_delay = 1.0  # Seconds to show dead state before evolving

        # Game mode (standard or oscillating pipes)
        self.game_mode = game_module.GAME_MODE  # Start with config default

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

        while running:
            # Show experiment selection if no experiment active
            if self.experiment is None:
                self.show_experiment_selection()
                if self.experiment is None:
                    break  # User quit during selection
                continue

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif event.key == pygame.K_s:
                        self.speed = 0 if self.speed == 1 else 1
                    elif event.key == pygame.K_o:
                        # Toggle oscillating pipes
                        self.game_mode = "oscillating" if self.game_mode == "standard" else "standard"
                        game_module.GAME_MODE = self.game_mode
                    elif event.key == pygame.K_r:
                        # Return to experiment selector
                        self._reset_state()
                        continue
                    elif event.key == pygame.K_ESCAPE:
                        running = False

            # Game logic (only if not paused)
            if not self.paused and not self.training_done:
                self._step_training()

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

        pygame.quit()

    def _reset_state(self):
        """Reset all state to return to experiment selector."""
        self.experiment = None
        self.generation = 0
        self.fitness_history = []
        self.training_done = False
        self.paused = False
        self.speed = 1
        self.neat_config = None
        self.neat_population = None
        self.genomes = []
        self.networks = []
        self.current_best_genome = None
        self.swarm_game = None
        self.death_time = None

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

            instructions = self.font.render("UP/DOWN to select, ENTER to start, ESC to quit", True, LIGHT_GRAY)
            self.screen.blit(instructions, (WINDOW_WIDTH // 2 - instructions.get_width() // 2, 450))

            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.experiment = None
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        selected = (selected - 1) % len(experiments)
                    elif event.key == pygame.K_DOWN:
                        selected = (selected + 1) % len(experiments)
                    elif event.key == pygame.K_RETURN:
                        self._start_experiment(experiments[selected])
                        return
                    elif event.key == pygame.K_ESCAPE:
                        self.experiment = None
                        return

    def _start_experiment(self, experiment_name):
        """Initialize experiment and first generation."""
        self.experiment = experiment_name
        exp = EXPERIMENTS[experiment_name]
        self.is_neat = exp.get("use_neat", True)
        self.is_recurrent = not exp["feed_forward"]
        self.hidden_layers = exp.get("hidden_layers", [])
        self.generation = 0
        self.fitness_history = []
        self.training_done = False

        if self.is_neat:
            self.neat_config = self._build_neat_config(experiment_name)
            self.neat_population = neat.Population(self.neat_config)
            # Get initial genomes
            self.genomes = list(self.neat_population.population.items())
            self._create_networks_neat()
        else:
            pop_size = SHARED_CONFIG["pop_size"]
            genome_size = calculate_genome_size(self.hidden_layers, self.is_recurrent)
            self.genomes = [random_genome(genome_size) for _ in range(pop_size)]
            self._create_networks_ga()

        # Start first generation's swarm game
        self._start_generation_game()

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

    def _create_networks_neat(self):
        """Create neural networks for NEAT genomes."""
        self.networks = []
        for genome_id, genome in self.genomes:
            if self.is_recurrent:
                net = neat.nn.RecurrentNetwork.create(genome, self.neat_config)
            else:
                net = neat.nn.FeedForwardNetwork.create(genome, self.neat_config)
            self.networks.append(net)

    def _create_networks_ga(self):
        """Create neural networks for GA genomes."""
        self.networks = []
        for genome in self.genomes:
            net = StaticNetwork(genome, self.hidden_layers, self.is_recurrent)
            self.networks.append(net)

    def _start_generation_game(self):
        """Start a new swarm game for the current generation."""
        num_birds = len(self.genomes)
        seed = int(time.time() * 1000) % (2**31)  # Random seed for variety
        self.swarm_game = SwarmGame(num_birds, seed=seed)
        self.swarm_game.reset()
        self.vis_frame = 0
        self.death_time = None

        # Reset recurrent networks
        if self.is_recurrent:
            for net in self.networks:
                if hasattr(net, 'reset'):
                    net.reset()

    def _step_training(self):
        """Step the training forward - runs swarm game and handles evolution."""
        if self.swarm_game is None:
            return

        # Check if generation is complete
        if self.swarm_game.all_dead or self.vis_frame >= self.max_frames:
            if self.death_time is None:
                self.death_time = time.time()
            elif time.time() - self.death_time >= self.death_delay:
                self._finish_generation()
            return

        # Step the swarm game with neural network decisions
        observations = self.swarm_game.get_observations()
        actions = []
        for i, (obs, net) in enumerate(zip(observations, self.networks)):
            if self.swarm_game.alive[i]:
                action = network_to_action(net.activate(obs))
                actions.append(action)
            else:
                actions.append(False)

        self.swarm_game.step(actions)
        self.vis_frame += 1

    def _save_csv(self):
        """Save fitness history to CSV file."""
        os.makedirs("results", exist_ok=True)
        csv_path = f"results/{self.experiment}_wizard.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["generation", "best_fitness", "avg_fitness", "min_fitness"])
            writer.writerows(self.fitness_history)

    def _finish_generation(self):
        """Finish current generation: record fitness, evolve, start next."""
        # Composite fitness for BREEDING (survival + score*1000) - ensures gradient
        fitnesses = self.swarm_game.get_fitnesses()

        # Pipe SCORES for DISPLAY (what user cares about)
        scores = self.swarm_game.scores
        best_score = max(scores)
        avg_score = sum(scores) / len(scores)
        min_score = min(scores)
        self.fitness_history.append((self.generation, best_score, avg_score, min_score))

        # Auto-save CSV after each generation
        self._save_csv()

        # Find best genome for network visualization
        best_idx = fitnesses.index(max(fitnesses))
        if self.is_neat:
            self.current_best_genome = self.genomes[best_idx][1]
        else:
            self.current_best_genome = self.genomes[best_idx]

        # Check if training is complete
        self.generation += 1
        if self.generation >= self.max_generations:
            self.training_done = True
            return

        # Evolve to next generation
        if self.is_neat:
            self._evolve_neat(fitnesses)
        else:
            self._evolve_ga(fitnesses)

        # Start next generation's game
        self._start_generation_game()

    def _evolve_neat(self, fitnesses):
        """Evolve NEAT population to next generation."""
        # Assign fitnesses to genomes
        for i, (genome_id, genome) in enumerate(self.genomes):
            genome.fitness = fitnesses[i]

        # Let NEAT do reproduction
        self.neat_population.population = {gid: g for gid, g in self.genomes}

        # Run one generation of evolution
        self.neat_population.species.speciate(
            self.neat_config, self.neat_population.population, self.neat_population.generation)

        # Reproduce
        self.neat_population.population = self.neat_population.reproduction.reproduce(
            self.neat_config, self.neat_population.species,
            SHARED_CONFIG["pop_size"], self.neat_population.generation)

        # Check for stagnation and possibly adjust
        self.neat_population.species.speciate(
            self.neat_config, self.neat_population.population, self.neat_population.generation)

        self.neat_population.generation += 1

        # Get new genomes
        self.genomes = list(self.neat_population.population.items())
        self._create_networks_neat()

    def _evolve_ga(self, fitnesses):
        """Evolve static GA population to next generation."""
        pop_size = SHARED_CONFIG["pop_size"]
        elitism = SHARED_CONFIG["elitism"]
        survival_threshold = GA_CONFIG["survival_threshold"]

        # Selection and reproduction
        ranked = sorted(zip(self.genomes, fitnesses), key=lambda x: -x[1])
        new_population = [list(ranked[i][0]) for i in range(elitism)]
        parent_pool = [g for g, _ in ranked[:max(2, int(pop_size * survival_threshold))]]

        while len(new_population) < pop_size:
            p1, p2 = random.sample(parent_pool, 2)
            child = mutate(uniform_crossover(p1, p2))
            new_population.append(child)

        self.genomes = new_population
        self._create_networks_ga()

    def draw_game_panel(self):
        """Draw the swarm visualization panel with sprites (this IS the evaluation)."""
        # Calculate scaling to fit game in panel
        scale_x = GAME_PANEL_WIDTH / SCREEN_WIDTH
        scale_y = WINDOW_HEIGHT / SCREEN_HEIGHT
        scale = min(scale_x, scale_y)
        offset_x = (GAME_PANEL_WIDTH - SCREEN_WIDTH * scale) / 2
        offset_y = (WINDOW_HEIGHT - SCREEN_HEIGHT * scale) / 2

        # Create a clip rect for the game area
        game_rect = pygame.Rect(offset_x, offset_y, SCREEN_WIDTH * scale, SCREEN_HEIGHT * scale)

        # Draw background
        if self.sprites_loaded:
            scaled_bg = pygame.transform.scale(self.bg_img,
                (int(SCREEN_WIDTH * scale), int(SCREEN_HEIGHT * scale)))
            self.game_surface.fill(DARK_GRAY)
            self.game_surface.blit(scaled_bg, (offset_x, offset_y))
        else:
            self.game_surface.fill(SKY_BLUE)

        if self.swarm_game is None:
            # Show waiting message
            text = self.font_large.render("Select an experiment...", True, WHITE)
            self.game_surface.blit(text, (GAME_PANEL_WIDTH // 2 - text.get_width() // 2,
                                          WINDOW_HEIGHT // 2))
            return

        # Set clip rect to prevent drawing outside game area
        self.game_surface.set_clip(game_rect)

        # Draw pipes
        if self.sprites_loaded:
            scaled_pipe = pygame.transform.scale(self.pipe_img,
                (int(self.pipe_img.get_width() * scale), int(self.pipe_img.get_height() * scale)))
            for pipe in self.swarm_game.pipes:
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
                pipe_x = int(pipe.x * scale + offset_x)
                pipe_w = int(PIPE_WIDTH * scale)
                top_h = int(pipe.gap_top * scale)
                pygame.draw.rect(self.game_surface, PIPE_GREEN,
                               (pipe_x, int(offset_y), pipe_w, top_h))
                bot_y = int(pipe.gap_bottom * scale + offset_y)
                bot_h = int((FLOOR_Y - pipe.gap_bottom) * scale)
                pygame.draw.rect(self.game_surface, PIPE_GREEN,
                               (pipe_x, bot_y, pipe_w, bot_h))

        # Remove clip for birds/UI (birds can fly off-screen)
        self.game_surface.set_clip(None)

        # Get survival frames for transparency (shows who's doing well)
        survival = self.swarm_game.get_survival_frames()
        min_surv = min(survival) if survival else 0
        max_surv = max(survival) if survival else 1

        # Sort birds by survival so better ones draw on top
        bird_data = list(zip(self.swarm_game.birds, self.swarm_game.alive, survival))
        bird_data.sort(key=lambda x: x[2])

        for bird, alive, surv_frames in bird_data:
            if not alive:
                continue

            # Calculate alpha based on survival time
            if max_surv > min_surv:
                normalized = (surv_frames - min_surv) / (max_surv - min_surv)
            else:
                normalized = 1.0
            alpha = int(80 + normalized * 175)  # 80-255 range

            # Bird position
            bx = int(bird.x * scale + offset_x)
            by = int(bird.y * scale + offset_y)

            if self.sprites_loaded:
                scaled_bird = pygame.transform.scale(self.bird_img,
                    (int(self.bird_img.get_width() * scale),
                     int(self.bird_img.get_height() * scale)))
                rotated = pygame.transform.rotate(scaled_bird, -bird.vel * 3)
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

        # Draw base/floor (set clip again for this)
        self.game_surface.set_clip(game_rect)
        if self.sprites_loaded:
            scaled_base = pygame.transform.scale(self.base_img,
                (int(SCREEN_WIDTH * scale), int(self.base_img.get_height() * scale)))
            base_y = int(FLOOR_Y * scale + offset_y)
            self.game_surface.blit(scaled_base, (offset_x, base_y))
        else:
            floor_y = int(FLOOR_Y * scale + offset_y)
            pygame.draw.rect(self.game_surface, DARK_GREEN,
                            (offset_x, floor_y, int(SCREEN_WIDTH * scale), WINDOW_HEIGHT - floor_y))
        self.game_surface.set_clip(None)

        # Draw score
        best_score = max(self.swarm_game.scores) if self.swarm_game.scores else 0
        score_text = self.font_score.render(str(best_score), True, WHITE)
        self.game_surface.blit(score_text,
            (GAME_PANEL_WIDTH // 2 - score_text.get_width() // 2, int(50 * scale + offset_y)))

        # Draw info overlay
        alive_count = sum(self.swarm_game.alive)
        total_count = len(self.swarm_game.alive)
        progress = int(100 * self.generation / self.max_generations) if self.max_generations > 0 else 0
        status = "PAUSED" if self.paused else ("DONE" if self.training_done else f"{progress}%")
        pipes_mode = "Oscillating" if self.game_mode == "oscillating" else "Static"

        info_lines = [
            f"Gen: {self.generation}/{self.max_generations} | {self.experiment}",
            f"Alive: {alive_count}/{total_count} | Pipes: {pipes_mode}",
            f"Status: {status} | Speed: {'Normal' if self.speed == 1 else 'Max'}",
            f"[SPACE] Pause | [S] Speed | [O] Pipes | [R] Reset | [ESC] Quit"
        ]
        for i, line in enumerate(info_lines):
            text = self.font.render(line, True, WHITE)
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

        max_gen = max(max(gens), 1) if gens else 1
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

        # Input layer - correct labels: vel, dist (horizontal), y_off (vertical offset)
        input_labels = ["vel", "dist", "y_off"]
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
        # Correct labels: vel, dist (horizontal distance), y_off (vertical offset)
        layer_labels = [["vel", "dist", "y_off"]]
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
