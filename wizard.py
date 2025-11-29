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
from config import BASE_CONFIG, EXPERIMENTS, SHARED_CONFIG, GA_CONFIG, MAX_FRAMES, PARAM_SETS
from trainer import (StaticNetwork, calculate_genome_size, random_genome,
                     mutate, uniform_crossover, binary_tournament_select, create_network)

WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 900
GAME_PANEL_WIDTH = int(WINDOW_WIDTH * 0.6)
RIGHT_PANEL_WIDTH = WINDOW_WIDTH - GAME_PANEL_WIDTH
RIGHT_PANEL_HEIGHT = WINDOW_HEIGHT // 2

ASSETS_PATH = os.path.join(os.path.dirname(__file__), "..", "imgs")

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

        self._load_sprites()

        self.game_surface = pygame.Surface((GAME_PANEL_WIDTH, WINDOW_HEIGHT))
        self.graph_surface = pygame.Surface((RIGHT_PANEL_WIDTH, RIGHT_PANEL_HEIGHT))
        self.network_surface = pygame.Surface((RIGHT_PANEL_WIDTH, RIGHT_PANEL_HEIGHT))

        self.experiment = None
        self.generation = 0
        self.max_generations = SHARED_CONFIG["generations"]
        self.fitness_history = []
        self.training_done = False
        self.paused = False
        self.speed = 1

        self.neat_config = None
        self.neat_population = None
        self.is_neat = True
        self.is_recurrent = False
        self.hidden_layers = []

        self.genomes = []
        self.networks = []
        self.current_best_genome = None

        self.swarm_game = None
        self.vis_frame = 0
        self.max_frames = MAX_FRAMES

        self.death_time = None
        self.death_delay = 1.0

        self.game_mode = game_module.GAME_MODE

    def _load_sprites(self):
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
        running = True

        while running:
            if self.experiment is None:
                self.show_experiment_selection()
                if self.experiment is None:
                    break
                continue

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif event.key == pygame.K_s:
                        self.speed = 0 if self.speed == 1 else 1
                    elif event.key == pygame.K_o:
                        self.game_mode = "oscillating" if self.game_mode == "standard" else "standard"
                        game_module.GAME_MODE = self.game_mode
                    elif event.key == pygame.K_r:
                        self._reset_state()
                        continue
                    elif event.key == pygame.K_ESCAPE:
                        running = False

            if not self.paused and not self.training_done:
                self._step_training()

            self.draw_game_panel()
            self.draw_graph_panel()
            self.draw_network_panel()

            self.screen.blit(self.game_surface, (0, 0))
            self.screen.blit(self.graph_surface, (GAME_PANEL_WIDTH, 0))
            self.screen.blit(self.network_surface, (GAME_PANEL_WIDTH, RIGHT_PANEL_HEIGHT))

            pygame.draw.line(self.screen, WHITE, (GAME_PANEL_WIDTH, 0), (GAME_PANEL_WIDTH, WINDOW_HEIGHT), 2)
            pygame.draw.line(self.screen, WHITE, (GAME_PANEL_WIDTH, RIGHT_PANEL_HEIGHT),
                           (WINDOW_WIDTH, RIGHT_PANEL_HEIGHT), 2)

            pygame.display.flip()
            self.clock.tick(60 if self.speed == 1 else 0)

        pygame.quit()

    def _reset_state(self):
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
        experiments = list(EXPERIMENTS.keys())
        selected = 0
        scroll_offset = 0
        visible_count = 12
        item_height = 40
        list_start_y = 150
        list_height = visible_count * item_height

        while True:
            self.screen.fill(DARK_GRAY)
            title = self.font_large.render("Select Experiment (12 configurations)", True, WHITE)
            self.screen.blit(title, (WINDOW_WIDTH // 2 - title.get_width() // 2, 80))

            subtitle = self.font.render("4 architectures × 3 parameter sets", True, LIGHT_GRAY)
            self.screen.blit(subtitle, (WINDOW_WIDTH // 2 - subtitle.get_width() // 2, 120))

            if selected < scroll_offset:
                scroll_offset = selected
            elif selected >= scroll_offset + visible_count:
                scroll_offset = selected - visible_count + 1

            for i in range(scroll_offset, min(scroll_offset + visible_count, len(experiments))):
                exp = experiments[i]
                display_idx = i - scroll_offset
                y_pos = list_start_y + display_idx * item_height

                color = YELLOW if i == selected else WHITE
                exp_info = EXPERIMENTS[exp]
                ff_rnn = "FF" if exp_info["feed_forward"] else "RNN"
                neat_ga = "NEAT" if exp_info.get("use_neat", True) else "Static"
                param_set = exp_info.get("param_set", "high")
                mutate_rate = PARAM_SETS[param_set]["weight_mutate_rate"]

                text = self.font.render(
                    f"{exp:20} | {neat_ga:6} {ff_rnn:3} | mutate={mutate_rate}",
                    True, color
                )
                self.screen.blit(text, (WINDOW_WIDTH // 2 - text.get_width() // 2, y_pos))

            if len(experiments) > visible_count:
                scroll_info = f"({selected + 1}/{len(experiments)})"
                scroll_text = self.font.render(scroll_info, True, LIGHT_GRAY)
                self.screen.blit(scroll_text, (WINDOW_WIDTH // 2 - scroll_text.get_width() // 2,
                                              list_start_y + list_height + 10))

            instructions = self.font.render("UP/DOWN to select, ENTER to start, ESC to quit", True, LIGHT_GRAY)
            self.screen.blit(instructions, (WINDOW_WIDTH // 2 - instructions.get_width() // 2,
                                           list_start_y + list_height + 40))

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
            self.genomes = list(self.neat_population.population.items())
            self._create_networks_neat()
        else:
            pop_size = SHARED_CONFIG["pop_size"]
            genome_size = calculate_genome_size(self.hidden_layers, self.is_recurrent)
            self.genomes = [random_genome(genome_size) for _ in range(pop_size)]
            self._create_networks_ga()

        self._start_generation_game()

    def _build_neat_config(self, experiment_name):
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

        config_text = self._dict_to_config_file(config_dict)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".cfg", delete=False) as f:
            f.write(config_text)
            config_path = f.name
        neat_config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                  neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
        os.unlink(config_path)
        return neat_config

    def _dict_to_config_file(self, config_dict):
        lines = []
        for section, params in config_dict.items():
            lines.append(f"[{section}]")
            for key, value in params.items():
                lines.append(f"{key} = {value}")
            lines.append("")
        return "\n".join(lines)

    def _create_networks_neat(self):
        self.networks = []
        for genome_id, genome in self.genomes:
            if self.is_recurrent:
                net = neat.nn.RecurrentNetwork.create(genome, self.neat_config)
            else:
                net = neat.nn.FeedForwardNetwork.create(genome, self.neat_config)
            self.networks.append(net)

    def _create_networks_ga(self):
        self.networks = []
        for genome in self.genomes:
            net = StaticNetwork(genome, self.hidden_layers, self.is_recurrent)
            self.networks.append(net)

    def _start_generation_game(self):
        num_birds = len(self.genomes)
        seed = int(time.time() * 1000) % (2**31)
        self.swarm_game = SwarmGame(num_birds, seed=seed)
        self.swarm_game.reset()
        self.vis_frame = 0
        self.death_time = None

        if self.is_recurrent:
            for net in self.networks:
                if hasattr(net, 'reset'):
                    net.reset()

    def _step_training(self):
        if self.swarm_game is None:
            return

        if self.swarm_game.all_dead or self.vis_frame >= self.max_frames:
            if self.death_time is None:
                self.death_time = time.time()
            elif time.time() - self.death_time >= self.death_delay:
                self._finish_generation()
            return

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
        os.makedirs("results", exist_ok=True)
        csv_path = f"results/{self.experiment}_wizard.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["generation", "best_fitness", "avg_fitness", "min_fitness"])
            writer.writerows(self.fitness_history)

    def _finish_generation(self):
        fitnesses = self.swarm_game.get_fitnesses()
        scores = self.swarm_game.scores
        best_score = max(scores)
        avg_score = sum(scores) / len(scores)
        min_score = min(scores)
        self.fitness_history.append((self.generation, best_score, avg_score, min_score))

        self._save_csv()

        best_idx = fitnesses.index(max(fitnesses))
        if self.is_neat:
            self.current_best_genome = self.genomes[best_idx][1]
        else:
            self.current_best_genome = self.genomes[best_idx]

        self.generation += 1
        if self.generation >= self.max_generations:
            self.training_done = True
            return

        if self.is_neat:
            self._evolve_neat(fitnesses)
        else:
            self._evolve_ga(fitnesses)

        self._start_generation_game()

    def _evolve_neat(self, fitnesses):
        for i, (genome_id, genome) in enumerate(self.genomes):
            genome.fitness = fitnesses[i]

        self.neat_population.population = {gid: g for gid, g in self.genomes}

        self.neat_population.species.speciate(
            self.neat_config, self.neat_population.population, self.neat_population.generation)

        self.neat_population.population = self.neat_population.reproduction.reproduce(
            self.neat_config, self.neat_population.species,
            SHARED_CONFIG["pop_size"], self.neat_population.generation)

        self.neat_population.species.speciate(
            self.neat_config, self.neat_population.population, self.neat_population.generation)

        self.neat_population.generation += 1

        self.genomes = list(self.neat_population.population.items())
        self._create_networks_neat()

    def _evolve_ga(self, fitnesses):
        pop_size = SHARED_CONFIG["pop_size"]
        elitism = SHARED_CONFIG["elitism"]

        exp = EXPERIMENTS[self.experiment]
        param_set_name = exp.get("param_set", "high")
        param_set = PARAM_SETS[param_set_name]
        weight_mutate_rate = param_set["weight_mutate_rate"]

        ranked = sorted(zip(self.genomes, fitnesses), key=lambda x: -x[1])
        new_population = [list(ranked[i][0]) for i in range(elitism)]

        while len(new_population) < pop_size:
            p1 = binary_tournament_select(ranked)
            p2 = binary_tournament_select(ranked)
            child = mutate(uniform_crossover(p1, p2), weight_mutate_rate)
            new_population.append(child)

        self.genomes = new_population
        self._create_networks_ga()

    def draw_game_panel(self):
        scale_x = GAME_PANEL_WIDTH / SCREEN_WIDTH
        scale_y = WINDOW_HEIGHT / SCREEN_HEIGHT
        scale = min(scale_x, scale_y)
        offset_x = (GAME_PANEL_WIDTH - SCREEN_WIDTH * scale) / 2
        offset_y = (WINDOW_HEIGHT - SCREEN_HEIGHT * scale) / 2

        game_rect = pygame.Rect(offset_x, offset_y, SCREEN_WIDTH * scale, SCREEN_HEIGHT * scale)

        if self.sprites_loaded:
            scaled_bg = pygame.transform.scale(self.bg_img,
                (int(SCREEN_WIDTH * scale), int(SCREEN_HEIGHT * scale)))
            self.game_surface.fill(DARK_GRAY)
            self.game_surface.blit(scaled_bg, (offset_x, offset_y))
        else:
            self.game_surface.fill(SKY_BLUE)

        if self.swarm_game is None:
            text = self.font_large.render("Select an experiment...", True, WHITE)
            self.game_surface.blit(text, (GAME_PANEL_WIDTH // 2 - text.get_width() // 2,
                                          WINDOW_HEIGHT // 2))
            return

        self.game_surface.set_clip(game_rect)

        if self.sprites_loaded:
            scaled_pipe = pygame.transform.scale(self.pipe_img,
                (int(self.pipe_img.get_width() * scale), int(self.pipe_img.get_height() * scale)))
            for pipe in self.swarm_game.pipes:
                pipe_x = int(pipe.x * scale + offset_x)
                pipe_top = pygame.transform.flip(scaled_pipe, False, True)
                top_y = int(pipe.gap_top * scale + offset_y) - pipe_top.get_height()
                self.game_surface.blit(pipe_top, (pipe_x, top_y))
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

        self.game_surface.set_clip(None)

        survival = self.swarm_game.get_survival_frames()
        min_surv = min(survival) if survival else 0
        max_surv = max(survival) if survival else 1

        bird_data = list(zip(self.swarm_game.birds, self.swarm_game.alive, survival))
        bird_data.sort(key=lambda x: x[2])

        for bird, alive, surv_frames in bird_data:
            if not alive:
                continue

            if max_surv > min_surv:
                normalized = (surv_frames - min_surv) / (max_surv - min_surv)
            else:
                normalized = 1.0
            alpha = int(80 + normalized * 175)

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
                rx = int(BIRD_RX * scale)
                ry = int(BIRD_RY * scale)
                cx = bx + int(40 * scale)
                cy = by + int(30 * scale)
                bird_surf = pygame.Surface((rx * 2, ry * 2), pygame.SRCALPHA)
                color = (*YELLOW[:3], alpha)
                pygame.draw.ellipse(bird_surf, color, (0, 0, rx * 2, ry * 2))
                self.game_surface.blit(bird_surf, (cx - rx, cy - ry))

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

        best_score = max(self.swarm_game.scores) if self.swarm_game.scores else 0
        score_text = self.font_score.render(str(best_score), True, WHITE)
        self.game_surface.blit(score_text,
            (GAME_PANEL_WIDTH // 2 - score_text.get_width() // 2, int(50 * scale + offset_y)))

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
        self.graph_surface.fill(DARK_GRAY)

        title = self.font.render("Fitness Over Generations", True, WHITE)
        self.graph_surface.blit(title, (RIGHT_PANEL_WIDTH // 2 - title.get_width() // 2, 10))

        if not self.fitness_history:
            text = self.font.render("No data yet...", True, LIGHT_GRAY)
            self.graph_surface.blit(text, (RIGHT_PANEL_WIDTH // 2 - text.get_width() // 2,
                                          RIGHT_PANEL_HEIGHT // 2))
            return

        margin = 50
        graph_x = margin
        graph_y = 40
        graph_w = RIGHT_PANEL_WIDTH - margin * 2
        graph_h = RIGHT_PANEL_HEIGHT - margin - graph_y

        pygame.draw.line(self.graph_surface, WHITE,
                        (graph_x, graph_y + graph_h), (graph_x + graph_w, graph_y + graph_h), 1)
        pygame.draw.line(self.graph_surface, WHITE,
                        (graph_x, graph_y), (graph_x, graph_y + graph_h), 1)

        gens = [h[0] for h in self.fitness_history]
        bests = [h[1] for h in self.fitness_history]
        avgs = [h[2] for h in self.fitness_history]

        max_gen = max(max(gens), 1) if gens else 1
        max_fit = max(max(bests), max(avgs)) if bests else 1
        max_fit = max(max_fit, 1)

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

        if bests:
            current_best = bests[-1]
            current_avg = avgs[-1]
            stats_text = f"Best: {current_best:.1f} | Avg: {current_avg:.1f}"
            stats_render = self.font.render(stats_text, True, WHITE)
            self.graph_surface.blit(stats_render, (graph_x, graph_y + graph_h + 25))

        pygame.draw.line(self.graph_surface, GREEN, (graph_x, graph_y - 15),
                        (graph_x + 20, graph_y - 15), 2)
        best_label = self.font.render("Best", True, GREEN)
        self.graph_surface.blit(best_label, (graph_x + 25, graph_y - 22))

        pygame.draw.line(self.graph_surface, BLUE, (graph_x + 80, graph_y - 15),
                        (graph_x + 100, graph_y - 15), 2)
        avg_label = self.font.render("Avg", True, BLUE)
        self.graph_surface.blit(avg_label, (graph_x + 105, graph_y - 22))

        y_label = self.font.render(f"{max_fit:.0f}", True, WHITE)
        self.graph_surface.blit(y_label, (5, graph_y))
        x_label = self.font.render(f"Gen {max_gen}", True, WHITE)
        self.graph_surface.blit(x_label, (graph_x + graph_w - 50, graph_y + graph_h + 5))

    def draw_network_panel(self):
        self.network_surface.fill(DARK_GRAY)

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
        genome = self.current_best_genome
        margin = 60
        width = RIGHT_PANEL_WIDTH - margin * 2
        height = RIGHT_PANEL_HEIGHT - margin * 2 - 20
        cx = RIGHT_PANEL_WIDTH // 2
        cy = RIGHT_PANEL_HEIGHT // 2 + 10

        input_nodes = [-1, -2, -3]
        output_nodes = [0]
        hidden_nodes = [k for k in genome.nodes.keys() if k not in input_nodes + output_nodes]

        if hidden_nodes:
            num_layers = 3
        else:
            num_layers = 2

        node_pos = {}
        layer_x = lambda l: margin + (l / (num_layers - 1)) * width if num_layers > 1 else cx

        input_labels = ["vel", "dist", "y_off"]
        for i, node in enumerate(input_nodes):
            y = cy - height // 4 + (i - 1) * (height // 3)
            node_pos[node] = (layer_x(0), y, input_labels[i])

        for i, node in enumerate(hidden_nodes):
            y = cy - (len(hidden_nodes) - 1) * 30 // 2 + i * 30
            node_pos[node] = (layer_x(1), y, f"h{node}")

        node_pos[0] = (layer_x(num_layers - 1), cy, "jump")

        for (src, dst), conn in genome.connections.items():
            if not conn.enabled:
                continue
            if src not in node_pos or dst not in node_pos:
                continue

            x1, y1, _ = node_pos[src]
            x2, y2, _ = node_pos[dst]

            weight = conn.weight
            if weight > 0:
                color = GREEN
            else:
                color = RED

            thickness = max(1, min(5, int(abs(weight))))

            pygame.draw.line(self.network_surface, color, (x1, y1), (x2, y2), thickness)

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

            text = self.font.render(label, True, WHITE)
            self.network_surface.blit(text, (x - text.get_width() // 2, y - text.get_height() // 2))

        info = f"Nodes: {len(genome.nodes)} | Connections: {sum(1 for c in genome.connections.values() if c.enabled)}"
        info_text = self.font.render(info, True, LIGHT_GRAY)
        self.network_surface.blit(info_text, (margin, RIGHT_PANEL_HEIGHT - 30))

    def _draw_static_network(self):
        margin = 60
        width = RIGHT_PANEL_WIDTH - margin * 2
        height = RIGHT_PANEL_HEIGHT - margin * 2 - 20
        cx = RIGHT_PANEL_WIDTH // 2
        cy = RIGHT_PANEL_HEIGHT // 2 + 10

        layer_sizes = [3] + self.hidden_layers + [1]
        num_layers = len(layer_sizes)

        node_pos = {}
        layer_labels = [["vel", "dist", "y_off"]]
        for i, size in enumerate(self.hidden_layers):
            layer_labels.append([f"h{j}" for j in range(size)])
        layer_labels.append(["jump"])

        for layer_idx, (size, labels) in enumerate(zip(layer_sizes, layer_labels)):
            x = margin + (layer_idx / (num_layers - 1)) * width if num_layers > 1 else cx
            for node_idx in range(size):
                y = cy - (size - 1) * 30 // 2 + node_idx * 30
                node_pos[(layer_idx, node_idx)] = (x, y, labels[node_idx])

        for l in range(num_layers - 1):
            for i in range(layer_sizes[l]):
                for j in range(layer_sizes[l + 1]):
                    x1, y1, _ = node_pos[(l, i)]
                    x2, y2, _ = node_pos[(l + 1, j)]
                    pygame.draw.line(self.network_surface, LIGHT_GRAY, (x1, y1), (x2, y2), 1)

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
