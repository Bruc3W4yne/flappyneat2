import os
import pickle
import argparse
import pygame
import neat

from game import FlappyGame, SCREEN_WIDTH, SCREEN_HEIGHT, FLOOR_Y, PIPE_WIDTH, BIRD_RX, BIRD_RY, network_to_action
from trainer import build_neat_config, StaticNetwork
from config import EXPERIMENTS

ASSETS_PATH = os.path.join(os.path.dirname(__file__), "..", "imgs")


class Renderer:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Flappy Bird - NEAT")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("arial", 32)
        self.bg = pygame.transform.scale(
            pygame.image.load(os.path.join(ASSETS_PATH, "bg.png")).convert(),
            (SCREEN_WIDTH, SCREEN_HEIGHT))
        self.bird_img = pygame.transform.scale2x(
            pygame.image.load(os.path.join(ASSETS_PATH, "bird1.png")).convert_alpha())
        self.pipe_img = pygame.transform.scale2x(
            pygame.image.load(os.path.join(ASSETS_PATH, "pipe.png")).convert_alpha())
        self.base_img = pygame.transform.scale2x(
            pygame.image.load(os.path.join(ASSETS_PATH, "base.png")).convert_alpha())

    def render(self, game, show_hitbox=False):
        self.screen.blit(self.bg, (0, 0))
        for pipe in game.pipes:
            pipe_top = pygame.transform.flip(self.pipe_img, False, True)
            self.screen.blit(pipe_top, (pipe.x, pipe.gap_top - pipe_top.get_height()))
            self.screen.blit(self.pipe_img, (pipe.x, pipe.gap_bottom))
        bird = game.bird
        rotated = pygame.transform.rotate(self.bird_img, -bird.vel * 3)
        self.screen.blit(rotated, (bird.x, bird.y))
        self.screen.blit(self.base_img, (0, FLOOR_Y))
        score_text = self.font.render(str(game.score), True, (255, 255, 255))
        self.screen.blit(score_text, (SCREEN_WIDTH // 2 - score_text.get_width() // 2, 50))
        if show_hitbox:
            for pipe in game.pipes:
                pygame.draw.rect(self.screen, (255, 0, 0), (pipe.x, 0, PIPE_WIDTH, pipe.gap_top), 2)
                pygame.draw.rect(self.screen, (255, 0, 0), (pipe.x, pipe.gap_bottom, PIPE_WIDTH, SCREEN_HEIGHT - pipe.gap_bottom), 2)
            cx, cy = game.bird.center
            pygame.draw.ellipse(self.screen, (255, 0, 0), (cx - BIRD_RX, cy - BIRD_RY, BIRD_RX * 2, BIRD_RY * 2), 2)
        pygame.display.flip()
        self.clock.tick(60)


def load_network(winner_path, experiment_name):
    with open(winner_path, "rb") as f:
        data = pickle.load(f)
    exp = EXPERIMENTS[experiment_name]
    is_recurrent = not exp["feed_forward"]
    if isinstance(data, tuple):
        winner, config = data
        if is_recurrent:
            return neat.nn.RecurrentNetwork.create(winner, config), is_recurrent, (winner, config)
        return neat.nn.FeedForwardNetwork.create(winner, config), is_recurrent, (winner, config)
    elif isinstance(data, dict):
        net = StaticNetwork(data["genome"], data["hidden_layers"], data["is_recurrent"])
        return net, data["is_recurrent"], data
    raise ValueError(f"Unknown format in {winner_path}")


def watch_agent(winner_path, experiment_name):
    net, is_recurrent, raw_data = load_network(winner_path, experiment_name)
    renderer = Renderer()
    game = FlappyGame()
    show_hitbox = False
    print("Press H to toggle hitboxes, ESC to quit")

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_h:
                show_hitbox = not show_hitbox

        obs = game.get_observation()
        action = network_to_action(net.activate(obs))
        _, _, done, _ = game.step(action)
        renderer.render(game, show_hitbox)

        if done:
            pygame.time.wait(1000)
            game.reset()
            if is_recurrent:
                net, _, _ = load_network(winner_path, experiment_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("winner_file")
    parser.add_argument("-e", "--experiment", choices=list(EXPERIMENTS.keys()), required=True)
    args = parser.parse_args()
    watch_agent(args.winner_file, args.experiment)
