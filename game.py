import random
import math
from config import GAME_MODE

SCREEN_WIDTH = 600
SCREEN_HEIGHT = 800
FLOOR_Y = 730
BIRD_X = 230
BIRD_START_Y = 350
BIRD_WIDTH = 34
BIRD_HEIGHT = 24
BIRD_RX = 30
BIRD_RY = 22
GRAVITY = 0.8
JUMP_VEL = -12
MAX_VEL = 15
PIPE_WIDTH = 104
PIPE_GAP = 160
PIPE_VEL = 5
MAX_GAP_CHANGE = 150


class Bird:
    def __init__(self):
        self.x = BIRD_X
        self.y = BIRD_START_Y
        self.vel = 0

    def jump(self):
        self.vel = JUMP_VEL

    def update(self):
        self.vel = min(self.vel + GRAVITY, MAX_VEL)
        self.y += self.vel

    @property
    def center(self):
        return (self.x + 40, self.y + 30)


class Pipe:
    def __init__(self, prev_gap_center=None):
        min_center = PIPE_GAP // 2 + 50
        max_center = FLOOR_Y - PIPE_GAP // 2 - 50
        if prev_gap_center is not None:
            min_center = max(min_center, prev_gap_center - MAX_GAP_CHANGE)
            max_center = min(max_center, prev_gap_center + MAX_GAP_CHANGE)
        self._base_gap_center = random.randint(int(min_center), int(max_center))
        self._gap_center = self._base_gap_center
        self.x = SCREEN_WIDTH + 100
        self.gap_top = self._gap_center - PIPE_GAP // 2
        self.gap_bottom = self._gap_center + PIPE_GAP // 2
        self.passed = False
        self.frame_count = 0

    def update(self):
        self.x -= PIPE_VEL
        self.frame_count += 1
        if GAME_MODE == "oscillating":
            offset = math.sin(self.frame_count * 0.05) * 30
            self._gap_center = self._base_gap_center + offset
            self.gap_top = self._gap_center - PIPE_GAP // 2
            self.gap_bottom = self._gap_center + PIPE_GAP // 2

    @property
    def off_screen(self):
        return self.x < -PIPE_WIDTH

    @property
    def gap_center(self):
        return self._gap_center

    @property
    def top_rect(self):
        return (self.x, 0, PIPE_WIDTH, self.gap_top)

    @property
    def bottom_rect(self):
        return (self.x, self.gap_bottom, PIPE_WIDTH, SCREEN_HEIGHT - self.gap_bottom)


def ellipse_rect_collide(cx, cy, rx, ry, rect):
    rect_x, rect_y, rect_w, rect_h = rect
    closest_x = max(rect_x, min(cx, rect_x + rect_w))
    closest_y = max(rect_y, min(cy, rect_y + rect_h))
    dx = (closest_x - cx) / rx
    dy = (closest_y - cy) / ry
    return (dx * dx + dy * dy) < 1


class FlappyGame:
    def __init__(self, seed=None):
        self.seed = seed
        self.reset()

    def reset(self):
        if self.seed:
            random.seed(self.seed)
        self.bird = Bird()
        self.pipes = [Pipe()]
        self.score = 0
        self.done = False
        return self.get_observation()

    def get_observation(self):
        pipe = self._next_pipe()
        if not pipe:
            return (0.0, 1.0, 0.0)
        cx, cy = self.bird.center
        norm_vel = max(-1.0, min(1.0, self.bird.vel / MAX_VEL))
        i = max(0.0, min(1.0, (pipe.x - cx) / SCREEN_WIDTH))
        j = max(-1.0, min(1.0, (pipe.gap_bottom - cy) / SCREEN_HEIGHT))
        return (norm_vel, i, j)

    def step(self, action):
        if self.done:
            return self.get_observation(), 0, True, self.score
        if action:
            self.bird.jump()
        self.bird.update()
        reward = self._update_pipes()
        self.done = self._check_collision()
        return self.get_observation(), reward, self.done, self.score

    def _next_pipe(self):
        for pipe in self.pipes:
            if pipe.x + PIPE_WIDTH > self.bird.x:
                return pipe
        return None

    def _update_pipes(self):
        reward = 0
        for pipe in self.pipes:
            pipe.update()
            if not pipe.passed and pipe.x + PIPE_WIDTH < self.bird.x:
                pipe.passed = True
                self.score += 1
                reward = 1
        self.pipes = [p for p in self.pipes if not p.off_screen]
        if self.pipes[-1].x < SCREEN_WIDTH - 200:
            self.pipes.append(Pipe(self.pipes[-1].gap_center))
        return reward

    def _check_collision(self):
        cx, cy = self.bird.center
        if cy + BIRD_RY >= FLOOR_Y or cy - BIRD_RY <= 0:
            return True
        for pipe in self.pipes:
            if ellipse_rect_collide(cx, cy, BIRD_RX, BIRD_RY, pipe.top_rect):
                return True
            if ellipse_rect_collide(cx, cy, BIRD_RX, BIRD_RY, pipe.bottom_rect):
                return True
        return False


def play_game(network, game, max_frames=5000):
    game.reset()
    for _ in range(max_frames):
        obs = game.get_observation()
        output = network.activate(obs)
        action = output[0] > 0.0 if isinstance(output, (list, tuple)) else output > 0.0
        _, _, done, _ = game.step(action)
        if done:
            break
    return game.score


class SwarmGame:
    """Multi-bird game for visualization - all birds share the same pipes."""

    def __init__(self, num_birds, seed=None):
        self.num_birds = num_birds
        self.seed = seed
        self.reset()

    def reset(self):
        if self.seed is not None:
            random.seed(self.seed)
        self.birds = [Bird() for _ in range(self.num_birds)]
        self.alive = [True] * self.num_birds
        self.scores = [0] * self.num_birds
        self.death_frames = [None] * self.num_birds  # Track when each bird died
        self.pipes = [Pipe()]
        self.frame = 0
        return self.get_observations()

    def get_observations(self):
        """Get observation for each bird."""
        observations = []
        pipe = self._next_pipe()
        for i, bird in enumerate(self.birds):
            if not self.alive[i]:
                observations.append((0.0, 1.0, 0.0))
                continue
            if not pipe:
                observations.append((0.0, 1.0, 0.0))
                continue
            cx, cy = bird.center
            norm_vel = max(-1.0, min(1.0, bird.vel / MAX_VEL))
            pipe_i = max(0.0, min(1.0, (pipe.x - cx) / SCREEN_WIDTH))
            j = max(-1.0, min(1.0, (pipe.gap_bottom - cy) / SCREEN_HEIGHT))
            observations.append((norm_vel, pipe_i, j))
        return observations

    def step(self, actions):
        """Step all birds with their respective actions."""
        self.frame += 1

        # Update each bird
        for i, (bird, action) in enumerate(zip(self.birds, actions)):
            if not self.alive[i]:
                continue
            if action:
                bird.jump()
            bird.update()

        # Update pipes (shared)
        self._update_pipes()

        # Check collisions
        for i, bird in enumerate(self.birds):
            if self.alive[i] and self._check_collision(bird):
                self.alive[i] = False
                self.death_frames[i] = self.frame  # Record when this bird died

        return self.get_observations(), self.alive.copy(), self.scores.copy()

    def _next_pipe(self):
        for pipe in self.pipes:
            if pipe.x + PIPE_WIDTH > BIRD_X:
                return pipe
        return None

    def _update_pipes(self):
        for pipe in self.pipes:
            pipe.update()
            if not pipe.passed and pipe.x + PIPE_WIDTH < BIRD_X:
                pipe.passed = True
                for i in range(self.num_birds):
                    if self.alive[i]:
                        self.scores[i] += 1
        self.pipes = [p for p in self.pipes if not p.off_screen]
        if self.pipes[-1].x < SCREEN_WIDTH - 200:
            self.pipes.append(Pipe(self.pipes[-1].gap_center))

    def _check_collision(self, bird):
        cx, cy = bird.center
        if cy + BIRD_RY >= FLOOR_Y or cy - BIRD_RY <= 0:
            return True
        for pipe in self.pipes:
            if ellipse_rect_collide(cx, cy, BIRD_RX, BIRD_RY, pipe.top_rect):
                return True
            if ellipse_rect_collide(cx, cy, BIRD_RX, BIRD_RY, pipe.bottom_rect):
                return True
        return False

    @property
    def all_dead(self):
        return not any(self.alive)

    def get_survival_frames(self):
        """Get survival frames for each bird (death_frame if dead, current frame if alive)."""
        return [
            self.death_frames[i] if self.death_frames[i] is not None else self.frame
            for i in range(self.num_birds)
        ]

    def get_fitnesses(self):
        """Get composite fitness: survival_frames + score * 1000.

        This ensures there's always a gradient even when scores are 0.
        Passing a pipe (score +1) is worth 1000 frames of survival (~17 seconds at 60fps).
        """
        survival = self.get_survival_frames()
        return [survival[i] + self.scores[i] * 1000 for i in range(self.num_birds)]
