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
        if min_center > max_center:
            max_center = min_center
        self._base_gap_center = random.randint(int(min_center), int(max_center))
        self._gap_center = self._base_gap_center
        self.x = SCREEN_WIDTH + 100
        self.gap_top = self._gap_center - PIPE_GAP // 2
        self.gap_bottom = self._gap_center + PIPE_GAP // 2
        self.passed = False
        self.frame_count = 0
        self.phase_offset = random.uniform(0, 2 * math.pi)

    def update(self):
        self.x -= PIPE_VEL
        self.frame_count += 1
        if GAME_MODE == "oscillating":
            offset = math.sin(self.frame_count * 0.05 + self.phase_offset) * 30
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


def network_to_action(output):
    return output[0] > 0.0 if isinstance(output, (list, tuple)) else output > 0.0


class FlappyGame:
    def __init__(self, num_birds=1, seed=None):
        self.num_birds = num_birds
        self.seed = seed
        self.reset()

    def reset(self):
        if self.seed is not None:
            random.seed(self.seed)
        self.birds = [Bird() for _ in range(self.num_birds)]
        self.alive = [True] * self.num_birds
        self.scores = [0] * self.num_birds
        self.death_frames = [None] * self.num_birds
        self.pipes = [Pipe()]
        self.frame = 0
        self.done = False
        return self.get_observation() if self.num_birds == 1 else self.get_observations()

    @property
    def bird(self):
        return self.birds[0]

    @property
    def score(self):
        return self.scores[0]

    def get_observation(self):
        return self._get_bird_observation(0)

    def step(self, action):
        if self.num_birds == 1:
            return self._step_single(action)
        else:
            return self._step_swarm(action)

    def _step_single(self, action):
        if self.done:
            return self.get_observation(), 0, True, self.score
        self.frame += 1
        if action:
            self.bird.jump()
        self.bird.update()
        reward = self._update_pipes_single()
        self.done = self._check_collision(self.bird)
        return self.get_observation(), reward, self.done, self.score

    def _update_pipes_single(self):
        reward = 0
        for pipe in self.pipes:
            pipe.update()
            if not pipe.passed and pipe.x + PIPE_WIDTH < self.bird.x:
                pipe.passed = True
                self.scores[0] += 1
                reward = 1
        self.pipes = [p for p in self.pipes if not p.off_screen]
        if self.pipes[-1].x < SCREEN_WIDTH - 200:
            self.pipes.append(Pipe(self.pipes[-1].gap_center))
        return reward

    def get_observations(self):
        return [self._get_bird_observation(i) for i in range(self.num_birds)]

    def _step_swarm(self, actions):
        self.frame += 1
        for i, (bird, action) in enumerate(zip(self.birds, actions)):
            if not self.alive[i]:
                continue
            if action:
                bird.jump()
            bird.update()
        self._update_pipes_swarm()
        for i, bird in enumerate(self.birds):
            if self.alive[i] and self._check_collision(bird):
                self.alive[i] = False
                self.death_frames[i] = self.frame
        return self.get_observations(), self.alive.copy(), self.scores.copy()

    def _update_pipes_swarm(self):
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

    def _get_bird_observation(self, idx):
        if not self.alive[idx]:
            return (0.0, 1.0, 0.0)
        pipe = self._next_pipe()
        if not pipe:
            return (0.0, 1.0, 0.0)
        bird = self.birds[idx]
        cx, cy = bird.center
        norm_vel = max(-1.0, min(1.0, bird.vel / MAX_VEL))
        pipe_dist = max(0.0, min(1.0, (pipe.x - cx) / SCREEN_WIDTH))
        y_offset = max(-1.0, min(1.0, (pipe.gap_bottom - cy) / SCREEN_HEIGHT))
        return (norm_vel, pipe_dist, y_offset)

    def _next_pipe(self):
        for pipe in self.pipes:
            if pipe.x + PIPE_WIDTH > BIRD_X:
                return pipe
        return None

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
        return [
            self.death_frames[i] if self.death_frames[i] is not None else self.frame
            for i in range(self.num_birds)
        ]

    def get_fitnesses(self):
        survival = self.get_survival_frames()
        return [survival[i] + self.scores[i] * 1000 for i in range(self.num_birds)]


SwarmGame = FlappyGame


def play_game(network, game, max_frames=5000):
    game.reset()
    frames = 0
    for frames in range(1, max_frames + 1):
        obs = game.get_observation()
        action = network_to_action(network.activate(obs))
        _, _, done, _ = game.step(action)
        if done:
            break
    return frames + game.score * 1000
