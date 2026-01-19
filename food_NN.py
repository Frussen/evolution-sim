import pygame
import numpy as np
import random
from collections import deque

pygame.init()
WIDTH, HEIGHT = 1200, 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Neural Predator-Prey + Grass & Meat – Toroidal")
clock = pygame.time.Clock()

# ─── Parametri ────────────────────────────────────────────────────────────────
N_PREY_START       = 30
N_PRED_START       = 10
MAX_SPEED_PREY     = 3.0
MAX_SPEED_PRED     = 3.5
VISION_RANGE_PREY  = 150
VISION_RANGE_PRED  = 280
EAT_DISTANCE       = 14

GRASS_SPAWN_RATE   = 1       # base probabilità per frame (~0.48 erba/sec a 60 fps)
GRASS_PER_SPAWN    = 1
GRASS_ENERGY       = 0.28

MEAT_LIFETIME      = 55000
MEAT_ENERGY        = 1.40

PRED_REPRO_PROB       = 0.004
PRED_MIN_FOR_REPRO    = 2
PRED_ENERGY_TO_REPRO  = 0.88
PRED_MUTATION_RATE    = 0.055

PREY_REPRO_PROB       = 0.012
PREY_MIN_FOR_REPRO    = 2
PREY_ENERGY_TO_REPRO  = 0.74
PREY_MUTATION_RATE    = 0.018

NN_HIDDEN          = 128
INPUT_SIZE         = 70
FOOD_MAX           = 9000

INITIAL_GRASS_TOTAL = 700


# ─── Funzioni di attivazione ──────────────────────────────────────────────────
def relu(x): return np.maximum(0, x)
def tanh(x): return np.tanh(x)


# ─── Rete neurale ─────────────────────────────────────────────────────────────
class SimpleNN:
    def __init__(self, input_size, hidden_size, output_size, scale=1.55):
        self.W1 = np.random.randn(input_size, hidden_size) * scale
        self.b1 = np.random.randn(hidden_size) * 0.25
        self.W2 = np.random.randn(hidden_size, output_size) * scale
        self.b2 = np.random.randn(output_size) * 0.15

    def forward(self, x):
        h = relu(np.dot(x, self.W1) + self.b1)
        out = tanh(np.dot(h, self.W2) + self.b2)
        return out

    def blend_with(self, other, mutation_rate, alpha=0.5):
        child = SimpleNN(self.W1.shape[0], self.W1.shape[1], self.W2.shape[1])
        child.W1 = alpha * self.W1 + (1 - alpha) * other.W1 + np.random.randn(*self.W1.shape) * mutation_rate
        child.b1 = alpha * self.b1 + (1 - alpha) * other.b1 + np.random.randn(*self.b1.shape) * mutation_rate * 0.35
        child.W2 = alpha * self.W2 + (1 - alpha) * other.W2 + np.random.randn(*self.W2.shape) * mutation_rate
        child.b2 = alpha * self.b2 + (1 - alpha) * other.b2 + np.random.randn(*self.b2.shape) * mutation_rate * 0.35
        return child


# ─── Cibo ─────────────────────────────────────────────────────────────────────
class Food:
    def __init__(self, x, y, kind="grass"):
        self.pos = np.array([x, y], dtype=float)
        self.kind = kind
        self.birth_time = pygame.time.get_ticks()
        self.eaten = False

    def is_alive(self):
        if self.kind == "meat":
            return pygame.time.get_ticks() - self.birth_time < MEAT_LIFETIME
        return True

    def color(self):
        if self.kind == "grass":
            return (50, 230, 50)
        return (255, 220, 40)


# ─── Agente ───────────────────────────────────────────────────────────────────
class Agent:
    def __init__(self, x, y, type_="prey", nn=None):
        self.pos = np.array([x, y], dtype=float)
        self.vel = np.random.randn(2) * 0.6
        self.type = type_
        self.energy = 1.0 if type_ == "prey" else 0.85
        self.last_eat_time = 0
        if nn is None:
            self.nn = SimpleNN(INPUT_SIZE, NN_HIDDEN, 2)
        else:
            self.nn = nn

    @property
    def speed(self):
        return np.linalg.norm(self.vel)

    def normalize_vel(self, max_speed):
        s = self.speed
        if s > max_speed:
            self.vel *= max_speed / s
        elif s < 0.18:
            self.vel += np.random.randn(2) * 0.38


# ─── Input per la rete ────────────────────────────────────────────────────────
def get_nn_input(agent, prey_list, pred_list, foods):
    my_pos = agent.pos
    my_vel = agent.vel
    is_pred = agent.type == "pred"
    vision = VISION_RANGE_PRED if is_pred else VISION_RANGE_PREY

    pred_info = []
    for p in pred_list:
        if p is agent: continue
        diff = p.pos - my_pos
        d = np.linalg.norm(diff)
        if d < vision * 1.5:
            norm_dist = np.clip(1 - d / vision, 0, 1)
            norm_dir = diff / (d + 1e-6)
            rel_vel = (p.vel - my_vel) / MAX_SPEED_PRED
            pred_info.append((norm_dist, norm_dir[0], norm_dir[1], rel_vel[0], rel_vel[1]))

    pred_info.sort(key=lambda x: -x[0])
    pred_info = pred_info[:6]
    pred_features = [0.0] * 30
    for i, info in enumerate(pred_info):
        pred_features[i*5:i*5+5] = info

    prey_info = []
    for p in prey_list:
        if p is agent: continue
        diff = p.pos - my_pos
        d = np.linalg.norm(diff)
        if d < vision * 1.4:
            norm_dist = np.clip(1 - d / vision, 0, 1)
            norm_dir = diff / (d + 1e-6)
            rel_vel = (p.vel - my_vel) / MAX_SPEED_PREY
            prey_info.append((norm_dist, norm_dir[0], norm_dir[1], rel_vel[0], rel_vel[1]))

    prey_info.sort(key=lambda x: -x[0])
    prey_info = prey_info[:8]
    prey_features = [0.0] * 40
    for i, info in enumerate(prey_info):
        prey_features[i*5:i*5+5] = info

    food_features = [0.0] * 5
    target_kind = "meat" if is_pred else "grass"
    candidates = [f for f in foods if f.kind == target_kind and f.is_alive()]
    if candidates:
        closest = min(candidates, key=lambda f: np.linalg.norm(f.pos - my_pos))
        dist = np.linalg.norm(closest.pos - my_pos)
        norm_dist = np.clip(1 - dist / vision, 0, 1)
        diff = closest.pos - my_pos
        norm_dir = diff / (dist + 1e-6)
        food_features = [norm_dist, norm_dir[0], norm_dir[1], 0.0, 0.0]

    inputs = np.concatenate([
        pred_features, prey_features, food_features,
        [agent.energy * 2 - 1],
        agent.vel / MAX_SPEED_PRED,
        agent.pos / np.array([WIDTH, HEIGHT]) * 2 - 1,
        [np.cos(pygame.time.get_ticks() * 0.0009)]
    ])[:70].astype(np.float32)

    return inputs


# ─── Colori ───────────────────────────────────────────────────────────────────
def energy_color(energy, is_prey=True):
    energy = max(0.0, min(1.0, energy))
    if is_prey:
        b = int(130 + 125 * energy)
        g = int(70 + 70 * energy)
        r = int(30 * (1 - energy))
    else:
        if energy > 0.82:
            r, g, b = 255, 140, 60
        elif energy > 0.50:
            r, g, b = 240, 70, 50
        else:
            r, g, b = 160, 30, 30
    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))
    return (r, g, b)


def draw_agent(screen, agent):
    pos = agent.pos.astype(int)
    color = energy_color(agent.energy, agent.type == "prey")

    angle = np.arctan2(agent.vel[1], agent.vel[0])
    size = 10 if agent.type == "pred" else 6 + 5 * agent.energy
    
    pts = [
        pos + np.array([np.cos(angle),      np.sin(angle)])      * size,
        pos + np.array([np.cos(angle+2.35), np.sin(angle+2.35)]) * size * 0.65,
        pos + np.array([np.cos(angle-2.35), np.sin(angle-2.35)]) * size * 0.65,
    ]
    pygame.draw.polygon(screen, color, pts)
    pygame.draw.polygon(screen, (0,0,0), pts, 1)


def draw_food(screen, food):
    if food.eaten or not food.is_alive():
        return
    pos = food.pos.astype(int)
    color = food.color()
    size = 6 if food.kind == "grass" else 8
    pygame.draw.circle(screen, color, pos, size)
    pygame.draw.circle(screen, (0,0,0), pos, size, 1)


# ─── Inizializzazione ─────────────────────────────────────────────────────────
prey_list = [Agent(random.uniform(0, WIDTH), random.uniform(0, HEIGHT), "prey") for _ in range(N_PREY_START)]
pred_list = [Agent(random.uniform(0, WIDTH), random.uniform(0, HEIGHT), "pred") for _ in range(N_PRED_START)]
all_agents = prey_list + pred_list

foods = deque(maxlen=FOOD_MAX)

for _ in range(INITIAL_GRASS_TOTAL):
    foods.append(Food(random.uniform(0, WIDTH), random.uniform(0, HEIGHT), "grass"))

font = pygame.font.SysFont("consolas", 18)
start_time = pygame.time.get_ticks()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            running = False

    dt = clock.tick(60) / 1000.0
    current_time = pygame.time.get_ticks()

    fade = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    fade.fill((4, 12, 6, 12))
    screen.blit(fade, (0, 0))

    # ─── SPAWN ERBA ───────────────────────────────────────────────────────────
    if random.random() < GRASS_SPAWN_RATE:
        for _ in range(GRASS_PER_SPAWN):
            foods.append(Food(
                random.uniform(0, WIDTH),
                random.uniform(0, HEIGHT),
                "grass"
            ))

    # ─── Update predatori ─────────────────────────────────────────────────────
    for pred in pred_list[:]:
        inputs = get_nn_input(pred, prey_list, pred_list, foods)
        accel = pred.nn.forward(inputs) * 1.55
        pred.vel += accel
        pred.normalize_vel(MAX_SPEED_PRED)

        drain = 0.00065 * dt * 60
        if current_time - pred.last_eat_time < 14000:
            drain *= 0.38
        pred.energy = max(0.035, pred.energy - drain)

        if pred.speed > 2.6:
            pred.energy -= 0.00022 * pred.speed**2 * dt * 60

    dead_preds = [p for p in pred_list if p.energy <= 0.035]
    for p in dead_preds:
        pred_list.remove(p)
        all_agents.remove(p)

    # ─── Update prede ─────────────────────────────────────────────────────────
    for pr in prey_list[:]:
        inputs = get_nn_input(pr, prey_list, pred_list, foods)
        accel = pr.nn.forward(inputs) * 1.15
        pr.vel += accel
        pr.normalize_vel(MAX_SPEED_PREY)

        pr.energy += 0.0035 * dt * 60
        pr.energy = min(1.0, pr.energy)
        pr.energy -= 0.0012 * dt * 60

        if pr.speed > 1.2:
            pr.energy -= 0.0009 * (pr.speed ** 2) * dt * 60

        if pr.energy <= 0.03:
            foods.append(Food(pr.pos[0], pr.pos[1], "meat"))
            prey_list.remove(pr)
            all_agents.remove(pr)

    # Movimento (toroidale)
    for a in all_agents:
        a.pos += a.vel * 60 * dt
        a.pos[0] %= WIDTH
        a.pos[1] %= HEIGHT

    # Predazione
    eaten_prey = []
    for pred in pred_list:
        if pred.energy < 0.12: continue
        for pr in prey_list:
            if np.linalg.norm(pr.pos - pred.pos) < EAT_DISTANCE:
                eaten_prey.append(pr)
                pred.energy = min(1.85, pred.energy + 1.15)
                pred.last_eat_time = current_time
                foods.append(Food(pr.pos[0], pr.pos[1], "meat"))
                break

    for dead in set(eaten_prey):
        if dead in prey_list:
            prey_list.remove(dead)
            all_agents.remove(dead)

    # Prede mangiano erba
    for pr in prey_list:
        for f in foods:
            if f.kind == "grass" and not f.eaten and f.is_alive():
                if np.linalg.norm(pr.pos - f.pos) < EAT_DISTANCE + 5:
                    pr.energy = min(1.0, pr.energy + GRASS_ENERGY)
                    pr.last_eat_time = current_time
                    f.eaten = True
                    break

    # Predatori mangiano carne
    for pred in pred_list:
        for f in foods:
            if f.kind == "meat" and not f.eaten and f.is_alive():
                if np.linalg.norm(pred.pos - f.pos) < EAT_DISTANCE + 7:
                    pred.energy = min(1.85, pred.energy + MEAT_ENERGY)
                    pred.last_eat_time = current_time
                    f.eaten = True
                    break

    # Pulizia cibi morti/mangiati
    to_remove = [f for f in foods if f.eaten or (f.kind == "meat" and not f.is_alive())]
    for f in to_remove:
        foods.remove(f)

    # ─── Riproduzione ─────────────────────────────────────────────────────────
    # Predatori
    pred_repro_p = 0.08 if len(pred_list) < 6 else 0.04 if len(pred_list) < 12 else PRED_REPRO_PROB
    if len(pred_list) >= PRED_MIN_FOR_REPRO and random.random() < pred_repro_p:
        candidates = [p for p in pred_list if p.energy > PRED_ENERGY_TO_REPRO]
        if len(candidates) >= 2:
            weights = [p.energy ** 1.6 for p in candidates]
            p1 = random.choices(candidates, weights=weights, k=1)[0]
            p2 = random.choices(candidates, weights=weights, k=1)[0]
            while p2 == p1:
                p2 = random.choices(candidates, weights=weights, k=1)[0]
            alpha = p1.energy / (p1.energy + p2.energy)
            child_nn = p1.nn.blend_with(p2.nn, PRED_MUTATION_RATE, alpha)
            child_pos = (p1.pos + p2.pos) / 2 + np.random.uniform(-110, 110, 2)
            child = Agent(child_pos[0], child_pos[1], "pred", child_nn)
            child.energy = 0.62
            child.vel *= 0.3
            pred_list.append(child)
            all_agents.append(child)
            p1.energy *= 0.65
            p2.energy *= 0.65
            p1.energy = max(0.22, p1.energy)
            p2.energy = max(0.22, p2.energy)

    # Prede
    prey_repro_p = 0.09 if len(prey_list) < 40 else 0.045 if len(prey_list) < 75 else PREY_REPRO_PROB
    if len(prey_list) >= PREY_MIN_FOR_REPRO and random.random() < prey_repro_p:
        candidates = [p for p in prey_list if p.energy > PREY_ENERGY_TO_REPRO]
        if len(candidates) >= 2:
            weights = [p.energy ** 1.6 for p in candidates]
            p1 = random.choices(candidates, weights=weights, k=1)[0]
            p2 = random.choices(candidates, weights=weights, k=1)[0]
            while p2 == p1:
                p2 = random.choices(candidates, weights=weights, k=1)[0]
            alpha = p1.energy / (p1.energy + p2.energy)
            child_nn = p1.nn.blend_with(p2.nn, PREY_MUTATION_RATE, alpha)
            child_pos = (p1.pos + p2.pos) / 2 + np.random.uniform(-70, 70, 2)
            child = Agent(child_pos[0], child_pos[1], "prey", child_nn)
            child.energy = 0.78
            child.vel *= 0.38
            prey_list.append(child)
            all_agents.append(child)
            p1.energy *= 0.65
            p2.energy *= 0.65
            p1.energy = max(0.22, p1.energy)
            p2.energy = max(0.22, p2.energy)

    # ─── Disegno ──────────────────────────────────────────────────────────────
    for f in foods:
        draw_food(screen, f)

    for agent in all_agents:
        draw_agent(screen, agent)

    # Info testo
    prey_c = len(prey_list)
    pred_c = len(pred_list)
    avg_prey = sum(p.energy for p in prey_list) / prey_c if prey_c else 0
    avg_pred = sum(p.energy for p in pred_list) / pred_c if pred_c else 0

    grass_c = sum(1 for f in foods if f.kind == "grass" and not f.eaten)
    meat_c  = sum(1 for f in foods if f.kind == "meat" and not f.eaten)

    elapsed_sec = (pygame.time.get_ticks() - start_time) // 1000
    m, s = divmod(elapsed_sec, 60)

    txt = font.render(
        f"Prey: {prey_c:3d} ({avg_prey:.2f})   Pred: {pred_c:2d} ({avg_pred:.2f})   "
        f"Grass: {grass_c:4d}   Meat: {meat_c:3d}   "
        f"Time: {m:02d}:{s:02d}   FPS: {clock.get_fps():.0f}",
        True, (220, 240, 220)
    )
    screen.blit(txt, (10, HEIGHT - 38))

    pygame.display.flip()

pygame.quit()