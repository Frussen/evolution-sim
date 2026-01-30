import pygame
import numpy as np
import random

pygame.init()
WIDTH, HEIGHT = 1200, 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Neural Predator-Prey + Evolution – Toroidale Balanced")
clock = pygame.time.Clock()

# ─── Parametri ────────────────────────────────────────────────────────────────
N_PREY_START       = 140     # ← aumentato da 110
N_PRED_START       = 20
MAX_SPEED_PREY     = 3.0
MAX_SPEED_PRED     = 3.5
VISION_RANGE_PREY  = 150
VISION_RANGE_PRED  = 280
EAT_DISTANCE       = 12

PRED_REPRO_PROB       = 0.004
PRED_MIN_FOR_REPRO    = 3
PRED_ENERGY_TO_REPRO  = 0.90
PRED_MUTATION_RATE    = 0.05

PREY_REPRO_PROB       = 0.011
PREY_MIN_FOR_REPRO    = 15
PREY_ENERGY_TO_REPRO  = 0.76
PREY_MUTATION_RATE    = 0.015

NN_HIDDEN          = 128

# ─── Attivazioni ──────────────────────────────────────────────────────────────
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

    def mutated_copy(self, mutation_rate):
        child = SimpleNN(self.W1.shape[0], self.W1.shape[1], self.W2.shape[1])
        child.W1 = self.W1 + np.random.randn(*self.W1.shape) * mutation_rate
        child.b1 = self.b1 + np.random.randn(*self.b1.shape) * mutation_rate * 0.35
        child.W2 = self.W2 + np.random.randn(*self.W2.shape) * mutation_rate
        child.b2 = self.b2 + np.random.randn(*self.b2.shape) * mutation_rate * 0.35
        return child

# ─── Agent ────────────────────────────────────────────────────────────────────
class Agent:
    def __init__(self, x, y, type_="prey", nn=None):
        self.pos = np.array([x, y], dtype=float)
        self.vel = np.random.randn(2) * 0.6
        self.type = type_
        self.energy = 1.0 if type_ == "prey" else 0.85
        self.last_eat_time = 0
        if nn is None:
            self.nn = SimpleNN(70, NN_HIDDEN, 2)
        else:
            self.nn = nn

    @property
    def speed(self):
        return np.linalg.norm(self.vel)

    def normalize_vel(self, max_speed):
        s = self.speed
        if s > max_speed:
            self.vel *= max_speed / s
        elif s < 0.20:
            self.vel += np.random.randn(2) * 0.35

# ─── Input NN ─────────────────────────────────────────────────────────────────
def get_nn_input(agent, prey_list, pred_list):
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
        pred_features[i*5 : i*5+5] = info

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
        prey_features[i*5 : i*5+5] = info

    inputs = np.concatenate([
        pred_features,
        prey_features,
        [agent.energy * 2 - 1],
        agent.vel / MAX_SPEED_PRED,
        agent.pos / np.array([WIDTH, HEIGHT]) * 2 - 1,
        [np.cos(pygame.time.get_ticks() * 0.0008)]
    ])

    return inputs.astype(np.float32)[:70]

# ─── Colori ───────────────────────────────────────────────────────────────────
def energy_color(energy, is_prey=True):
    if is_prey:
        r = int(100 * (1 - energy))
        g = int(140 + 100 * energy)
        return (r, g, 55)
    else:
        if energy > 0.8:  return (255, 100, 80)
        if energy > 0.45: return (240, 60, 60)
        return (150, 30, 30)

def draw_agent(screen, agent):
    pos = agent.pos.astype(int)
    if agent.speed < 0.22:
        color = energy_color(agent.energy, agent.type == "prey")
        pygame.draw.circle(screen, color, pos, 6)
        return

    angle = np.arctan2(agent.vel[1], agent.vel[0])
    size = 9.5 if agent.type == "pred" else 5 + 4 * agent.energy
    pts = [
        pos + np.array([np.cos(angle),      np.sin(angle)])      * size,
        pos + np.array([np.cos(angle+2.4),  np.sin(angle+2.4)])  * size * 0.6,
        pos + np.array([np.cos(angle-2.4),  np.sin(angle-2.4)])  * size * 0.6,
    ]
    color = energy_color(agent.energy, agent.type == "prey")
    pygame.draw.polygon(screen, color, pts)
    pygame.draw.polygon(screen, (0,0,0), pts, 1)

# ─── Inizializzazione ─────────────────────────────────────────────────────────
prey_list = [Agent(random.uniform(0,WIDTH), random.uniform(0,HEIGHT), "prey")
             for _ in range(N_PREY_START)]
pred_list = [Agent(random.uniform(0,WIDTH), random.uniform(0,HEIGHT), "pred")
             for _ in range(N_PRED_START)]
all_agents = prey_list + pred_list

font = pygame.font.SysFont("consolas", 18)

start_time = pygame.time.get_ticks()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            running = False

    dt = clock.tick(60) / 1000.0
    current_time = pygame.time.get_ticks()

    # Fade
    fade = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    fade.fill((6, 21, 9, 14))
    screen.blit(fade, (0, 0))

    # ─── Update Predatori ─────────────────────────────────────────────────────
    for pred in pred_list[:]:
        inputs = get_nn_input(pred, prey_list, pred_list)
        accel = pred.nn.forward(inputs) * 1.5
        
        # Bonus iniziale per bootstrap (primi 25 secondi)
        if current_time - start_time < 25000:
            accel *= 1.35
            pred.energy = min(1.0, pred.energy + 0.0006 * dt * 60)

        pred.vel += accel
        pred.normalize_vel(MAX_SPEED_PRED)

        drain = 0.0006 * dt * 60   # ← ridotto

        if current_time - pred.last_eat_time < 15000:
            drain *= 0.4

        pred.energy = max(0.04, pred.energy - drain)

        if pred.speed > 2.5:
            pred.energy -= 0.00025 * pred.speed**2 * dt * 60

    dead_preds = [p for p in pred_list if p.energy <= 0.04]
    for p in dead_preds:
        pred_list.remove(p)
        all_agents.remove(p)

    # ─── Update Prede ─────────────────────────────────────────────────────────
    for pr in prey_list:
        inputs = get_nn_input(pr, prey_list, pred_list)
        accel = pr.nn.forward(inputs) * 1.12
        pr.vel += accel
        pr.normalize_vel(MAX_SPEED_PREY)
        pr.energy = min(1.0, pr.energy + 0.019 * dt * 60)

        if pr.speed > 2.5:
            pr.energy -= 0.00008 * pr.speed**2 * dt * 60   # ← ridotto

    # ─── Movimento toroidale ──────────────────────────────────────────────────
    for a in all_agents:
        a.pos += a.vel * 60 * dt
        a.pos[0] %= WIDTH
        a.pos[1] %= HEIGHT

    # ─── Predazione ───────────────────────────────────────────────────────────
    to_remove = []
    for pred in pred_list:
        if pred.energy < 0.13: continue
        for pr in prey_list:
            diff = pr.pos - pred.pos
            dist = np.linalg.norm(diff)
            if dist < EAT_DISTANCE:
                to_remove.append(pr)
                pred.energy = min(1.80, pred.energy + 1.10)
                pred.last_eat_time = current_time
                break

    for dead in set(to_remove):
        if dead in prey_list:
            prey_list.remove(dead)
            all_agents.remove(dead)

    # ─── Riproduzione ─────────────────────────────────────────────────────────
    pred_repro_p = PRED_REPRO_PROB
    if len(pred_list) < 6:   pred_repro_p = 0.08
    elif len(pred_list) < 12: pred_repro_p = 0.04
    else: pred_repro_p = 0.004

    if len(pred_list) >= PRED_MIN_FOR_REPRO and random.random() < pred_repro_p:
        candidates = [p for p in pred_list if p.energy > PRED_ENERGY_TO_REPRO]
        if len(candidates) >= 2:
            weights = [p.energy ** 1.5 for p in candidates]
            p1 = random.choices(candidates, weights=weights, k=1)[0]
            p2 = random.choices(candidates, weights=weights, k=1)[0]
            donor = p1 if p1.energy > p2.energy else p2
            child_nn = donor.nn.mutated_copy(PRED_MUTATION_RATE)
            child = Agent(
                donor.pos[0] + random.uniform(-100,100),
                donor.pos[1] + random.uniform(-100,100),
                "pred", nn=child_nn
            )
            child.energy = 0.65
            child.vel *= 0.35
            pred_list.append(child)
            all_agents.append(child)

    prey_repro_p = PREY_REPRO_PROB
    if len(prey_list) < 40:   prey_repro_p = 0.08
    elif len(prey_list) < 70: prey_repro_p = 0.04
    else: prey_repro_p = 0.011

    if len(prey_list) >= 8 and random.random() < prey_repro_p:
        candidates = [p for p in prey_list if p.energy > 0.60]
        if len(candidates) >= 1:
            donor = random.choice(candidates)
            child_nn = donor.nn.mutated_copy(PREY_MUTATION_RATE)
            child = Agent(
                donor.pos[0] + random.uniform(-60,60),
                donor.pos[1] + random.uniform(-60,60),
                "prey", nn=child_nn
            )
            child.energy = 0.80
            child.vel *= 0.4
            prey_list.append(child)
            all_agents.append(child)

    # ─── Disegno ──────────────────────────────────────────────────────────────
    for agent in all_agents:
        draw_agent(screen, agent)

    prey_c = len(prey_list)
    pred_c = len(pred_list)
    avg_pred_e = sum(p.energy for p in pred_list) / pred_c if pred_c > 0 else 0
    avg_prey_e = sum(p.energy for p in prey_list) / prey_c if prey_c > 0 else 0

    elapsed_ms = pygame.time.get_ticks() - start_time
    elapsed_sec = elapsed_ms // 1000
    minutes = elapsed_sec // 60
    seconds = elapsed_sec % 60
    timestamp = f"{minutes:02d}:{seconds:02d}"

    txt = font.render(
        f"Prey: {prey_c:3d} ({avg_prey_e:.2f})   Pred: {pred_c:2d} ({avg_pred_e:.2f})   Time: {timestamp}   FPS: {clock.get_fps():.0f}",
        True, (230, 245, 210)
    )
    screen.blit(txt, (10, HEIGHT - 35))

    pygame.display.flip()

pygame.quit()