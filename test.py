import pygame
import numpy as np
import random

pygame.init()
WIDTH, HEIGHT = 1200, 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Enhanced Predator-Prey Boids-ish Simulation")
clock = pygame.time.Clock()

# ─── Parameters ───────────────────────────────────────────────────────────────
N_PREY_START      = 80
N_PRED_START      = 20
MAX_SPEED_PREY    = 3
MAX_SPEED_PRED    = 3.7
VISION_RANGE_PREY = 100
VISION_RANGE_PRED = 270
SEPARATION_DIST   = 25
EAT_DISTANCE      = 11

# ─── Agent Class ──────────────────────────────────────────────────────────────
class Agent:
    def __init__(self, x, y, type_="prey"):
        self.pos = np.array([x, y], dtype=float)
        self.vel = np.random.randn(2) * 0.8
        self.type = type_
        self.energy = 1.0 if type_ == "prey" else 0.65

    @property
    def speed(self):
        return np.linalg.norm(self.vel)

    def normalize_vel(self, max_speed):
        s = self.speed
        if s > max_speed:
            self.vel *= max_speed / s
        elif s < 0.3:
            self.vel += np.random.randn(2) * 0.35

# ─── Helpers ──────────────────────────────────────────────────────────────────
def get_visible_agents(agent, targets, vision_range):
    visible = []
    for other in targets:
        if other is agent:
            continue
        diff = other.pos - agent.pos
        # toroidal wrap
        diff -= np.round(diff / [WIDTH, HEIGHT]) * [WIDTH, HEIGHT]
        dist = np.linalg.norm(diff)
        if dist < vision_range:
            visible.append((other, dist, diff))
    return visible


def energy_color(energy):
    g = int(180 * energy)
    r = int(180 * (1 - energy))
    return (r, g, 40)


def draw_agent(screen, agent):
    pos = agent.pos.astype(int)
    if agent.speed < 0.2:
        color = energy_color(agent.energy) if agent.type == "prey" else (140, 20, 20)
        pygame.draw.circle(screen, color, pos, 5)
        return

    angle = np.arctan2(agent.vel[1], agent.vel[0])
    size = 7 if agent.type == "pred" else 4 + int(3 * agent.energy)
    pts = [
        pos + np.array([ np.cos(angle),               np.sin(angle)              ]) * size,
        pos + np.array([ np.cos(angle + 2.35), np.sin(angle + 2.35)]) * size * 0.55,
        pos + np.array([ np.cos(angle - 2.35), np.sin(angle - 2.35)]) * size * 0.55,
    ]
    color = energy_color(agent.energy) if agent.type == "prey" else \
            (220, 40, 40) if agent.energy > 0.5 else (140, 20, 20)
    pygame.draw.polygon(screen, color, pts)
    pygame.draw.polygon(screen, (0,0,0), pts, 1)


# ─── Initialization ───────────────────────────────────────────────────────────
prey_list = [Agent(random.uniform(0, WIDTH), random.uniform(0, HEIGHT), "prey")
             for _ in range(N_PREY_START)]
pred_list = [Agent(random.uniform(0, WIDTH), random.uniform(0, HEIGHT), "pred")
             for _ in range(N_PRED_START)]
all_agents = prey_list + pred_list

font = pygame.font.SysFont("consolas", 18)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            running = False

    dt = clock.tick(60) / 1000.0

    # Gentle fade instead of full clear
    fade = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    fade.fill((6, 21, 9, 14))          # low alpha → nice trails
    screen.blit(fade, (0, 0))

    # screen.fill((2, 5, 2))

    # ─── Update Predators ─────────────────────────────────────────────────────
    for pred in pred_list[:]:  # copy because we might remove
        visible = get_visible_agents(pred, prey_list, VISION_RANGE_PRED)

        if visible:
            closest_diff = min(visible, key=lambda x: x[1])[2]
            accel = closest_diff / (np.linalg.norm(closest_diff) + 1e-6) * 1.1
        else:
            accel = np.random.randn(2) * 0.14

        pred.vel += accel
        pred.normalize_vel(MAX_SPEED_PRED)

        # Energy drain
        pred.energy = max(0.08, pred.energy - 0.003 * dt * 60)

    # Remove dead predators
    dead_preds = [p for p in pred_list if p.energy <= 0.08]
    for dead in dead_preds:
        pred_list.remove(dead)
        all_agents.remove(dead)

    # ─── Update Prey ──────────────────────────────────────────────────────────
    for pr in prey_list:
        visible_preds = get_visible_agents(pr, pred_list, VISION_RANGE_PREY)
        visible_prey  = get_visible_agents(pr, prey_list, VISION_RANGE_PREY)

        accel = np.zeros(2)

        # Flee predators
        if visible_preds:
            for pred, dist, diff in visible_preds:
                panic = (VISION_RANGE_PREY - dist) / VISION_RANGE_PREY
                accel -= diff / (dist + 1e-5) * panic * 1.45

        # Separation
        for other, dist, diff in visible_prey:
            if dist < SEPARATION_DIST:
                accel -= diff / (dist + 1e-6) * 0.9

        # Weak cohesion
        if len(visible_prey) >= 3:
            avg_pos = np.mean([o.pos for o, _, _ in visible_prey], axis=0)
            to_center = avg_pos - pr.pos
            to_center -= np.round(to_center / [WIDTH, HEIGHT]) * [WIDTH, HEIGHT]
            accel += to_center * 0.0007 * len(visible_prey)

        # Wander
        accel += np.random.randn(2) * 0.085

        pr.vel += accel
        pr.normalize_vel(MAX_SPEED_PREY)

        # Gain energy (grazing)
        pr.energy = min(1.0, pr.energy + 0.016 * dt * 60)

    # ─── Movement + reflective boundaries (no more wrapping) ──────────────────
    for agent in all_agents:
        agent.pos += agent.vel * 60 * dt

        # Left / right bounce
        if agent.pos[0] < 0:
            agent.pos[0] = -agent.pos[0]
            agent.vel[0] = -agent.vel[0] * 0.92
        elif agent.pos[0] > WIDTH:
            agent.pos[0] = 2 * WIDTH - agent.pos[0]
            agent.vel[0] = -agent.vel[0] * 0.92

        # Top / bottom bounce
        if agent.pos[1] < 0:
            agent.pos[1] = -agent.pos[1]
            agent.vel[1] = -agent.vel[1] * 0.92
        elif agent.pos[1] > HEIGHT:
            agent.pos[1] = 2 * HEIGHT - agent.pos[1]
            agent.vel[1] = -agent.vel[1] * 0.92

    # ─── Predation ────────────────────────────────────────────────────────────
    to_remove = []
    for pred in pred_list:
        if pred.energy < 0.18:
            continue
        for pr in prey_list:
            diff = pr.pos - pred.pos
            dist = np.linalg.norm(diff)
            if dist < EAT_DISTANCE:
                to_remove.append(pr)
                pred.energy = min(1.35, pred.energy + 0.68)
                break  # one prey per predator per frame

    for dead in to_remove:
        if dead in prey_list:
            prey_list.remove(dead)
            all_agents.remove(dead)

    # ─── Respawn prey if too few ──────────────────────────────────────────────
    if len(prey_list) < 50:
        for _ in range(4):
            new_prey = Agent(
                random.uniform(0, WIDTH),
                random.uniform(0, HEIGHT),
                "prey"
            )
            prey_list.append(new_prey)
            all_agents.append(new_prey)

    # ─── Draw everything ──────────────────────────────────────────────────────
    for agent in all_agents:
        # # Range di vista predatori
        # if agent.type == "pred":
        #     pos = agent.pos.astype(int)
        #     vision_surf = pygame.Surface((int(VISION_RANGE_PRED*2), int(VISION_RANGE_PRED*2)), pygame.SRCALPHA)
        #     pygame.draw.circle(
        #         vision_surf,
        #         (180, 40, 40, 1),
        #         (VISION_RANGE_PRED, VISION_RANGE_PRED),
        #         VISION_RANGE_PRED
        #     )
        #     screen.blit(
        #         vision_surf,
        #         (pos[0] - VISION_RANGE_PRED, pos[1] - VISION_RANGE_PRED)
        #     )

        draw_agent(screen, agent)

    # HUD
    prey_count = len(prey_list)
    pred_count = len(pred_list)
    avg_energy = sum(p.energy for p in pred_list) / pred_count if pred_count > 0 else 0

    txt = font.render(
        f"Prey: {prey_count:3d}   Pred: {pred_count:2d}   Avg pred energy: {avg_energy:.2f}   FPS: {clock.get_fps():.0f}",
        True, (200, 220, 180)
    )
    screen.blit(txt, (10, HEIGHT - 30))

    pygame.display.flip()

pygame.quit()