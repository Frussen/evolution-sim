import pygame
import math
import random
import copy

# ================= CONFIGURAZIONE ECO-SISTEMA =================
WIDTH, HEIGHT = 1000, 700
FPS = 60
MARGIN = 15

SECTORS = 5
INPUTS_PER_SECTOR = 4           # 1 distanza + 3 tipi
NUM_WALL_SENSORS = 4
NUM_INPUTS = SECTORS * INPUTS_PER_SECTOR + NUM_WALL_SENSORS

START_PLANTS      = 200
START_PREY        = 170
START_PREDATORS   = 30

MAX_ENERGY        = 230
CHILD_ENERGY_VAL  = 90

PREY_REPRO_AT     = 170
PREY_BASAL_COST   = 0.3
MAX_SPEED_PREY    = 2.7
PREY_EAT_GAIN     = 40  

PRED_REPRO_AT     = 210
PRED_BASAL_COST   = 0.7
MAX_SPEED_PRED    = 2.5  
PRED_EAT_GAIN     = 70    

MOVE_COST_FACTOR  = 0.025 
MAX_TURN          = math.pi / 15
MAX_VISION        = 160
NEW_PLANTS_PER_FRAME = 2
PLANT_SPAWN_CHANCE = 0.8

# ================= VISUALIZZAZIONE =================
show_vision    = False           # inizia con OFF
show_prey_net  = False
show_pred_net  = False

highlight_prey = None
highlight_pred = None

# Colori
COLOR_BG          = (15, 15, 22)
COLOR_PLANT       = (60, 220, 80)
COLOR_PREY        = (70, 160, 255)
COLOR_PRED        = (255, 70, 70)
COLOR_WALL        = (50, 50, 65)

COLOR_VISION_PREY = (100, 180, 255, 40)
COLOR_VISION_PRED = (255, 100, 100, 50)

# ================= PULSANTI RETE =================
PREY_NET_BUTTON = pygame.Rect(WIDTH - 170, 27, 140, 32)
PRED_NET_BUTTON = pygame.Rect(WIDTH - 170, 67, 140, 32)

WEIGHT_THRESHOLD = 0.20
MAX_EDGES = 90

# ================= CLASSI =================

class NeuralNet:
    def __init__(self):
        def rw(): return random.gauss(0, 1.0)
        self.w_ih = [[rw() for _ in range(NUM_INPUTS)] for _ in range(14)]
        self.b_h   = [rw() * 0.1 for _ in range(14)]
        self.w_ho  = [[rw() for _ in range(14)] for _ in range(2)]
        self.b_o   = [rw() * 0.1 for _ in range(2)]

    def forward(self, inputs):
        h = [math.tanh(self.b_h[j] + sum(inputs[i] * self.w_ih[j][i] for i in range(NUM_INPUTS))) for j in range(14)]
        out = [math.tanh(self.b_o[j] + sum(h[i] * self.w_ho[j][i] for i in range(14))) for j in range(2)]
        return out

    def mutate(self):
        for layer in (self.w_ih, self.w_ho):
            for row in layer:
                for i in range(len(row)):
                    if random.random() < 0.2: 
                        row[i] += random.gauss(0, 0.3)


class Creature:
    def __init__(self, x, y, is_prey, brain=None):
        self.x = max(MARGIN, min(WIDTH - MARGIN, x))
        self.y = max(MARGIN, min(HEIGHT - MARGIN, y))
        self.heading = random.uniform(0, math.pi*2)
        self.speed = 1.2
        self.energy = 100
        self.health = 100
        self.is_prey = is_prey
        self.brain = brain if brain else NeuralNet()
        self.radius = 6.0 if is_prey else 9.5

    def get_inputs(self, plants, creatures):
        sectors = SECTORS
        sector_dist = [1.0] * sectors
        sector_type = [0] * (sectors * 3)
        fov = math.pi * 1.6 if self.is_prey else math.pi * 1.2
        
        for ent in (plants + creatures):
            if ent is self: continue
            dx, dy = ent.x - self.x, ent.y - self.y
            dist = math.hypot(dx, dy)
            if dist > MAX_VISION: continue

            angle = (math.atan2(dy, dx) - self.heading + math.pi) % (2*math.pi) - math.pi
            if abs(angle) > fov/2: continue

            s = int((angle + fov/2) / fov * sectors)
            s = max(0, min(sectors-1, s))
            
            d_norm = dist / MAX_VISION
            if d_norm < sector_dist[s]:
                sector_dist[s] = d_norm
                for i in range(3): sector_type[s*3 + i] = 0
                if hasattr(ent, 'is_prey'): 
                    if ent.is_prey: 
                        sector_type[s*3 + 1] = 1     # preda
                    else:           
                        sector_type[s*3 + 2] = 1     # predatore
                else:               
                    sector_type[s*3 + 0] = 1         # pianta
        
        d_top    = min(1.0, (self.y - MARGIN) / MAX_VISION)
        d_bottom = min(1.0, (HEIGHT - MARGIN - self.y) / MAX_VISION)
        d_left   = min(1.0, (self.x - MARGIN) / MAX_VISION)
        d_right  = min(1.0, (WIDTH - MARGIN - self.x) / MAX_VISION)
        
        res = []
        for i in range(sectors):
            res.append(sector_dist[i])
            res.extend(sector_type[i*3 : i*3+3])
        res.extend([d_top, d_bottom, d_left, d_right])
        return res, sector_dist, sector_type, fov

    def update(self):
        cost = PREY_BASAL_COST if self.is_prey else PRED_BASAL_COST
        self.energy -= cost + (self.speed * MOVE_COST_FACTOR)
        if self.energy <= 0:
            self.health -= 1.5
            self.energy = 0
        return self.health > 0


class Plant:
    def __init__(self):
        self.x = random.uniform(MARGIN + 5, WIDTH - MARGIN - 5)
        self.y = random.uniform(MARGIN + 5, HEIGHT - MARGIN - 5)
        self.radius = 4


# ================= SETUP =================

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
font = pygame.font.SysFont("Consolas", 14, bold=True)
font_small = pygame.font.SysFont("Consolas", 12)

plants = [Plant() for _ in range(START_PLANTS)]
creatures = [Creature(random.random()*WIDTH, random.random()*HEIGHT, True) for _ in range(START_PREY)]
creatures += [Creature(random.random()*WIDTH, random.random()*HEIGHT, False) for _ in range(START_PREDATORS)]

running = True
start_ticks = pygame.time.get_ticks()

# Pulsante toggle Vision
BUTTON_RECT = pygame.Rect(27, 27, 127, 37)
BUTTON_COLOR_ON  = (140, 140, 140)
BUTTON_COLOR_OFF = (60, 60, 60)

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if BUTTON_RECT.collidepoint(event.pos):
                show_vision = not show_vision
                if not show_vision:
                    show_prey_net = False
                    show_pred_net = False
            if show_vision:
                if PREY_NET_BUTTON.collidepoint(event.pos):
                    show_prey_net = not show_prey_net
                    if show_prey_net:
                        show_pred_net = False
                if PRED_NET_BUTTON.collidepoint(event.pos):
                    show_pred_net = not show_pred_net
                    if show_pred_net:
                        show_prey_net = False

    screen.fill(COLOR_BG)
    pygame.draw.rect(screen, COLOR_WALL, (MARGIN, MARGIN, WIDTH-2*MARGIN, HEIGHT-2*MARGIN), 2)

    # Generazione piante
    for _ in range(NEW_PLANTS_PER_FRAME):
        if random.random() < PLANT_SPAWN_CHANCE:
            plants.append(Plant())

    for p in plants:
        pygame.draw.circle(screen, COLOR_PLANT, (int(p.x), int(p.y)), p.radius)

    new_gen = []
    n_prey, n_pred = 0, 0
    vision_data = {}

    for c in creatures[:]:
        if c.is_prey: n_prey += 1
        else: n_pred += 1

        inputs, sector_dist, sector_type, fov = c.get_inputs(plants, creatures)
        vision_data[c] = (sector_dist, sector_type, fov)

        out = c.brain.forward(inputs)
        
        c.heading += out[0] * MAX_TURN
        max_s = MAX_SPEED_PREY if c.is_prey else MAX_SPEED_PRED
        c.speed = max(0.4, min(max_s, c.speed + out[1] * 0.2))
        
        nx = c.x + math.cos(c.heading) * c.speed
        ny = c.y + math.sin(c.heading) * c.speed

        if nx < MARGIN or nx > WIDTH - MARGIN:
            nx = max(MARGIN, min(WIDTH - MARGIN, nx))
            c.speed *= 0.1
        if ny < MARGIN or ny > HEIGHT - MARGIN:
            ny = max(MARGIN, min(HEIGHT - MARGIN, ny))
            c.speed *= 0.1
        
        c.x, c.y = nx, ny

        if c.is_prey:
            for p in plants[:]:
                if math.hypot(c.x-p.x, c.y-p.y) < c.radius + p.radius:
                    c.energy = min(MAX_ENERGY, c.energy + PREY_EAT_GAIN)
                    plants.remove(p)
                    break
        else:
            for other in creatures:
                if other.is_prey and other.health > 0:
                    if math.hypot(c.x-other.x, c.y-other.y) < c.radius + other.radius:
                        c.energy = min(MAX_ENERGY, c.energy + PRED_EAT_GAIN)
                        other.health = -100 
                        break

        if not c.update():
            creatures.remove(c)
            if c in vision_data:
                del vision_data[c]
        else:
            limit = PREY_REPRO_AT if c.is_prey else PRED_REPRO_AT
            if c.energy > limit:
                c.energy -= CHILD_ENERGY_VAL
                off_brain = copy.deepcopy(c.brain)
                off_brain.mutate()
                new_gen.append(Creature(c.x, c.y, c.is_prey, off_brain))

            color = COLOR_PREY if c.is_prey else COLOR_PRED
            pygame.draw.circle(screen, color, (int(c.x), int(c.y)), int(c.radius))
            pygame.draw.line(screen, (100, 100, 100), (c.x, c.y), 
                            (c.x + math.cos(c.heading)*12, c.y + math.sin(c.heading)*12), 2)

    creatures.extend(new_gen)

    # Gestione creature evidenziate
    if highlight_prey and (highlight_prey not in creatures or highlight_prey.health <= 0):
        highlight_prey = None

    if highlight_pred and (highlight_pred not in creatures or highlight_pred.health <= 0):
        highlight_pred = None

    prey_list = [c for c in creatures if c.is_prey and c.health > 0]
    pred_list = [c for c in creatures if not c.is_prey and c.health > 0]

    if highlight_prey is None and prey_list:
        highlight_prey = random.choice(prey_list)

    if highlight_pred is None and pred_list:
        highlight_pred = random.choice(pred_list)

    # =============================================
    #    DISEGNO VISION SOLO SE ATTIVA
    # =============================================
    if show_vision:
        def draw_vision(creature, alpha_color):
            if not creature or creature not in vision_data:
                return
                
            sector_dist, sector_type, fov = vision_data[creature]
            sector_angle = fov / SECTORS
            start_angle = creature.heading - fov/2

            s = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            points = [(creature.x, creature.y)]
            for i in range(33):
                ang = start_angle + (fov * i / 32)
                px = creature.x + math.cos(ang) * MAX_VISION
                py = creature.y + math.sin(ang) * MAX_VISION
                points.append((px, py))
            pygame.draw.polygon(s, alpha_color, points)
            screen.blit(s, (0, 0))

            for s_idx in range(SECTORS):
                ang = start_angle + sector_angle * (s_idx + 0.5)
                dist = sector_dist[s_idx] * MAX_VISION

                if dist >= MAX_VISION * 0.99:
                    continue

                color = (140,140,140)
                for t in range(3):
                    if sector_type[s_idx*3 + t]:
                        if t == 0:   color = (100, 255, 140)
                        elif t == 1: color = (120, 200, 255)
                        elif t == 2: color = (255, 120, 120)
                        break

                ex = creature.x + math.cos(ang) * dist
                ey = creature.y + math.sin(ang) * dist

                pygame.draw.line(screen, (*color, 140), 
                               (creature.x, creature.y), (ex, ey), 1)
                pygame.draw.circle(screen, color, (int(ex), int(ey)), 5, 2)

        if highlight_prey:
            draw_vision(highlight_prey, COLOR_VISION_PREY)
            pygame.draw.circle(screen, (220,220,80), (int(highlight_prey.x), int(highlight_prey.y)), 11, 2)

        if highlight_pred:
            draw_vision(highlight_pred, COLOR_VISION_PRED)
            pygame.draw.circle(screen, (255,180,80), (int(highlight_pred.x), int(highlight_pred.y)), 13, 2)

    # =============================================
    #    PULSANTI RETE NEURALE
    # =============================================
    if show_vision:
        pygame.draw.rect(screen, (100, 140, 220), PREY_NET_BUTTON, border_radius=6)
        pygame.draw.rect(screen, (220,220,220), PREY_NET_BUTTON, 2, border_radius=6)
        text_prey = font_small.render("Prey Net", True, (255,255,255))
        screen.blit(text_prey, text_prey.get_rect(center=PREY_NET_BUTTON.center))

        pygame.draw.rect(screen, (220, 100, 100), PRED_NET_BUTTON, border_radius=6)
        pygame.draw.rect(screen, (220,220,220), PRED_NET_BUTTON, 2, border_radius=6)
        text_pred = font_small.render("Pred Net", True, (255,255,255))
        screen.blit(text_pred, text_pred.get_rect(center=PRED_NET_BUTTON.center))

    # =============================================
    #    GRAFO TOP COLLEGAMENTI
    # =============================================
    if show_vision and (show_prey_net or show_pred_net):
        creature = highlight_prey if show_prey_net else highlight_pred
        title = "Prey Neural Network" if show_prey_net else "Pred Neural Network"
        color_title = (140, 200, 255) if show_prey_net else (255, 140, 140)

        if creature:
            brain = creature.brain

            edges = []

            # Input → Hidden
            for i in range(NUM_INPUTS):
                for h in range(14):
                    w = brain.w_ih[h][i]
                    edges.append((("I"+str(i)), ("H"+str(h)), w))

            # Hidden → Output
            for h in range(14):
                for o in range(2):
                    w = brain.w_ho[o][h]
                    edges.append((("H"+str(h)), ("O"+str(o)), w))

            # Top collegamenti per |peso|
            edges.sort(key=lambda x: abs(x[2]), reverse=True)
            top_edges = [e for e in edges if abs(e[2]) >= WEIGHT_THRESHOLD][:MAX_EDGES]

            # Posizioni nodi (già alzate)
            col_x = [220, 520, 820]
            node_y = {}
            y_spacing = 27

            for i in range(NUM_INPUTS):
                node_y["I"+str(i)] = 30 + i * y_spacing
            for h in range(14):
                node_y["H"+str(h)] = 130 + h * y_spacing
            node_y["O0"] = 250
            node_y["O1"] = 330

            # Titolo
            title_surf = font.render(title, True, color_title)
            screen.blit(title_surf, (WIDTH//2 - title_surf.get_width()//2, 40))

            # Nodi
            for name, y_pos in node_y.items():
                x_pos = col_x[0] if name.startswith("I") else col_x[1] if name.startswith("H") else col_x[2]
                pygame.draw.circle(screen, (160,160,160), (x_pos, y_pos), 7)

            # Collegamenti – versione sottile e proporzionale al peso
            for src, dst, weight in top_edges:
                if src not in node_y or dst not in node_y:
                    continue
                x1 = col_x[0] if src.startswith("I") else col_x[1] if src.startswith("H") else col_x[2]
                y1 = node_y[src]
                x2 = col_x[0] if dst.startswith("I") else col_x[1] if dst.startswith("H") else col_x[2]
                y2 = node_y[dst]

                abs_w = abs(weight)
                # Spessore sottile e proporzionale: min 1 px, max ~4 px
                thickness = max(1, min(4, int(1 + 3.5 * abs_w)))
                
                line_color = (220, 60, 60) if weight > 0 else (60, 60, 220)

                pygame.draw.line(screen, line_color, (x1+7, y1), (x2-7, y2), thickness)

            # Legenda
            leg_y = HEIGHT - 50
            pygame.draw.line(screen, (220,60,60), (WIDTH//2 - 120, leg_y), (WIDTH//2 - 80, leg_y), 4)
            screen.blit(font_small.render("Positive", True, (220,220,220)), (WIDTH//2 - 70, leg_y - 12))
            pygame.draw.line(screen, (60,60,220), (WIDTH//2 + 20, leg_y), (WIDTH//2 + 60, leg_y), 4)
            screen.blit(font_small.render("Negative", True, (220,220,220)), (WIDTH//2 + 70, leg_y - 12))

    # =============================================
    #    PULSANTE TOGGLE VISION
    # =============================================
    pygame.draw.rect(screen, BUTTON_COLOR_ON if show_vision else BUTTON_COLOR_OFF, BUTTON_RECT, border_radius=6)
    pygame.draw.rect(screen, (220,220,220), BUTTON_RECT, 2, border_radius=6)
    
    btn_text = font_small.render(f"Vision: {'ON' if show_vision else 'OFF'}", True, (255,255,255))
    text_rect = btn_text.get_rect(center=BUTTON_RECT.center)
    screen.blit(btn_text, text_rect)

    # UI info
    seconds = (pygame.time.get_ticks() - start_ticks) // 1000
    timer = f"{seconds // 60:02}:{seconds % 60:02}"
    status = f"TIME: {timer} | PREY: {n_prey} | PRED: {n_pred} | PLANTS: {len(plants)}"
    screen.blit(font.render(status, True, (200, 200, 200)), (27, HEIGHT - 42))
    
    if n_prey == 0 or n_pred == 0:
        screen.blit(font.render("EXTINCTION EVENT", True, (255, 50, 50)), (WIDTH//2 - 80, HEIGHT//2 - 20))

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()