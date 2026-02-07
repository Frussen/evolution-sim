import pygame
import math
import random
import copy

# ================= CONFIGURAZIONE ECO-SISTEMA =================
WIDTH, HEIGHT = 1920, 1080
FPS = 60
MARGIN = 15

SECTORS = 5
INPUTS_PER_SECTOR = 4           # 1 distanza + 3 tipi
NUM_WALL_SENSORS = 4
NUM_INPUTS = 13  # zone_x, zone_y, zone_r, inside_flag, self_x, self_y, dist_perim, dx_to_center, dy_to_center, d_top, d_bottom, d_left, d_right

START_PLANTS      = 0
START_PREY        = 400
START_PREDATORS   = 0

MAX_ENERGY        = 420
CHILD_ENERGY_VAL  = 130

# Costo energetico aggiuntivo per la riproduzione (tutti)
REPRO_ENERGY_COST = 100

PREY_REPRO_AT     = 400
PREY_BASAL_COST   = 0.1
MAX_SPEED_PREY    = 7.0
PREY_EAT_GAIN     = 40  

# predator-related constants removed

MOVE_COST_FACTOR  = 0.025 
MAX_TURN          = math.pi / 15
MAX_VISION        = 160
NEW_PLANTS_PER_FRAME = 4
PLANT_SPAWN_CHANCE = 0.08
# Probabilità che una nuova pianta spawni dentro la zona protetta
ZONE_PLANT_BIAS = 0.8

# Quando una preda è entro questa soglia (px) dal perimetro (usata per sensori)
PERIMETER_THRESHOLD = 12

# Movimento automatico lento della zona (pixel per frame)
ZONE_AUTO_MOVE_SPEED = 0.6
# Massima variazione angolare per frame (random walk direzione)
ZONE_AUTO_MOVE_ANG_DELTA = 0.05

# ================= VISUALIZZAZIONE =================
show_vision    = False           # inizia con OFF
show_net       = False

highlight_prey = None

# Colori
COLOR_BG          = (15, 15, 22)
COLOR_PLANT       = (60, 220, 80)
COLOR_PREY        = (70, 160, 255)
COLOR_WALL        = (50, 50, 65)

COLOR_VISION_PREY = (100, 180, 255, 40)
COLOR_VISION_PREY = (100, 180, 255, 40)

# ================= PULSANTI RETE =================
NET_BUTTON = pygame.Rect(WIDTH - 170, 47, 140, 32)
# Pulsante scelta casuale preda
RANDOM_PREY_BUTTON = pygame.Rect(WIDTH - 170, 7, 140, 28)

# Zona protetta (cerchio: x,y,r)
NO_PRED_ZONE = {
    'x': WIDTH // 2,
    'y': HEIGHT // 2,
    'r': 120,
}

# Interazione zona (drag/resize)
ZONE_HANDLE_WIDTH = 8
zone_dragging = False
zone_resizing = False
zone_drag_offset = (0, 0)
_zone_init_angle = random.random() * 2 * math.pi
zone_vx = math.cos(_zone_init_angle) * ZONE_AUTO_MOVE_SPEED
zone_vy = math.sin(_zone_init_angle) * ZONE_AUTO_MOVE_SPEED

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
        self.birth_time = pygame.time.get_ticks()
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
        
        # Rimuoviamo gli input dei settori dalla rete: manteniamo però
        # sector_dist e sector_type per la sola visualizzazione.
        # Aggiungi input: frazione di prede dentro il NO_PRED_ZONE e frazione fuori
        total_prey = 0
        inside_prey = 0
        for cr in creatures:
            if hasattr(cr, 'is_prey') and cr.is_prey and cr.health > 0:
                total_prey += 1
                if math.hypot(cr.x - NO_PRED_ZONE['x'], cr.y - NO_PRED_ZONE['y']) < NO_PRED_ZONE['r']:
                    inside_prey += 1

        if total_prey > 0:
            inside_frac = inside_prey / total_prey
        else:
            inside_frac = 0.0
        outside_frac = 1.0 - inside_frac

        # Informazioni sulla NO_PRED_ZONE normalizzate (0..1)
        zone_x = NO_PRED_ZONE['x'] / WIDTH
        zone_y = NO_PRED_ZONE['y'] / HEIGHT
        zone_r = NO_PRED_ZONE['r'] / max(WIDTH, HEIGHT)

        # Input aggiuntivo: flag se la creatura (self) è dentro la zona (1.0) o fuori (0.0)
        inside_flag = 1.0 if math.hypot(self.x - NO_PRED_ZONE['x'], self.y - NO_PRED_ZONE['y']) < NO_PRED_ZONE['r'] else 0.0

        # Rete riceve posizione/dimensione della zona, flag inside/outside,
        # le coordinate normalizzate della creatura stessa e la distanza dal perimetro
        self_x = self.x / WIDTH
        self_y = self.y / HEIGHT
        d_center = math.hypot(self.x - NO_PRED_ZONE['x'], self.y - NO_PRED_ZONE['y'])
        dist_perim = abs(d_center - NO_PRED_ZONE['r']) / max(WIDTH, HEIGHT)
        # Direzione normalizzata verso il centro (dx, dy)
        dx_to_center = (NO_PRED_ZONE['x'] - self.x) / WIDTH
        dy_to_center = (NO_PRED_ZONE['y'] - self.y) / HEIGHT
        res = [zone_x, zone_y, zone_r, inside_flag, self_x, self_y, dist_perim, dx_to_center, dy_to_center, d_top, d_bottom, d_left, d_right]
        return res, sector_dist, sector_type, fov

    def update(self):
        cost = PREY_BASAL_COST
        self.energy -= cost + (self.speed * MOVE_COST_FACTOR)
        if self.energy <= 0:
            self.health -= 1.5
            self.energy = 0
        return self.health > 0


class Plant:
    def __init__(self):
        # Posiziona la pianta casualmente all'interno del cerchio NO_PRED_ZONE
        ang = random.random() * 2 * math.pi
        # r distribuito uniformemente nella superficie del cerchio
        rad = NO_PRED_ZONE['r'] * math.sqrt(random.random()) * 0.95
        self.x = NO_PRED_ZONE['x'] + math.cos(ang) * rad
        self.y = NO_PRED_ZONE['y'] + math.sin(ang) * rad
        # clamp per sicurezza dentro i margini
        self.x = max(MARGIN + 5, min(WIDTH - MARGIN - 5, self.x))
        self.y = max(MARGIN + 5, min(HEIGHT - MARGIN - 5, self.y))
        self.radius = 4


# ================= SETUP =================

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
font = pygame.font.SysFont("Consolas", 14, bold=True)
font_small = pygame.font.SysFont("Consolas", 12)

plants = [Plant() for _ in range(START_PLANTS)]
creatures = [Creature(random.random()*WIDTH, random.random()*HEIGHT, True) for _ in range(START_PREY)]

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
            mx, my = event.pos
            # Toggle vision button
            if BUTTON_RECT.collidepoint((mx, my)):
                show_vision = not show_vision
                if not show_vision:
                    show_net = False
                continue

            # Pulsanti scelta casuale
            if RANDOM_PREY_BUTTON.collidepoint((mx, my)):
                prey_candidates = [c for c in creatures if c.is_prey and c.health > 0]
                if prey_candidates:
                    highlight_prey = random.choice(prey_candidates)
                    show_net = False
                continue

            # NET button (se vision attiva)
            if show_vision and NET_BUTTON.collidepoint((mx, my)):
                show_net = not show_net
                continue

            # Zona protetta: click on circumference => start resize; click inside => start drag
            dxz = mx - NO_PRED_ZONE['x']
            dyz = my - NO_PRED_ZONE['y']
            dz = math.hypot(dxz, dyz)
            if abs(dz - NO_PRED_ZONE['r']) <= ZONE_HANDLE_WIDTH:
                zone_resizing = True
                # store offset not necessary for resizing
                continue
            if dz < NO_PRED_ZONE['r'] - ZONE_HANDLE_WIDTH:
                zone_dragging = True
                zone_drag_offset = (NO_PRED_ZONE['x'] - mx, NO_PRED_ZONE['y'] - my)
                continue

            # Selezione creatura tramite click (sia prede che predatori)
            closest = None
            closest_d = 1e9
            for c in creatures:
                if c.health <= 0:
                    continue
                d = math.hypot(c.x - mx, c.y - my)
                if d <= c.radius + 4 and d < closest_d:
                    closest = c
                    closest_d = d

            if closest:
                highlight_prey = closest
        elif event.type == pygame.MOUSEMOTION:
            mx, my = event.pos
            if zone_dragging:
                NO_PRED_ZONE['x'] = int(mx + zone_drag_offset[0])
                NO_PRED_ZONE['y'] = int(my + zone_drag_offset[1])
                # clamp within margins
                NO_PRED_ZONE['x'] = max(MARGIN + NO_PRED_ZONE['r'], min(WIDTH - MARGIN - NO_PRED_ZONE['r'], NO_PRED_ZONE['x']))
                NO_PRED_ZONE['y'] = max(MARGIN + NO_PRED_ZONE['r'], min(HEIGHT - MARGIN - NO_PRED_ZONE['r'], NO_PRED_ZONE['y']))
            if zone_resizing:
                cx, cy = NO_PRED_ZONE['x'], NO_PRED_ZONE['y']
                newr = int(max(16, math.hypot(mx - cx, my - cy)))
                # cap radius to screen
                newr = min(newr, min(WIDTH, HEIGHT) // 2)
                NO_PRED_ZONE['r'] = newr
        elif event.type == pygame.MOUSEBUTTONUP:
            zone_dragging = False
            zone_resizing = False

    # Movimento automatico lento della zona quando non viene draggato/resized
    if not zone_dragging and not zone_resizing:
        # piccola variazione angolare alla velocità corrente
        ang = math.atan2(zone_vy, zone_vx)
        ang += random.uniform(-ZONE_AUTO_MOVE_ANG_DELTA, ZONE_AUTO_MOVE_ANG_DELTA)
        zone_vx = math.cos(ang) * ZONE_AUTO_MOVE_SPEED
        zone_vy = math.sin(ang) * ZONE_AUTO_MOVE_SPEED
        # aggiorna posizione
        NO_PRED_ZONE['x'] += zone_vx
        NO_PRED_ZONE['y'] += zone_vy
        # rimbalzo sui bordi: se tocca il muro inverti la componente di velocità
        left = MARGIN + NO_PRED_ZONE['r']
        right = WIDTH - MARGIN - NO_PRED_ZONE['r']
        top = MARGIN + NO_PRED_ZONE['r']
        bottom = HEIGHT - MARGIN - NO_PRED_ZONE['r']
        if NO_PRED_ZONE['x'] < left:
            NO_PRED_ZONE['x'] = left
            zone_vx = -zone_vx
        elif NO_PRED_ZONE['x'] > right:
            NO_PRED_ZONE['x'] = right
            zone_vx = -zone_vx
        if NO_PRED_ZONE['y'] < top:
            NO_PRED_ZONE['y'] = top
            zone_vy = -zone_vy
        elif NO_PRED_ZONE['y'] > bottom:
            NO_PRED_ZONE['y'] = bottom
            zone_vy = -zone_vy

    screen.fill(COLOR_BG)
    pygame.draw.rect(screen, COLOR_WALL, (MARGIN, MARGIN, WIDTH-2*MARGIN, HEIGHT-2*MARGIN), 2)

    # Generazione piante (ORA: solo dentro il cerchio, non sulla circonferenza)
    for _ in range(NEW_PLANTS_PER_FRAME):
        if random.random() < PLANT_SPAWN_CHANCE:
            # spawn interno al cerchio (distribuzione uniforme sulla superficie)
            ang = random.random() * 2 * math.pi
            rad = NO_PRED_ZONE['r'] * math.sqrt(random.random()) * 0.95
            px = NO_PRED_ZONE['x'] + math.cos(ang) * rad
            py = NO_PRED_ZONE['y'] + math.sin(ang) * rad

            # clamp per sicurezza
            px = max(MARGIN + 5, min(WIDTH - MARGIN - 5, px))
            py = max(MARGIN + 5, min(HEIGHT - MARGIN - 5, py))
            p = Plant()
            p.x = px
            p.y = py
            plants.append(p)

    for p in plants:
        pygame.draw.circle(screen, COLOR_PLANT, (int(p.x), int(p.y)), p.radius)

    # Disegna zona protetta per predatori (riempimento semitrasparente + bordo colorato)
    rz = int(NO_PRED_ZONE['r'])
    zx = int(NO_PRED_ZONE['x'])
    zy = int(NO_PRED_ZONE['y'])
    surf = pygame.Surface((rz*2, rz*2), pygame.SRCALPHA)
    # Colore riempimento: verde tenue semitrasparente
    pygame.draw.circle(surf, (60, 200, 140, 90), (rz, rz), rz)
    screen.blit(surf, (zx - rz, zy - rz))
    # Bordo visibile
    pygame.draw.circle(screen, (40, 180, 100), (zx, zy), rz, 3)

    new_gen = []
    n_prey = 0
    vision_data = {}

    for c in creatures[:]:
        if c.is_prey: n_prey += 1
        # predators removed

        inputs, sector_dist, sector_type, fov = c.get_inputs(plants, creatures)
        vision_data[c] = (sector_dist, sector_type, fov)

        out = c.brain.forward(inputs)
        
        c.heading += out[0] * MAX_TURN
        max_s = MAX_SPEED_PREY
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

        # (Perimeter energy bonus removed)

        # Prey eat plants
        if c.is_prey:
            for p in plants[:]:
                if math.hypot(c.x-p.x, c.y-p.y) < c.radius + p.radius:
                    c.energy = min(MAX_ENERGY, c.energy + PREY_EAT_GAIN)
                    plants.remove(p)
                    break

        if not c.update():
            creatures.remove(c)
            if c in vision_data:
                del vision_data[c]
        else:
            limit = PREY_REPRO_AT
            if c.energy > limit:
                # Permetti la riproduzione indipendentemente dalla posizione (anche dentro il cerchio)
                c.energy -= CHILD_ENERGY_VAL
                c.energy -= REPRO_ENERGY_COST
                if c.energy < 0:
                    c.energy = 0
                off_brain = copy.deepcopy(c.brain)
                off_brain.mutate()
                new_gen.append(Creature(c.x, c.y, c.is_prey, off_brain))

            color = COLOR_PREY
            pygame.draw.circle(screen, color, (int(c.x), int(c.y)), int(c.radius))
            pygame.draw.line(screen, (100, 100, 100), (c.x, c.y), 
                            (c.x + math.cos(c.heading)*12, c.y + math.sin(c.heading)*12), 2)

    creatures.extend(new_gen)

    # Gestione creature evidenziate
    if highlight_prey and (highlight_prey not in creatures or highlight_prey.health <= 0):
        highlight_prey = None

    prey_list = [c for c in creatures if c.is_prey and c.health > 0]

    # Creatura selezionata (dal click)
    selected = None
    if highlight_prey and highlight_prey in creatures and highlight_prey.health > 0:
        selected = highlight_prey

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

        # predators removed: nothing to draw here

    # =============================================
    #    PULSANTE RETE NEURALE (unico)
    # =============================================
    # Pulsanti scelta casuale (visibili sempre)
    pygame.draw.rect(screen, (100, 140, 220), RANDOM_PREY_BUTTON, border_radius=6)
    pygame.draw.rect(screen, (220,220,220), RANDOM_PREY_BUTTON, 2, border_radius=6)
    text_rp = font_small.render("Rand Prey", True, (255,255,255))
    screen.blit(text_rp, text_rp.get_rect(center=RANDOM_PREY_BUTTON.center))

    # Random predator button removed

    # Pulsante rete (visibile solo con vision)
    if show_vision:
        pygame.draw.rect(screen, (100, 140, 220) if show_net else (60,60,60), NET_BUTTON, border_radius=6)
        pygame.draw.rect(screen, (220,220,220), NET_BUTTON, 2, border_radius=6)
        text_net = font_small.render("Neural Net", True, (255,255,255))
        screen.blit(text_net, text_net.get_rect(center=NET_BUTTON.center))

    # =============================================
    #    GRAFO TOP COLLEGAMENTI
    # =============================================
    if show_vision and show_net and selected:
        creature = selected
        title = "Prey Neural Network" if creature.is_prey else "Pred Neural Network"
        color_title = (140, 200, 255) if creature.is_prey else (255, 140, 140)

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
    status = f"TIME: {timer} | PREY: {n_prey} | PLANTS: {len(plants)}"
    screen.blit(font.render(status, True, (200, 200, 200)), (27, HEIGHT - 42))
    # Pannello statistiche per la creatura selezionata (clic)
    if selected:
        px = WIDTH - 290
        py = 120
        pw = 260
        ph = 120
        pygame.draw.rect(screen, (40,40,50), (px, py, pw, ph))
        pygame.draw.rect(screen, (120,120,140), (px, py, pw, ph), 2)

        t = "Prey"
        title_s = font.render(f"Selected: {t}", True, (220,220,220))
        screen.blit(title_s, (px + 10, py + 6))

        energy_s = font_small.render(f"Energy: {int(selected.energy)}", True, (200,200,200))
        health_s = font_small.render(f"Health: {int(selected.health)}", True, (200,200,200))
        speed_s = font_small.render(f"Speed: {selected.speed:.2f}", True, (200,200,200))
        pos_s = font_small.render(f"Pos: ({int(selected.x)},{int(selected.y)})", True, (200,200,200))
        age_s = font_small.render(f"Age: {(pygame.time.get_ticks() - selected.birth_time)//1000}s", True, (200,200,200))

        screen.blit(energy_s, (px + 12, py + 36))
        screen.blit(health_s, (px + 12, py + 56))
        screen.blit(speed_s, (px + 12, py + 76))
        screen.blit(pos_s, (px + 12, py + 96))
        screen.blit(age_s, (px + 140, py + 36))
    
    if n_prey == 0:
        screen.blit(font.render("EXTINCTION EVENT", True, (255, 50, 50)), (WIDTH//2 - 80, HEIGHT//2 - 20))

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()