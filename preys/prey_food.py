import pygame
import math
import random
from random import uniform
import copy

# Initialize pygame early to query display info so WIDTH/HEIGHT adapt to screen
pygame.init()
info = pygame.display.Info()
screen_w, screen_h = info.current_w, info.current_h

# ================= CONFIGURAZIONE ECO-SISTEMA =================
# Desired max default size (keeps window reasonable on very large screens)
_MAX_DESIRED_W, _MAX_DESIRED_H = 1500, 900
# Use most of the screen but cap to desired maximums
WIDTH = min(_MAX_DESIRED_W, int(screen_w * 1))
HEIGHT = min(_MAX_DESIRED_H, int(screen_h * 0.85))
FPS = 60
MARGIN = 15

SECTORS = 5
INPUTS_PER_SECTOR = 4           # legacy (unused when using simple inputs)
# Sensor set: dx/dy to nearest plant, explicit distance, speed, energy
NUM_INPUTS = 5

# Circle teleport configuration
CIRCLE_TELEPORT_INTERVAL = 3600  # teleport every 3600 frames (~60 seconds at 60 FPS)

START_PLANTS      = 200
START_PREY        = 120

MAX_ENERGY        = 170
CHILD_ENERGY_VAL  = 80

PREY_REPRO_AT     = 150
PREY_BASAL_COST   = 0.8
MAX_SPEED_PREY    = 3.7
PREY_EAT_GAIN     = 90  

# Energy gain per frame when inside the grass circle
PREY_INSIDE_GAIN  = 0

MOVE_COST_FACTOR  = 0.01
MAX_TURN          = math.pi / 15
MAX_VISION        = 160
NEW_PLANTS_PER_FRAME = 3
PLANT_SPAWN_CHANCE = 0.8
# Percentuale di piante che crescono ovunque (0.0 = solo cerchio, 1.0 = solo ovunque)
PLANT_ANYWHERE_RATIO = 1

# ================= VISUALIZZAZIONE =================
show_vision    = False           # inizia con OFF
show_prey_net  = False
show_events    = False            # mostra cerchi nascita/morte

highlight_prey = None

# Colori
COLOR_BG          = (15, 15, 22)
COLOR_PLANT       = (60, 220, 80)
COLOR_PREY        = (70, 160, 255)
COLOR_WALL        = (50, 50, 65)

COLOR_VISION_PREY = (100, 180, 255, 40)

# ================= PULSANTI RETE =================
# Prey Net e Random Prey buttons (DESTRA)
PREY_NET_BUTTON = pygame.Rect(WIDTH - 170, 67, 140, 32)
RANDOM_PREY_BUTTON = pygame.Rect(WIDTH - 170, 107, 140, 32)

# Events button (SINISTRA)
EVENTS_BUTTON = pygame.Rect(27, 107, 127, 37)

WEIGHT_THRESHOLD = 0.20
MAX_EDGES = 90

# ================= SLIDER SPAWN RATIO =================
SLIDER_PLANT_RATIO = pygame.Rect(27, 175, 127, 20)  # x, y, width, height (spostato in basso)
slider_dragging = False

# ================= TRACCIA EVENTI NASCITA/MORTE =================
birth_events = []  # Lista di (x, y, timer) per cerchi verdi
death_events = []  # Lista di (x, y, timer) per cerchi rossi
EVENT_DISPLAY_TIME = 7  # Frame per cui mostrare il cerchio

# ================= CLASSI =================

class NeuralNet:
    def __init__(self):
        def rw(): return random.gauss(0, 1.0)
        # Input → Hidden: NUM_INPUTS (5) → 32 hidden neurons
        self.w_ih = [[rw() for _ in range(NUM_INPUTS)] for _ in range(32)]
        self.b_h   = [rw() * 0.1 for _ in range(32)]
        # Hidden → Output: 32 hidden → 2 outputs (steering, speed)
        self.w_ho  = [[rw() for _ in range(32)] for _ in range(2)]
        self.b_o   = [rw() * 0.1 for _ in range(2)]

    def forward(self, inputs):
        # Hidden layer: 32 neurons with tanh activation
        h = [math.tanh(self.b_h[j] + sum(inputs[i] * self.w_ih[j][i] for i in range(NUM_INPUTS))) for j in range(32)]
        # Output layer: 2 neurons with tanh activation
        out = [math.tanh(self.b_o[j] + sum(h[i] * self.w_ho[j][i] for i in range(32))) for j in range(2)]
        return out

    def mutate(self):
        for layer in (self.w_ih, self.w_ho):
            for row in layer:
                for i in range(len(row)):
                    if random.random() < 0.5: 
                        row[i] += random.gauss(0, 1.5)


class Creature:
    def __init__(self, x, y, is_prey, brain=None):
        self.x = max(MARGIN, min(WIDTH - MARGIN, x))
        self.y = max(MARGIN, min(HEIGHT - MARGIN, y))
        self.heading = random.uniform(0, math.pi*2)
        self.speed = 1.2
        self.prev_speed = 1.2  # Track previous speed for acceleration/deceleration memory
        self.energy = 100
        self.health = 100
        self.age = 0
        self.is_prey = is_prey
        self.brain = brain if brain else NeuralNet()
        self.radius = 6.0 if is_prey else 9.5

    def get_inputs(self, plants):
        # Sensory input: 5 channels for richer decision-making
        # [0] dx_plant_n: relative x to nearest plant (normalized)
        # [1] dy_plant_n: relative y to nearest plant (normalized)
        # [2] dist_plant_n: distance to nearest plant (explicit, helps navigation)
        # [3] speed_n: current speed (memory of movement)
        # [4] energy_n: current energy (urgency to eat)
        scale = max(WIDTH, HEIGHT) / 2.0
        diag = math.hypot(WIDTH, HEIGHT)
        
        # Find nearest plant
        nearest_plant = None
        min_plant_dist = float('inf')
        for p in plants:
            d = math.hypot(p.x - self.x, p.y - self.y)
            if d < min_plant_dist:
                min_plant_dist = d
                nearest_plant = p
        
        if nearest_plant:
            dx_plant = nearest_plant.x - self.x
            dy_plant = nearest_plant.y - self.y
            dx_plant_n = max(-1.0, min(1.0, dx_plant / scale))
            dy_plant_n = max(-1.0, min(1.0, dy_plant / scale))
            dist_plant_n = min(1.0, min_plant_dist / diag)
        else:
            # No plants visible -> zero inputs
            dx_plant_n = 0.0
            dy_plant_n = 0.0
            dist_plant_n = 0.0
        
        # Speed feedback (0.0 to 1.0, normalized to MAX_SPEED_PREY)
        speed_n = min(1.0, self.speed / MAX_SPEED_PREY)
        
        # Energy feedback (0.0 to 1.0, normalized to MAX_ENERGY)
        energy_n = min(1.0, self.energy / MAX_ENERGY)

        return [dx_plant_n, dy_plant_n, dist_plant_n, speed_n, energy_n]

    def update(self):
        cost = PREY_BASAL_COST
        self.energy -= cost + (self.speed * MOVE_COST_FACTOR)
        if self.energy <= 0:
            self.health -= 1.5
            self.energy = 0
        # track lifetime in frames
        self.age += 1
        return self.health > 0


class Plant:
    def __init__(self, anywhere=False):
        # spawn inside grass circle or anywhere on screen
        if anywhere:
            # Spawn anywhere on screen
            self.x = random.uniform(MARGIN + 5, WIDTH - MARGIN - 5)
            self.y = random.uniform(MARGIN + 5, HEIGHT - MARGIN - 5)
        else:
            # spawn only inside grass circle
            while True:
                x = random.uniform(MARGIN + 5, WIDTH - MARGIN - 5)
                y = random.uniform(MARGIN + 5, HEIGHT - MARGIN - 5)
                dist_to_circle = math.hypot(x - grass_circle_x, y - grass_circle_y)
                if dist_to_circle <= grass_circle_radius:
                    self.x = x
                    self.y = y
                    break
        self.radius = 4


# ================= SETUP =================

screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
font = pygame.font.SysFont("Consolas", 14, bold=True)
font_small = pygame.font.SysFont("Consolas", 12)

# Grass spawn circle (must be defined before creating plants)
grass_circle_x = WIDTH // 2
grass_circle_y = HEIGHT // 2
grass_circle_radius = 200
# Start at 45 degrees angle (equal vx and vy)
grass_circle_vx = 1.5
grass_circle_vy = 1.5
grass_circle_velocity_timer = 0
grass_circle_velocity_interval = 120  # change velocity every 120 frames (~2 seconds at 60 FPS)
grass_circle_teleport_timer = 0  # timer for automatic teleportation
grass_circle_moving = True  # toggle for movement
dragging_circle = False
resizing_circle = False
drag_offset_x = 0
drag_offset_y = 0

plants = [Plant(anywhere=random.random() < PLANT_ANYWHERE_RATIO) for _ in range(START_PLANTS)]
creatures = [Creature(random.random()*WIDTH, random.random()*HEIGHT, True) for _ in range(START_PREY)]

running = True
start_ticks = pygame.time.get_ticks()

# Pulsante toggle Vision
# Pulsante toggle Vision (SPOSTATO A DESTRA)
BUTTON_RECT = pygame.Rect(WIDTH - 170, 27, 140, 32)
BUTTON_COLOR_ON  = (140, 140, 140)
BUTTON_COLOR_OFF = (60, 60, 60)

# Button to pause/resume grass circle (SINISTRA)
GRASS_CIRCLE_BUTTON = pygame.Rect(27, 27, 127, 37)
GRASS_BUTTON_COLOR_ON  = (100, 220, 100)
GRASS_BUTTON_COLOR_OFF = (60, 100, 60)

# Button to teleport grass circle (SINISTRA)
GRASS_TELEPORT_BUTTON = pygame.Rect(27, 67, 127, 37)
GRASS_TELEPORT_COLOR = (200, 150, 100)

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if BUTTON_RECT.collidepoint(event.pos):
                show_vision = not show_vision
                if not show_vision:
                    show_prey_net = False
            if GRASS_CIRCLE_BUTTON.collidepoint(event.pos):
                grass_circle_moving = not grass_circle_moving
            if GRASS_TELEPORT_BUTTON.collidepoint(event.pos):
                # Manual teleport
                grass_circle_x = random.uniform(MARGIN + grass_circle_radius, WIDTH - MARGIN - grass_circle_radius)
                grass_circle_y = random.uniform(MARGIN + grass_circle_radius, HEIGHT - MARGIN - grass_circle_radius)
                grass_circle_teleport_timer = 0
            # Check if clicking on slider
            if SLIDER_PLANT_RATIO.collidepoint(event.pos):
                slider_dragging = True
            if show_vision and PREY_NET_BUTTON.collidepoint(event.pos):
                show_prey_net = not show_prey_net
            if show_vision and RANDOM_PREY_BUTTON.collidepoint(event.pos):
                # pick a random living prey as highlight
                prey_list_click = [c for c in creatures if c.is_prey and c.health > 0]
                if prey_list_click:
                    highlight_prey = random.choice(prey_list_click)
            if EVENTS_BUTTON.collidepoint(event.pos):
                show_events = not show_events

            # select prey by clicking on them (left click)
            if event.button == 1:
                for c in creatures:
                    if c.is_prey and c.health > 0:
                        if math.hypot(c.x - event.pos[0], c.y - event.pos[1]) <= c.radius + 3:
                            highlight_prey = c
                            break

                # check if clicking on grass circle edge to resize
                dist_to_center = math.hypot(event.pos[0] - grass_circle_x, event.pos[1] - grass_circle_y)
                if abs(dist_to_center - grass_circle_radius) < 15:  # within 15px of edge
                    resizing_circle = True
                # check if clicking on grass circle to drag
                elif dist_to_center <= grass_circle_radius + 5:
                    dragging_circle = True
                    drag_offset_x = event.pos[0] - grass_circle_x
                    drag_offset_y = event.pos[1] - grass_circle_y

        elif event.type == pygame.MOUSEBUTTONUP:
            dragging_circle = False
            resizing_circle = False
            slider_dragging = False

        elif event.type == pygame.MOUSEMOTION:
            # Handle slider dragging
            if slider_dragging:
                # Map mouse x to ratio [0, 1]
                relative_x = event.pos[0] - SLIDER_PLANT_RATIO.x
                PLANT_ANYWHERE_RATIO = max(0.0, min(1.0, relative_x / SLIDER_PLANT_RATIO.width))
            
            if dragging_circle:
                grass_circle_x = event.pos[0] - drag_offset_x
                grass_circle_y = event.pos[1] - drag_offset_y
                # clamp to screen
                grass_circle_x = max(MARGIN, min(WIDTH - MARGIN, grass_circle_x))
                grass_circle_y = max(MARGIN, min(HEIGHT - MARGIN, grass_circle_y))

            if resizing_circle:
                dist = math.hypot(event.pos[0] - grass_circle_x, event.pos[1] - grass_circle_y)
                grass_circle_radius = max(30, min(500, int(dist)))

    # Update grass circle position (unless dragging)
    if not dragging_circle and grass_circle_moving:
        # Update velocity timer and randomly change speed magnitude
        grass_circle_velocity_timer += 1
        if grass_circle_velocity_timer >= grass_circle_velocity_interval:
            # Get current velocity magnitude and direction
            current_speed = math.hypot(grass_circle_vx, grass_circle_vy)
            if current_speed > 0:
                direction_x = grass_circle_vx / current_speed
                direction_y = grass_circle_vy / current_speed
            else:
                direction_x, direction_y = 1, 0
            
            # Change speed magnitude but keep direction
            new_speed = random.uniform(0.5, 4)
            grass_circle_vx = direction_x * new_speed
            grass_circle_vy = direction_y * new_speed
            grass_circle_velocity_timer = 0
        # Auto-teleport timer
        grass_circle_teleport_timer += 1
        if grass_circle_teleport_timer >= CIRCLE_TELEPORT_INTERVAL:
            grass_circle_x = random.uniform(MARGIN + grass_circle_radius, WIDTH - MARGIN - grass_circle_radius)
            grass_circle_y = random.uniform(MARGIN + grass_circle_radius, HEIGHT - MARGIN - grass_circle_radius)
            grass_circle_teleport_timer = 0
        grass_circle_x += grass_circle_vx
        grass_circle_y += grass_circle_vy

        # Bounce off edges
        if grass_circle_x - grass_circle_radius <= MARGIN or grass_circle_x + grass_circle_radius >= WIDTH - MARGIN:
            grass_circle_vx *= -1
            grass_circle_x = max(MARGIN + grass_circle_radius, min(WIDTH - MARGIN - grass_circle_radius, grass_circle_x))

        if grass_circle_y - grass_circle_radius <= MARGIN or grass_circle_y + grass_circle_radius >= HEIGHT - MARGIN:
            grass_circle_vy *= -1
            grass_circle_y = max(MARGIN + grass_circle_radius, min(HEIGHT - MARGIN - grass_circle_radius, grass_circle_y))

    screen.fill(COLOR_BG)
    pygame.draw.rect(screen, COLOR_WALL, (MARGIN, MARGIN, WIDTH-2*MARGIN, HEIGHT-2*MARGIN), 2)

    # Generazione piante
    for _ in range(NEW_PLANTS_PER_FRAME):
        if random.random() < PLANT_SPAWN_CHANCE:
            # Uso PLANT_ANYWHERE_RATIO per determinare dove far crescere la pianta
            plants.append(Plant(anywhere=random.random() < PLANT_ANYWHERE_RATIO))

    for p in plants:
        pygame.draw.circle(screen, COLOR_PLANT, (int(p.x), int(p.y)), p.radius)

    new_gen = []
    n_prey = 0
    vision_data = {}

    for c in creatures[:]:
        if c.is_prey: n_prey += 1

        inputs = c.get_inputs(plants)

        out = c.brain.forward(inputs)
        
        c.heading += out[0] * MAX_TURN
        max_s = MAX_SPEED_PREY
        c.prev_speed = c.speed  # Save current speed before updating
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

        # Eating plants (restored)
        for p in plants[:]:
            if math.hypot(c.x-p.x, c.y-p.y) < c.radius + p.radius:
                c.energy = min(MAX_ENERGY, c.energy + PREY_EAT_GAIN)
                try:
                    plants.remove(p)
                except ValueError:
                    pass
                break

        # Energy gain when inside the grass circle (kept)
        if c.is_prey:
            d_to_circle = math.hypot(c.x - grass_circle_x, c.y - grass_circle_y)
            if d_to_circle <= grass_circle_radius:
                c.energy = min(MAX_ENERGY, c.energy + PREY_INSIDE_GAIN)

        if not c.update():
            death_events.append((c.x, c.y, EVENT_DISPLAY_TIME))  # Traccia morte
            creatures.remove(c)
            if c in vision_data:
                del vision_data[c]
        else:
            limit = PREY_REPRO_AT
            if c.energy > limit:
                c.energy -= CHILD_ENERGY_VAL
                off_brain = copy.deepcopy(c.brain)
                off_brain.mutate()
                offspring = Creature(c.x, c.y, c.is_prey, off_brain)
                birth_events.append((offspring.x, offspring.y, EVENT_DISPLAY_TIME))  # Traccia nascita
                new_gen.append(offspring)

        color = COLOR_PREY
        pygame.draw.circle(screen, color, (int(c.x), int(c.y)), int(c.radius))
        pygame.draw.line(screen, (100, 100, 100), (c.x, c.y), 
                        (c.x + math.cos(c.heading)*12, c.y + math.sin(c.heading)*12), 2)

    creatures.extend(new_gen)

    # Aggiorna e visualizza eventi di nascita/morte
    birth_events = [(x, y, t-1) for x, y, t in birth_events if t > 0]
    death_events = [(x, y, t-1) for x, y, t in death_events if t > 0]
    
    # Visualizza eventi di nascita/morte se attivi
    if show_events:
        for x, y, timer in birth_events:
            alpha = int(255 * (timer / EVENT_DISPLAY_TIME))  # Fade out effect
            pygame.draw.circle(screen, (60, 220, 80), (int(x), int(y)), 15, 3)
        
        for x, y, timer in death_events:
            alpha = int(255 * (timer / EVENT_DISPLAY_TIME))  # Fade out effect
            pygame.draw.circle(screen, (220, 60, 60), (int(x), int(y)), 15, 3)

    # Gestione creature evidenziate
    if highlight_prey and (highlight_prey not in creatures or highlight_prey.health <= 0):
        highlight_prey = None

    prey_list = [c for c in creatures if c.is_prey and c.health > 0]

    if highlight_prey is None and prey_list:
        highlight_prey = random.choice(prey_list)

    # =============================================
    #    DISEGNO VISION SOLO SE ATTIVA
    # =============================================
    if show_vision and highlight_prey:
        # Visualizza tutti i 5 input della preda
        inputs = highlight_prey.get_inputs(plants)
        
        # Input 0-1: direzione verso pianta più vicina (dx_plant_n, dy_plant_n)
        # Input 2: distanza esplicita dalla pianta
        # Input 3: velocità corrente
        # Input 4: energia corrente
        dx_plant_n, dy_plant_n = inputs[0], inputs[1]
        dist_plant_n = inputs[2]
        speed_n = inputs[3]
        energy_n = inputs[4]
        
        # Linea verde verso la pianta più vicina
        if abs(dx_plant_n) > 0.01 or abs(dy_plant_n) > 0.01:
            line_len = 100
            end_x = highlight_prey.x + dx_plant_n * line_len
            end_y = highlight_prey.y + dy_plant_n * line_len
            pygame.draw.line(screen, (60, 220, 80), (int(highlight_prey.x), int(highlight_prey.y)), (int(end_x), int(end_y)), 2)

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
                        if t == 0:   color = (100, 255, 140)   # plant
                        elif t == 1: color = (120, 200, 255)   # prey
                        elif t == 2: color = (255, 0, 0)   # circle
                        break

                ex = creature.x + math.cos(ang) * dist
                ey = creature.y + math.sin(ang) * dist

                pygame.draw.line(screen, (*color, 140), 
                               (creature.x, creature.y), (ex, ey), 1)
                pygame.draw.circle(screen, color, (int(ex), int(ey)), 5, 2)

        if highlight_prey:
            # Highlight circle around selected prey
            pygame.draw.circle(screen, (220,220,80), (int(highlight_prey.x), int(highlight_prey.y)), 11, 2)

        # Draw stats panel for selected prey
        if highlight_prey:
            panel_w, panel_h = 220, 140
            panel_x = WIDTH - panel_w - 20
            panel_y = HEIGHT - panel_h - 20
            pygame.draw.rect(screen, (30,30,36), (panel_x, panel_y, panel_w, panel_h), border_radius=8)
            pygame.draw.rect(screen, (100,100,110), (panel_x, panel_y, panel_w, panel_h), 2, border_radius=8)

            # Compute selected prey input values for display
            # Find nearest plant for display
            nearest_plant = None
            min_plant_dist = float('inf')
            for p in plants:
                d = math.hypot(p.x - highlight_prey.x, p.y - highlight_prey.y)
                if d < min_plant_dist:
                    min_plant_dist = d
                    nearest_plant = p
            
            lines = []
            lines.append(f"Type: Prey")
            lines.append(f"Energy: {int(highlight_prey.energy)} / {MAX_ENERGY}")
            lines.append(f"Health: {int(highlight_prey.health)}")
            lifetime_s = highlight_prey.age / FPS
            lines.append(f"Lifetime: {lifetime_s:.1f}s ({highlight_prey.age}f)")
            lines.append(f"Speed: {highlight_prey.speed:.2f}")
            lines.append(f"Pos: {int(highlight_prey.x)},{int(highlight_prey.y)}")
            if nearest_plant:
                lines.append(f"Nearest plant: {min_plant_dist:.1f}px")
            else:
                lines.append(f"Nearest plant: NONE")

            text_y = panel_y + 10
            for line in lines:
                surf = font_small.render(line, True, (220,220,220))
                screen.blit(surf, (panel_x + 12, text_y))
                text_y += 18

            # Draw visual aid: line to nearest plant only
            if nearest_plant:
                pygame.draw.line(screen, (60, 220, 80), (int(highlight_prey.x), int(highlight_prey.y)), (int(nearest_plant.x), int(nearest_plant.y)), 2)

    # =============================================
    #    PULSANTI RETE NEURALE
    # =============================================
    if show_vision:
        pygame.draw.rect(screen, (100, 140, 220), PREY_NET_BUTTON, border_radius=6)
        pygame.draw.rect(screen, (220,220,220), PREY_NET_BUTTON, 2, border_radius=6)
        text_prey = font_small.render("Prey Net", True, (255,255,255))
        screen.blit(text_prey, text_prey.get_rect(center=PREY_NET_BUTTON.center))

        # Random Prey button
        pygame.draw.rect(screen, (120, 180, 120), RANDOM_PREY_BUTTON, border_radius=6)
        pygame.draw.rect(screen, (220,220,220), RANDOM_PREY_BUTTON, 2, border_radius=6)
        text_rand = font_small.render("Random Prey", True, (255,255,255))
        screen.blit(text_rand, text_rand.get_rect(center=RANDOM_PREY_BUTTON.center))

    # =============================================
    #    GRAFO TOP COLLEGAMENTI
    # =============================================
    if show_vision and show_prey_net:
        creature = highlight_prey
        title = "Prey Neural Network"
        color_title = (140, 200, 255)

        if creature:
            brain = creature.brain

            edges = []

            # Input → Hidden
            for i in range(NUM_INPUTS):
                for h in range(32):
                    w = brain.w_ih[h][i]
                    edges.append((("I"+str(i)), ("H"+str(h)), w))

            # Hidden → Output
            for h in range(32):
                for o in range(2):
                    w = brain.w_ho[o][h]
                    edges.append((("H"+str(h)), ("O"+str(o)), w))

            # Top collegamenti per |peso|
            edges.sort(key=lambda x: abs(x[2]), reverse=True)
            top_edges = [e for e in edges if abs(e[2]) >= WEIGHT_THRESHOLD][:MAX_EDGES]

            # Posizioni nodi (centered on screen)
            center_x = WIDTH // 2
            x_offset = min(320, WIDTH // 6)
            col_x = [center_x - x_offset, center_x, center_x + x_offset]
            y_spacing = 27

            # Vertical arrangement: center each column vertically
            input_top = max(60, HEIGHT // 2 - (NUM_INPUTS * y_spacing) // 2)
            hidden_top = max(60, HEIGHT // 2 - (32 * y_spacing) // 2)

            node_y = {}
            for i in range(NUM_INPUTS):
                node_y["I"+str(i)] = input_top + i * y_spacing
            for h in range(32):
                node_y["H"+str(h)] = hidden_top + h * y_spacing

            node_y["O0"] = HEIGHT // 2 - y_spacing // 2
            node_y["O1"] = HEIGHT // 2 + y_spacing // 2

            # Titolo: posiziona sopra la rete (sopra il nodo più alto)
            title_surf = font.render(title, True, color_title)
            top_node_y = min(node_y.values()) if node_y else input_top
            title_y = max(24, top_node_y - 28)
            screen.blit(title_surf, (center_x - title_surf.get_width()//2, title_y))

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
    #    DRAW GRASS SPAWN CIRCLE
    # =============================================
    # Draw transparent green fill inside the circle
    grass_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    pygame.draw.circle(grass_surface, (60, 220, 80, 50), (int(grass_circle_x), int(grass_circle_y)), grass_circle_radius)
    screen.blit(grass_surface, (0, 0))
    
    # Draw circle circumference in grass color
    pygame.draw.circle(screen, COLOR_PLANT, (int(grass_circle_x), int(grass_circle_y)), grass_circle_radius, 3)

    # =============================================
    #    PULSANTE TOGGLE VISION
    # =============================================
    pygame.draw.rect(screen, BUTTON_COLOR_ON if show_vision else BUTTON_COLOR_OFF, BUTTON_RECT, border_radius=6)
    pygame.draw.rect(screen, (220,220,220), BUTTON_RECT, 2, border_radius=6)
    
    btn_text = font_small.render(f"Vision: {'ON' if show_vision else 'OFF'}", True, (255,255,255))
    text_rect = btn_text.get_rect(center=BUTTON_RECT.center)
    screen.blit(btn_text, text_rect)

    # Grass Circle pause/resume button
    pygame.draw.rect(screen, GRASS_BUTTON_COLOR_ON if grass_circle_moving else GRASS_BUTTON_COLOR_OFF, GRASS_CIRCLE_BUTTON, border_radius=6)
    pygame.draw.rect(screen, (220,220,220), GRASS_CIRCLE_BUTTON, 2, border_radius=6)
    
    grass_btn_text = font_small.render(f"Circle: {'RUN' if grass_circle_moving else 'STOP'}", True, (255,255,255))
    grass_text_rect = grass_btn_text.get_rect(center=GRASS_CIRCLE_BUTTON.center)
    screen.blit(grass_btn_text, grass_text_rect)

    # Teleport button
    pygame.draw.rect(screen, GRASS_TELEPORT_COLOR, GRASS_TELEPORT_BUTTON, border_radius=6)
    pygame.draw.rect(screen, (220,220,220), GRASS_TELEPORT_BUTTON, 2, border_radius=6)
    
    teleport_btn_text = font_small.render("Teleport", True, (255,255,255))
    teleport_text_rect = teleport_btn_text.get_rect(center=GRASS_TELEPORT_BUTTON.center)
    screen.blit(teleport_btn_text, teleport_text_rect)

    # Events button (show/hide birth/death circles)
    pygame.draw.rect(screen, (200, 100, 150) if show_events else (80, 40, 60), EVENTS_BUTTON, border_radius=6)
    pygame.draw.rect(screen, (220,220,220), EVENTS_BUTTON, 2, border_radius=6)
    
    events_btn_text = font_small.render(f"Events: {'ON' if show_events else 'OFF'}", True, (255,255,255))
    events_text_rect = events_btn_text.get_rect(center=EVENTS_BUTTON.center)
    screen.blit(events_btn_text, events_text_rect)

    # Draw slider for plant spawn ratio
    pygame.draw.rect(screen, (60, 60, 80), SLIDER_PLANT_RATIO, border_radius=4)
    pygame.draw.rect(screen, (150, 150, 150), SLIDER_PLANT_RATIO, 2, border_radius=4)
    
    # Draw knob (draggable indicator)
    knob_x = SLIDER_PLANT_RATIO.x + PLANT_ANYWHERE_RATIO * SLIDER_PLANT_RATIO.width
    knob_y = SLIDER_PLANT_RATIO.y + SLIDER_PLANT_RATIO.height / 2
    pygame.draw.circle(screen, (220, 180, 100), (int(knob_x), int(knob_y)), 8)
    
    # Label
    ratio_text = font_small.render(f"Plant ratio: {int(PLANT_ANYWHERE_RATIO*100)}% anywhere", True, (220, 220, 220))
    screen.blit(ratio_text, (SLIDER_PLANT_RATIO.x, SLIDER_PLANT_RATIO.y - 18))

    # UI info
    seconds = (pygame.time.get_ticks() - start_ticks) // 1000
    timer = f"{seconds // 60:02}:{seconds % 60:02}"
    status = f"TIME: {timer} | PREY: {n_prey} | PLANTS: {len(plants)}"
    screen.blit(font.render(status, True, (200, 200, 200)), (27, HEIGHT - 42))
    
    if n_prey == 0:
        screen.blit(font.render("EXTINCTION EVENT", True, (255, 50, 50)), (WIDTH//2 - 80, HEIGHT//2 - 20))

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()