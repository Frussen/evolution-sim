import pygame
import math
import random
import copy

# ================= CONFIGURAZIONE ECO-SISTEMA =================
WIDTH, HEIGHT = 1000, 700
FPS = 60
MARGIN = 15

SECTORS = 5
INPUTS_PER_SECTOR = 5
NUM_WALL_SENSORS = 4
NUM_TERRAIN_SENSORS = 4
NUM_INPUTS = SECTORS * INPUTS_PER_SECTOR + NUM_WALL_SENSORS + NUM_TERRAIN_SENSORS + 2

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
PRED_BASAL_COST   = 0.5
MAX_SPEED_PRED    = 2.5  
PRED_EAT_GAIN     = 70    

MOVE_COST_FACTOR  = 0.025 
MAX_TURN          = math.pi / 15
MAX_VISION        = 160
NEW_PLANTS_PER_FRAME = 2
PLANT_SPAWN_CHANCE = 0.8

THIRST_DECREASE_RATE = 0.1
THIRST_THRESHOLD = 50
MAX_THIRST = 100
DRINK_GAIN = 30

DAY_NIGHT_CYCLE = 1800
NIGHT_VISION_REDUCTION = 0.5

# ================= COLORI =================
COLOR_BG          = (15, 15, 22)
COLOR_PLANT       = (60, 220, 80)
COLOR_PREY        = (70, 160, 255)
COLOR_PRED        = (255, 70, 70)
COLOR_WALL        = (50, 50, 65)
COLOR_WATER       = (0, 100, 255)
COLOR_CARCASS     = (100, 50, 50)
COLOR_TERRAIN_OPEN = (200, 180, 100)
COLOR_TERRAIN_TALL_GRASS = (100, 150, 50)
COLOR_SELECT      = (255, 180, 0)   # Giallo/arancione per selezione

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


class Terrain:
    def __init__(self):
        self.grid = [[random.choice([0, 1]) for _ in range(WIDTH // 20)] for _ in range(HEIGHT // 20)]

    def get_type(self, x, y):
        gx = int(x // 20)
        gy = int(y // 20)
        if 0 <= gx < WIDTH // 20 and 0 <= gy < HEIGHT // 20:
            return self.grid[gy][gx]
        return 0

    def draw(self, screen):
        for y in range(len(self.grid)):
            for x in range(len(self.grid[0])):
                color = COLOR_TERRAIN_OPEN if self.grid[y][x] == 0 else COLOR_TERRAIN_TALL_GRASS
                pygame.draw.rect(screen, color, (x*20, y*20, 20, 20))


class Water:
    def __init__(self):
        self.x = random.uniform(WIDTH * 0.3, WIDTH * 0.7)
        self.y = random.uniform(HEIGHT * 0.3, HEIGHT * 0.7)
        self.radius = 30


class Carcass:
    def __init__(self, x, y, energy):
        self.x = x
        self.y = y
        self.energy = energy / 2
        self.radius = 8
        self.decay = 0.05

    def update(self):
        self.energy -= self.decay
        return self.energy > 0


class Creature:
    def __init__(self, x, y, is_prey, brain=None):
        self.x = max(MARGIN, min(WIDTH - MARGIN, x))
        self.y = max(MARGIN, min(HEIGHT - MARGIN, y))
        self.heading = random.uniform(0, math.pi*2)
        self.speed = 1.2
        self.energy = 100
        self.health = 100
        self.thirst = 100
        self.is_prey = is_prey
        self.brain = brain if brain else NeuralNet()
        self.radius = 6.0 if is_prey else 9.5

    def get_inputs(self, plants, creatures, waters, carcasses, terrain, is_day):
        sectors = SECTORS
        sector_dist = [1.0] * sectors
        sector_type = [0] * (sectors * 4)
        fov = math.pi * 1.6 if self.is_prey else math.pi * 1.2
        vision_mod = 1.0 if is_day else NIGHT_VISION_REDUCTION
        
        entities = plants + creatures + waters + carcasses
        for ent in entities:
            if ent is self: continue
            dx, dy = ent.x - self.x, ent.y - self.y
            dist = math.hypot(dx, dy)
            if dist > MAX_VISION * vision_mod: continue

            angle = (math.atan2(dy, dx) - self.heading + math.pi) % (2*math.pi) - math.pi
            if abs(angle) > fov/2: continue

            s = int((angle + fov/2) / fov * sectors)
            s = max(0, min(sectors-1, s))
            
            d_norm = dist / (MAX_VISION * vision_mod)
            if d_norm < sector_dist[s]:
                sector_dist[s] = d_norm
                for i in range(4): sector_type[s*4 + i] = 0
                if isinstance(ent, Plant): 
                    sector_type[s*4 + 0] = 1
                elif isinstance(ent, Creature):
                    if ent.is_prey: 
                        sector_type[s*4 + 1] = 1
                    else:           
                        sector_type[s*4 + 2] = 1
                elif isinstance(ent, Water) or isinstance(ent, Carcass):
                    sector_type[s*4 + 3] = 1

        d_top    = min(1.0, (self.y - MARGIN) / MAX_VISION)
        d_bottom = min(1.0, (HEIGHT - MARGIN - self.y) / MAX_VISION)
        d_left   = min(1.0, (self.x - MARGIN) / MAX_VISION)
        d_right  = min(1.0, (WIDTH - MARGIN - self.x) / MAX_VISION)
        
        terrain_inputs = [0] * NUM_TERRAIN_SENSORS
        directions = [0, math.pi/2, math.pi, 3*math.pi/2]
        for i, d in enumerate(directions):
            tx = self.x + math.cos(self.heading + d) * 10
            ty = self.y + math.sin(self.heading + d) * 10
            terrain_inputs[i] = terrain.get_type(tx, ty)

        res = []
        for i in range(sectors):
            res.append(sector_dist[i])
            res.extend(sector_type[i*4 : i*4+4])
        res.extend([d_top, d_bottom, d_left, d_right])
        res.extend(terrain_inputs)
        res.append(self.thirst / MAX_THIRST)
        res.append(1.0 if is_day else 0.0)
        return res

    def update(self, waters, carcasses):
        cost = PREY_BASAL_COST if self.is_prey else PRED_BASAL_COST
        self.energy -= cost + (self.speed * MOVE_COST_FACTOR)
        self.thirst -= THIRST_DECREASE_RATE
        
        if self.energy <= 0 or self.thirst <= 0:
            self.health -= 1.5
            self.energy = max(0, self.energy)
            self.thirst = max(0, self.thirst)
        
        drank = False
        for w in waters:
            if math.hypot(self.x - w.x, self.y - w.y) < self.radius + w.radius:
                self.thirst = min(MAX_THIRST, self.thirst + DRINK_GAIN)
                drank = True
                break
        
        if not self.is_prey:
            for carc in carcasses[:]:
                if math.hypot(self.x - carc.x, self.y - carc.y) < self.radius + carc.radius:
                    self.energy = min(MAX_ENERGY, self.energy + carc.energy / 2)
                    carc.energy = 0
                    break
        
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
small_font = pygame.font.SysFont("Consolas", 12)

terrain = Terrain()
plants = [Plant() for _ in range(START_PLANTS)]
waters = [Water() for _ in range(3)]
carcasses = []
creatures = [Creature(random.random()*WIDTH, random.random()*HEIGHT, True) for _ in range(START_PREY)]
creatures += [Creature(random.random()*WIDTH, random.random()*HEIGHT, False) for _ in range(START_PREDATORS)]

running = True
start_ticks = pygame.time.get_ticks()
frame_count = 0
selected_creature = None

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = event.pos
            if event.button == 1:  # sinistro → seleziona
                selected_creature = None
                for c in creatures:
                    if math.hypot(mx - c.x, my - c.y) <= c.radius + 10:
                        selected_creature = c
                        break
            elif event.button == 3:  # destro → deseleziona
                selected_creature = None
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                selected_creature = None

    screen.fill(COLOR_BG)
    terrain.draw(screen)
    pygame.draw.rect(screen, COLOR_WALL, (MARGIN, MARGIN, WIDTH-2*MARGIN, HEIGHT-2*MARGIN), 2)

    # Ciclo giorno/notte
    is_day = (frame_count % DAY_NIGHT_CYCLE) < DAY_NIGHT_CYCLE // 2
    if not is_day:
        night = pygame.Surface((WIDTH, HEIGHT))
        night.fill((0,0,0))
        night.set_alpha(100)
        screen.blit(night, (0,0))

    # Generazione piante
    for _ in range(NEW_PLANTS_PER_FRAME):
        if random.random() < PLANT_SPAWN_CHANCE:
            plants.append(Plant())

    for p in plants:
        pygame.draw.circle(screen, COLOR_PLANT, (int(p.x), int(p.y)), p.radius)

    for w in waters:
        pygame.draw.circle(screen, COLOR_WATER, (int(w.x), int(w.y)), w.radius)

    for carc in carcasses[:]:
        if not carc.update():
            carcasses.remove(carc)
        else:
            pygame.draw.circle(screen, COLOR_CARCASS, (int(carc.x), int(carc.y)), int(carc.radius))

    new_gen = []
    n_prey = 0
    n_pred = 0

    for c in creatures[:]:
        if c.is_prey: n_prey += 1
        else: n_pred += 1

        inputs = c.get_inputs(plants, creatures, waters, carcasses, terrain, is_day)
        out = c.brain.forward(inputs)
        
        c.heading += out[0] * MAX_TURN
        max_s = MAX_SPEED_PREY if c.is_prey else MAX_SPEED_PRED
        terrain_type = terrain.get_type(c.x, c.y)
        speed_mod = 1.0 if terrain_type == 0 else 0.7
        c.speed = max(0.4, min(max_s, c.speed + out[1] * 0.4)) * speed_mod
        
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
                if math.hypot(c.x - p.x, c.y - p.y) < c.radius + p.radius:
                    c.energy = min(MAX_ENERGY, c.energy + PREY_EAT_GAIN)
                    plants.remove(p)
                    break
        else:
            for other in creatures:
                if other.is_prey and other.health > 0:
                    if math.hypot(c.x - other.x, c.y - other.y) < c.radius + other.radius:
                        c.energy = min(MAX_ENERGY, c.energy + PRED_EAT_GAIN)
                        other.health = -100
                        break

        if not c.update(waters, carcasses):
            carcasses.append(Carcass(c.x, c.y, c.energy))
            creatures.remove(c)
        else:
            limit = PREY_REPRO_AT if c.is_prey else PRED_REPRO_AT
            if c.energy > limit:
                c.energy -= CHILD_ENERGY_VAL
                off_brain = copy.deepcopy(c.brain)
                off_brain.mutate()
                new_gen.append(Creature(c.x, c.y, c.is_prey, off_brain))

            color = COLOR_PREY if c.is_prey else COLOR_PRED
            pygame.draw.circle(screen, color, (int(c.x), int(c.y)), int(c.radius))
            pygame.draw.line(screen, (100,100,100),
                             (c.x, c.y),
                             (c.x + math.cos(c.heading)*12, c.y + math.sin(c.heading)*12), 2)

    creatures.extend(new_gen)

    # ================= SELEZIONE CREATURA =================
    if selected_creature and selected_creature in creatures:
        # Highlight
        pygame.draw.circle(screen, COLOR_SELECT, 
                            (int(selected_creature.x), int(selected_creature.y)), 
                            int(selected_creature.radius + 8), 4)

        # Pannello dimensioni
        panel_x, panel_y = WIDTH - 270, HEIGHT - 240
        panel = pygame.Surface((270, 220))
        panel.fill((25, 25, 35))
        panel.set_alpha(220)
        screen.blit(panel, (panel_x, panel_y))

        title = font.render("CREATURA SELEZIONATA", True, (255, 220, 100))
        screen.blit(title, (panel_x + 12, panel_y + 10))

        # Coordinate iniziali testo
        current_y = panel_y + 45
        bar_width = 240
        bar_height = 12
        bar_offset_x = 15

        # Tipo
        screen.blit(small_font.render(
            f"Tipo:  {'Preda' if selected_creature.is_prey else 'Predatore'}", 
            True, (210, 230, 255)), 
            (panel_x + bar_offset_x, current_y))
        current_y += 28

        # Energia + barra
        screen.blit(small_font.render(
            f"Energia: {selected_creature.energy:6.1f} / {MAX_ENERGY}", 
            True, (210, 230, 255)), 
            (panel_x + bar_offset_x, current_y))
        current_y += 18

        # Barra energia
        pygame.draw.rect(screen, (50,50,50), 
                        (panel_x + bar_offset_x, current_y, bar_width, bar_height))
        e_ratio = max(0, min(1, selected_creature.energy / MAX_ENERGY))
        pygame.draw.rect(screen, (100, 220, 100), 
                        (panel_x + bar_offset_x, current_y, bar_width * e_ratio, bar_height))
        current_y += 22

        # Sete + barra
        screen.blit(small_font.render(
            f"Sete:    {selected_creature.thirst:6.1f} / {MAX_THIRST}", 
            True, (210, 230, 255)), 
            (panel_x + bar_offset_x, current_y))
        current_y += 18

        # Barra sete
        pygame.draw.rect(screen, (50,50,50), 
                        (panel_x + bar_offset_x, current_y, bar_width, bar_height))
        t_ratio = max(0, min(1, selected_creature.thirst / MAX_THIRST))
        pygame.draw.rect(screen, (60, 140, 255), 
                        (panel_x + bar_offset_x, current_y, bar_width * t_ratio, bar_height))
        current_y += 22

        # Salute
        screen.blit(small_font.render(
            f"Salute:  {selected_creature.health:6.1f} / 100", 
            True, (210, 230, 255)), 
            (panel_x + bar_offset_x, current_y))
        current_y += 22

        # Velocità
        screen.blit(small_font.render(
            f"Velocità:{selected_creature.speed:7.2f}", 
            True, (210, 230, 255)), 
            (panel_x + bar_offset_x, current_y))
    else:
        hint = small_font.render("Clic sinistro su creatura • destro / ESC per deselezionare", 
                                True, (140, 140, 160))
        screen.blit(hint, (20, HEIGHT - 28))

    # UI generale (rimane invariata)
    seconds = (pygame.time.get_ticks() - start_ticks) // 1000
    timer = f"{seconds // 60:02d}:{seconds % 60:02d}"
    status = f"TIME: {timer} | PREY: {n_prey} | PRED: {n_pred} | PLANTS: {len(plants)} | CARC: {len(carcasses)}"
    screen.blit(font.render(status, True, (200, 200, 200)), (27, HEIGHT - 42))
    
    if n_prey == 0 or n_pred == 0:
        screen.blit(font.render("EXTINCTION EVENT", True, (255, 50, 50)),
                    (WIDTH//2 - 100, HEIGHT//2 - 20))

    pygame.display.flip()
    clock.tick(FPS)
    frame_count += 1

pygame.quit()