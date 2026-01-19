import pygame
import math
import random
import copy

# ================= CONFIGURAZIONE ECO-SISTEMA CON PERIMETRO =================
WIDTH, HEIGHT = 1000, 700
FPS = 60
MARGIN = 15  # Spessore del "muro" fisico

# Popolazioni
START_PLANTS      = 200
START_PREY        = 100
START_PREDATORS   = 40

# Parametri Energia
MAX_ENERGY        = 200
CHILD_ENERGY_VAL  = 70

# PREDE
PREY_REPRO_AT     = 150
PREY_BASAL_COST   = 0.3 # Leggermente aumentato per bilanciare la facilità di trovare cibo
MAX_SPEED_PREY    = 2.7  

# PREDATORI
PRED_REPRO_AT     = 180
PRED_BASAL_COST   = 0.5  
MAX_SPEED_PRED    = 2.3  
PRED_EAT_GAIN     = 60    

# Dinamiche
MOVE_COST_FACTOR  = 0.025 
MAX_TURN          = math.pi / 20
MAX_VISION        = 160
NEW_PLANTS_PER_FRAME = 2
PLANT_SPAWN_CHANCE = 0.8

# Colori
COLOR_BG          = (15, 15, 22)
COLOR_PLANT       = (60, 220, 80)
COLOR_PREY        = (70, 160, 255)
COLOR_PRED        = (255, 70, 70)
COLOR_WALL        = (50, 50, 65)

# ================= CLASSI IA =================

class NeuralNet:
    def __init__(self):
        # Input: 25 (Setti visivi) + 4 (Muri) = 29
        # Nodi nascosti: 14, Output: 2
        def rw(): return random.gauss(0, 1.0)
        self.w_ih = [[rw() for _ in range(29)] for _ in range(14)]
        self.b_h   = [rw() * 0.1 for _ in range(14)]
        self.w_ho  = [[rw() for _ in range(14)] for _ in range(2)]
        self.b_o   = [rw() * 0.1 for _ in range(2)]

    def forward(self, inputs):
        h = [math.tanh(self.b_h[j] + sum(inputs[i] * self.w_ih[j][i] for i in range(29))) for j in range(14)]
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
        self.energy = 85
        self.health = 100
        self.is_prey = is_prey
        self.brain = brain if brain else NeuralNet()
        self.radius = 6.0 if is_prey else 9.5

    def get_inputs(self, plants, creatures):
        # 1. SENSORI VISIVI (Settori)
        sectors = 5
        sector_dist = [1.0] * sectors
        sector_type = [0] * (sectors * 4)
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
                for i in range(4): sector_type[s*4 + i] = 0
                if hasattr(ent, 'is_prey'): 
                    if ent.is_prey: sector_type[s*4 + 2] = 1 # Preda
                    else:           sector_type[s*4 + 3] = 1 # Predatore
                else:               sector_type[s*4 + 0] = 1 # Pianta
        
        # 2. SENSORI PERIMETRO (Distanza normalizzata dai 4 bordi)
        # 0.0 = sul bordo, 1.0 = lontano/oltre MAX_VISION
        d_top = min(1.0, (self.y - MARGIN) / MAX_VISION)
        d_bottom = min(1.0, (HEIGHT - MARGIN - self.y) / MAX_VISION)
        d_left = min(1.0, (self.x - MARGIN) / MAX_VISION)
        d_right = min(1.0, (WIDTH - MARGIN - self.x) / MAX_VISION)
        
        # Composizione finale input
        res = []
        for i in range(sectors):
            res.append(sector_dist[i])
            res.extend(sector_type[i*4 : i*4+4])
        res.extend([d_top, d_bottom, d_left, d_right])
        return res

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

# ================= CORE SIMULATION =================

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
font = pygame.font.SysFont("Consolas", 14, bold=True)

plants = [Plant() for _ in range(START_PLANTS)]
creatures = [Creature(random.random()*WIDTH, random.random()*HEIGHT, True) for _ in range(START_PREY)]
creatures += [Creature(random.random()*WIDTH, random.random()*HEIGHT, False) for _ in range(START_PREDATORS)]

running = True
start_ticks = pygame.time.get_ticks()

while running:
    screen.fill(COLOR_BG)
    # Disegna il perimetro
    pygame.draw.rect(screen, COLOR_WALL, (MARGIN, MARGIN, WIDTH-2*MARGIN, HEIGHT-2*MARGIN), 2)

    for event in pygame.event.get():
        if event.type == pygame.QUIT: running = False

    for _ in range(NEW_PLANTS_PER_FRAME):
        if random.random() < PLANT_SPAWN_CHANCE:
            plants.append(Plant())

    for p in plants:
        pygame.draw.circle(screen, COLOR_PLANT, (int(p.x), int(p.y)), p.radius)

    new_gen = []
    n_prey, n_pred = 0, 0

    for c in creatures[:]:
        if c.is_prey: n_prey += 1
        else: n_pred += 1

        inputs = c.get_inputs(plants, creatures)
        out = c.brain.forward(inputs)
        
        # Controllo sterzo e velocità
        c.heading += out[0] * MAX_TURN
        max_s = MAX_SPEED_PREY if c.is_prey else MAX_SPEED_PRED
        c.speed = max(0.4, min(max_s, c.speed + out[1] * 0.2))
        
        # Nuovo calcolo posizione con blocco muro
        nx = c.x + math.cos(c.heading) * c.speed
        ny = c.y + math.sin(c.heading) * c.speed

        # Collisione muri: se sbatte, la velocità viene azzerata e la posizione bloccata
        if nx < MARGIN or nx > WIDTH - MARGIN:
            nx = max(MARGIN, min(WIDTH - MARGIN, nx))
            c.speed *= 0.1
        if ny < MARGIN or ny > HEIGHT - MARGIN:
            ny = max(MARGIN, min(HEIGHT - MARGIN, ny))
            c.speed *= 0.1
        
        c.x, c.y = nx, ny

        # Interazioni cibo
        if c.is_prey:
            for p in plants[:]:
                if math.hypot(c.x-p.x, c.y-p.y) < c.radius + p.radius:
                    c.energy = min(MAX_ENERGY, c.energy + 40)
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
        else:
            limit = PREY_REPRO_AT if c.is_prey else PRED_REPRO_AT
            if c.energy > limit:
                c.energy -= CHILD_ENERGY_VAL
                off_brain = copy.deepcopy(c.brain)
                off_brain.mutate()
                new_gen.append(Creature(c.x, c.y, c.is_prey, off_brain))

            color = COLOR_PREY if c.is_prey else COLOR_PRED
            pygame.draw.circle(screen, color, (int(c.x), int(c.y)), int(c.radius))
            # Indicatore direzione
            pygame.draw.line(screen, (100, 100, 100), (c.x, c.y), 
                             (c.x + math.cos(c.heading)*12, c.y + math.sin(c.heading)*12), 2)

    creatures.extend(new_gen)
    
    # UI
    seconds = (pygame.time.get_ticks() - start_ticks) // 1000
    timer = f"{seconds // 60:02}:{seconds % 60:02}"
    status = f"TIME: {timer} | PREY: {n_prey} | PRED: {n_pred} | PLANTS: {len(plants)}"
    screen.blit(font.render(status, True, (200, 200, 200)), (20, HEIGHT - 25))
    
    if n_prey == 0 or n_pred == 0:
        screen.blit(font.render("EXTINCTION EVENT", True, (255, 50, 50)), (WIDTH//2 - 60, HEIGHT//2))

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()