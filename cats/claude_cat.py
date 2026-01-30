import pygame
import pymunk
import numpy as np
import random
import math
from collections import deque

# Configurazione simulazione
WIDTH, HEIGHT = 1400, 700
FPS = 60
POPULATION_SIZE = 40
SIMULATION_STEPS = 800
MUTATION_RATE = 0.15
MUTATION_STRENGTH = 0.4
ELITE_RATIO = 0.25
CROSSOVER_RATE = 0.7
SEED_MUTATION_RATE = 0.25
SEED_MUTATION_STRENGTH = 0.6
GROUND_Y = 480

# Colori migliorati
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
SKY_TOP = (135, 206, 250)
SKY_BOTTOM = (220, 240, 255)
GRASS_LIGHT = (124, 252, 120)
GRASS_DARK = (60, 179, 113)
ORANGE_CAT = (255, 160, 50)
ORANGE_DARK = (200, 100, 20)
ORANGE_LIGHT = (255, 200, 100)
CREAM = (255, 240, 210)
PINK = (255, 192, 203)
BROWN = (139, 90, 43)
GRAY = (100, 100, 100)

MAX_WATCH_CATS = 15
HISTORY_LENGTH = 100


class NeuralNetwork:
    """Rete neurale con architettura migliorata: 24 input -> 16 hidden -> 8 hidden -> 8 output"""
    def __init__(self, input_size=24, hidden1_size=16, hidden2_size=8, output_size=8, weights=None):
        self.input_size = input_size
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.output_size = output_size
        
        if weights is None:
            # Xavier initialization per convergenza più rapida
            self.W1 = np.random.randn(input_size, hidden1_size) * np.sqrt(2.0/input_size)
            self.b1 = np.zeros(hidden1_size)
            self.W2 = np.random.randn(hidden1_size, hidden2_size) * np.sqrt(2.0/hidden1_size)
            self.b2 = np.zeros(hidden2_size)
            self.W3 = np.random.randn(hidden2_size, output_size) * np.sqrt(2.0/hidden2_size)
            self.b3 = np.zeros(output_size)
        else:
            self.set_weights(weights)
    
    def forward(self, x):
        """Forward pass con attivazioni tanh per controllo motorio fluido"""
        x = np.array(x, dtype=np.float32)
        # Primo layer hidden
        h1 = np.tanh(np.dot(x, self.W1) + self.b1)
        # Secondo layer hidden
        h2 = np.tanh(np.dot(h1, self.W2) + self.b2)
        # Output layer
        o = np.tanh(np.dot(h2, self.W3) + self.b3)
        return o
    
    def get_weights(self):
        return [self.W1.copy(), self.b1.copy(), self.W2.copy(), self.b2.copy(), 
                self.W3.copy(), self.b3.copy()]
    
    def set_weights(self, weights):
        self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = weights
    
    def mutate(self, rate=MUTATION_RATE, strength=MUTATION_STRENGTH):
        """Mutazione gaussiana dei pesi"""
        for W in [self.W1, self.W2, self.W3]:
            mask = np.random.random(W.shape) < rate
            W += mask * np.random.randn(*W.shape) * strength
        for b in [self.b1, self.b2, self.b3]:
            mask = np.random.random(b.shape) < rate
            b += mask * np.random.randn(*b.shape) * strength


class Cat:
    """Gatto articolato con muscoli spring-damper e sensori migliorati"""
    def __init__(self, space, x_start, brain=None, color_hue=0):
        self.space = space
        self.start_x = x_start
        self.fitness = 0
        self.alive = True
        self.steps = 0
        self.brain = brain if brain else NeuralNetwork()
        self.color_hue = color_hue  # Per variazioni di colore individuali
        
        # Dimensioni corpo più realistiche
        self.body_width, self.body_height = 90, 38
        self.leg_width, self.leg_height = 10, 32
        
        # Statistiche fitness
        self.max_x = x_start
        self.distance_traveled = 0
        self.fall_penalty = 0
        self.stability_bonus = 0
        self.ground_contact_time = 0
        self.joint_drag_penalty = 0
        
        # Costruzione corpo principale (massa aumentata per dominare le gambe)
        mass = 18
        moment = pymunk.moment_for_box(mass, (self.body_width, self.body_height))
        self.body = pymunk.Body(mass, moment)
        self.body.position = (x_start, 340)
        
        self.body_shape = pymunk.Poly.create_box(self.body, (self.body_width, self.body_height))
        self.body_shape.friction = 0.8
        self.body_shape.filter = pymunk.ShapeFilter(group=1)
        self.space.add(self.body, self.body_shape)
        
        self.legs = []
        half_w = self.body_width/2
        half_h = self.body_height/2
        
        # Posizionamento gambe: anteriori e posteriori con offset realistico
        leg_positions = [
            (half_w - 15, half_h),   # Anteriore destra
            (half_w - 15, half_h),   # Anteriore destra (doppia)
            (-half_w + 15, half_h),  # Posteriore sinistra
            (-half_w + 15, half_h)   # Posteriore sinistra (doppia)
        ]
        
        for i, (offset_x, offset_y) in enumerate(leg_positions):
            is_front = i < 2
            
            # Coscia (thigh) - massa ridotta per bilanciamento
            thigh_mass = 1.2
            thigh_moment = pymunk.moment_for_segment(thigh_mass, (0,0), (0, self.leg_height), self.leg_width/2)
            thigh = pymunk.Body(thigh_mass, thigh_moment)
            
            thigh_x = x_start + offset_x
            thigh_y = 340 + offset_y
            thigh.position = (thigh_x, thigh_y)
            
            # Angoli iniziali più naturali
            if is_front:
                thigh.angle = -0.2 if i == 0 else 0.2
            else:
                thigh.angle = 0.25 if i == 2 else -0.25
            
            thigh_shape = pymunk.Segment(thigh, (0, 0), (0, self.leg_height), self.leg_width/2)
            thigh_shape.friction = 1.3
            thigh_shape.filter = pymunk.ShapeFilter(group=1)
            self.space.add(thigh, thigh_shape)
            
            # Joint corpo-coscia con limite rotazione
            joint = pymunk.PivotJoint(self.body, thigh, (thigh_x, thigh_y))
            self.space.add(joint)
            
            # Limiti angolari realistici per biomeccanica felina
            if is_front:
                limit = pymunk.RotaryLimitJoint(self.body, thigh, -math.pi/3.5, math.pi/3.5)
            else:
                limit = pymunk.RotaryLimitJoint(self.body, thigh, -math.pi/2.8, math.pi/2.8)
            self.space.add(limit)
            
            # Motore per il muscolo della coscia (forza ridotta per realismo)
            motor_thigh = pymunk.SimpleMotor(self.body, thigh, 0)
            motor_thigh.max_force = 1800000
            self.space.add(motor_thigh)
            
            # Polpaccio (calf) - massa ridotta per bilanciamento
            calf_mass = 0.8
            calf_len = self.leg_height * 0.9
            calf_moment = pymunk.moment_for_segment(calf_mass, (0,0), (0, calf_len), self.leg_width/2)
            calf = pymunk.Body(calf_mass, calf_moment)
            
            calf_x = thigh_x + math.sin(thigh.angle) * self.leg_height * 0.15
            calf_y = thigh_y + math.cos(thigh.angle) * self.leg_height
            calf.position = (calf_x, calf_y)
            calf.angle = thigh.angle
            
            calf_shape = pymunk.Segment(calf, (0, 0), (0, calf_len), self.leg_width/2)
            calf_shape.friction = 1.6
            calf_shape.filter = pymunk.ShapeFilter(group=1)
            self.space.add(calf, calf_shape)
            
            # Joint ginocchio
            knee_joint = pymunk.PivotJoint(thigh, calf, (calf_x, calf_y))
            self.space.add(knee_joint)
            knee_limit = pymunk.RotaryLimitJoint(thigh, calf, -math.pi/2.0, 0.2)
            self.space.add(knee_limit)
            
            # Motore per il muscolo del polpaccio (forza ridotta per realismo)
            motor_calf = pymunk.SimpleMotor(thigh, calf, 0)
            motor_calf.max_force = 1200000
            self.space.add(motor_calf)
            
            self.legs.append({
                'thigh': thigh, 'calf': calf,
                'thigh_motor': motor_thigh, 'calf_motor': motor_calf,
                'is_front': is_front,
                'ground_contact': False
            })
        
        # Coda per bilanciamento (massa ridotta)
        tail_mass = 0.6
        tail_moment = pymunk.moment_for_segment(tail_mass, (0,0), (-40, -18), 6)
        self.tail = pymunk.Body(tail_mass, tail_moment)
        tail_x = x_start - half_w
        tail_y = 340 - 8
        self.tail.position = (tail_x, tail_y)
        tail_shape = pymunk.Segment(self.tail, (0, 0), (-40, -18), 6)
        tail_shape.friction = 0.4
        tail_shape.filter = pymunk.ShapeFilter(group=1)
        self.space.add(self.tail, tail_shape)
        
        tail_joint = pymunk.PivotJoint(self.body, self.tail, (tail_x, tail_y))
        self.space.add(tail_joint)
        tail_limit = pymunk.RotaryLimitJoint(self.body, self.tail, -math.pi/2.5, math.pi/2.5)
        self.space.add(tail_limit)
        
        # Stabilizzazione iniziale
        for _ in range(20):
            space.step(1/FPS)
    
    def get_nn_input(self):
        """Input sensoriali estesi per la rete neurale (24 inputs)"""
        inputs = []
        
        # 1-3: Orientamento e velocità corpo
        angle_body = self.body.angle % (2*math.pi)
        if angle_body > math.pi:
            angle_body -= 2*math.pi
        inputs.append(angle_body / math.pi)
        
        vx, vy = self.body.velocity
        inputs.append(np.clip(vx / 150.0, -1, 1))
        inputs.append(np.clip(vy / 150.0, -1, 1))
        
        # 4: Altezza dal suolo
        inputs.append((GROUND_Y - self.body.position.y) / 250.0)
        
        # 5: Velocità angolare corpo (per bilanciamento)
        inputs.append(np.clip(self.body.angular_velocity / 5.0, -1, 1))
        
        # 6-21: Per ogni gamba (4 gambe × 4 sensori = 16 inputs)
        for leg in self.legs:
            # Angolo coscia relativo al corpo
            thigh_angle = (leg['thigh'].angle - self.body.angle) % (2*math.pi)
            if thigh_angle > math.pi:
                thigh_angle -= 2*math.pi
            inputs.append(thigh_angle / math.pi)
            
            # Angolo polpaccio relativo alla coscia
            calf_angle = (leg['calf'].angle - leg['thigh'].angle) % (2*math.pi)
            if calf_angle > math.pi:
                calf_angle -= 2*math.pi
            inputs.append(calf_angle / math.pi)
            
            # Velocità angolare coscia
            inputs.append(np.clip(leg['thigh'].angular_velocity / 15.0, -1, 1))
            
            # Velocità angolare polpaccio
            inputs.append(np.clip(leg['calf'].angular_velocity / 15.0, -1, 1))
        
        # 22-23: Angolo e velocità angolare coda (per bilanciamento)
        tail_angle = (self.tail.angle - self.body.angle) % (2*math.pi)
        if tail_angle > math.pi:
            tail_angle -= 2*math.pi
        inputs.append(tail_angle / math.pi)
        inputs.append(np.clip(self.tail.angular_velocity / 10.0, -1, 1))
        
        # 24: Bias (sempre 1.0)
        inputs.append(1.0)
        
        return inputs
    
    def act(self):
        """Controllo muscolare via rete neurale"""
        inputs = self.get_nn_input()
        outputs = self.brain.forward(inputs)
        
        # Applica outputs ai motori delle gambe con forza maggiorata
        for i, leg in enumerate(self.legs):
            thigh_speed = outputs[i*2] * 4.5
            calf_speed = outputs[i*2 + 1] * 4.0
            leg['thigh_motor'].rate = thigh_speed
            leg['calf_motor'].rate = calf_speed
    
    def update(self):
        """Update logica gatto con fitness avanzata"""
        self.act()
        self.steps += 1
        
        # Calcolo distanza percorsa
        current_x = self.body.position.x
        if current_x > self.max_x:
            self.max_x = current_x
        self.distance_traveled = current_x - self.start_x
        
        # Fitness base: distanza verso destra
        self.fitness = self.distance_traveled
        
        # Penalità per articolazioni intermedie che toccano il terreno
        # (ginocchia/gomiti - ovvero le posizioni delle "calf" che si connettono alle "thigh")
        if self.steps > 60:
            for leg in self.legs:
                # Posizione ginocchio/gomito (punto di connessione tra thigh e calf)
                knee_y = leg['calf'].position.y - (self.leg_height * 0.9) * abs(math.cos(leg['calf'].angle))
                
                # Se il ginocchio è troppo vicino al terreno (entro 10 pixel), applica penalità
                if knee_y >= GROUND_Y - 10:
                    self.joint_drag_penalty += 0.3  # Penalità progressiva
        
        # Penalità per instabilità (dopo warm-up iniziale)
        if self.steps > 60:
            body_angle = abs(self.body.angle)
            
            # Penalità per caduta o capovolgimento
            if self.body.position.y > GROUND_Y + 40 or body_angle > 2.2:
                self.alive = False
                self.fall_penalty = 50
                self.fitness -= self.fall_penalty
            
            # Bonus per stabilità (angolo vicino a zero)
            if body_angle < 0.4:
                self.stability_bonus += 0.1
        
        # Bonus per movimento in avanti costante
        if self.steps > 100:
            avg_speed = self.distance_traveled / self.steps
            if avg_speed > 0.5:
                self.fitness += avg_speed * 2
        
        # Applica penalità articolazioni
        self.fitness -= self.joint_drag_penalty
        self.fitness += self.stability_bonus
        
        return self.alive


class Evolution:
    """Sistema di evoluzione con selezione per torneo e crossover migliorato"""
    def __init__(self):
        self.population = []
        self.generation = 1
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.best_brain = None
        self.best_fitness_ever = -float('inf')
        self.top_brains = []
        self.stagnation_counter = 0
        
    def create_population(self, space, start_x):
        self.population = []
        for i in range(POPULATION_SIZE):
            brain = NeuralNetwork() if self.best_brain is None else self._mutate_brain(self.best_brain)
            cat = Cat(space, start_x, brain)
            self.population.append(cat)
    
    def _mutate_brain(self, brain):
        new_brain = NeuralNetwork(weights=brain.get_weights())
        new_brain.mutate()
        return new_brain
    
    def crossover(self, brain1, brain2):
        """Crossover uniforme con mutazione"""
        w1 = brain1.get_weights()
        w2 = brain2.get_weights()
        new_weights = []
        
        # Crossover uniforme per ogni layer
        for a, b in zip(w1, w2):
            if random.random() < CROSSOVER_RATE:
                mask = np.random.random(a.shape) < 0.5
                new_w = np.where(mask, a, b)
            else:
                new_w = a.copy() if random.random() < 0.5 else b.copy()
            new_weights.append(new_w)
        
        child = NeuralNetwork(weights=new_weights)
        child.mutate()
        return child
    
    def tournament_selection(self, population, k=3):
        """Selezione per torneo: scegli k individui random e prendi il migliore"""
        tournament = random.sample(population, k)
        return max(tournament, key=lambda x: x.fitness)
    
    def evolve(self):
        """Evoluzione con elitismo, torneo e crossover"""
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        
        # Statistiche generazione
        best_fitness = self.population[0].fitness
        avg_fitness = np.mean([cat.fitness for cat in self.population])
        self.best_fitness_history.append(best_fitness)
        self.avg_fitness_history.append(avg_fitness)
        
        # Tracking record assoluto
        if best_fitness > self.best_fitness_ever:
            self.best_fitness_ever = best_fitness
            self.best_brain = NeuralNetwork(weights=self.population[0].brain.get_weights())
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1
        
        # Salva top brains per modalità watch
        self.top_brains = []
        for i in range(min(MAX_WATCH_CATS, len(self.population))):
            self.top_brains.append(NeuralNetwork(weights=self.population[i].brain.get_weights()))
        
        # Elitismo: mantieni i migliori
        elite_count = max(3, int(POPULATION_SIZE * ELITE_RATIO))
        elite_cats = self.population[:elite_count]
        elite_brains = [cat.brain for cat in elite_cats]
        new_population = []
        
        # Copia elite diretta
        for brain in elite_brains[:2]:
            new_population.append(NeuralNetwork(weights=brain.get_weights()))
        
        # Genera nuova popolazione tramite crossover e selezione per torneo
        while len(new_population) < POPULATION_SIZE:
            parent1 = self.tournament_selection(self.population)
            parent2 = self.tournament_selection(self.population)
            child = self.crossover(parent1.brain, parent2.brain)
            new_population.append(child)
        
        self.generation += 1
        return new_population


def draw_gradient_background(screen, camera_x):
    """Background con gradiente cielo e parallasse cloud"""
    # Gradiente cielo
    for y in range(0, GROUND_Y):
        ratio = y / GROUND_Y
        r = int(SKY_TOP[0] + (SKY_BOTTOM[0] - SKY_TOP[0]) * ratio)
        g = int(SKY_TOP[1] + (SKY_BOTTOM[1] - SKY_TOP[1]) * ratio)
        b = int(SKY_TOP[2] + (SKY_BOTTOM[2] - SKY_TOP[2]) * ratio)
        pygame.draw.line(screen, (r,g,b), (0, y), (WIDTH, y))
    
    # Sole
    pygame.draw.circle(screen, (255, 255, 200), (WIDTH - 150, 120), 60)
    pygame.draw.circle(screen, (255, 250, 150), (WIDTH - 150, 120), 50)
    
    # Nuvole con parallasse (si muovono più lentamente della camera)
    cloud_offset = -(camera_x * 0.12) % (WIDTH + 500) - 250
    for i in range(6):
        x = (i * 350 + int(cloud_offset)) % (WIDTH + 500) - 250
        y = 80 + i * 30
        size = 40 + (i % 3) * 10
        # Nuvole più soffici con più cerchi
        pygame.draw.circle(screen, WHITE, (x, y), size)
        pygame.draw.circle(screen, WHITE, (x+35, y-8), size-5)
        pygame.draw.circle(screen, WHITE, (x+70, y), size)
        pygame.draw.circle(screen, WHITE, (x+35, y+10), size-8)
    
    # Terreno con gradiente
    grass_height = HEIGHT - GROUND_Y
    for y in range(GROUND_Y, HEIGHT):
        ratio = (y - GROUND_Y) / grass_height
        r = int(GRASS_LIGHT[0] + (GRASS_DARK[0] - GRASS_LIGHT[0]) * ratio)
        g = int(GRASS_LIGHT[1] + (GRASS_DARK[1] - GRASS_LIGHT[1]) * ratio)
        b = int(GRASS_LIGHT[2] + (GRASS_DARK[2] - GRASS_LIGHT[2]) * ratio)
        pygame.draw.line(screen, (r,g,b), (0, y), (WIDTH, y))
    
    # Linea terreno principale
    pygame.draw.line(screen, (40, 120, 40), (0, GROUND_Y), (WIDTH, GROUND_Y), 4)
    
    # Fili d'erba animati
    stripe_offset = -(camera_x * 0.7) % 100
    for i in range(-3, WIDTH//100 + 5):
        x = i * 100 + int(stripe_offset)
        h = 18 + (i % 4) * 6
        sway = math.sin(pygame.time.get_ticks() * 0.001 + i) * 3
        pygame.draw.line(screen, (30, 110, 30), (x, GROUND_Y), (x-10+sway, GROUND_Y+h), 3)


def draw_cat_improved(screen, cat, camera_x, alpha=255):
    """Disegna gatto con grafica migliorata e animazioni"""
    if not cat:
        return
    
    body_pos = cat.body.position
    half_w = cat.body_width/2
    half_h = cat.body_height/2
    
    cx = int(body_pos.x - camera_x)
    cy = int(body_pos.y)
    
    # Variazione colore basata su hue individuale
    hue_shift = cat.color_hue * 30
    cat_color = (
        max(0, min(255, ORANGE_CAT[0] + hue_shift)),
        max(0, min(255, ORANGE_CAT[1])),
        max(0, min(255, ORANGE_CAT[2] - hue_shift//2))
    )
    dark_color = (
        max(0, min(255, ORANGE_DARK[0] + hue_shift)),
        max(0, min(255, ORANGE_DARK[1])),
        max(0, min(255, ORANGE_DARK[2] - hue_shift//2))
    )
    
    # === CODA ONDEGGIANTE ===
    tail_start_x = cx - half_w + 10
    tail_start_y = cy - 5
    tail_mid_x = int((cat.tail.position.x - camera_x) * 0.5 + tail_start_x * 0.5)
    tail_mid_y = int((cat.tail.position.y) * 0.5 + tail_start_y * 0.5) - 12
    tail_end_x = int(cat.tail.position.x - camera_x)
    tail_end_y = int(cat.tail.position.y)
    
    # Coda con curve smooth
    pygame.draw.line(screen, dark_color, (tail_start_x, tail_start_y), (tail_mid_x, tail_mid_y), 12)
    pygame.draw.line(screen, dark_color, (tail_mid_x, tail_mid_y), (tail_end_x, tail_end_y), 9)
    pygame.draw.line(screen, BLACK, (tail_start_x, tail_start_y), (tail_mid_x, tail_mid_y), 3)
    pygame.draw.line(screen, BLACK, (tail_mid_x, tail_mid_y), (tail_end_x, tail_end_y), 3)
    pygame.draw.circle(screen, dark_color, (tail_end_x, tail_end_y), 5)
    
    # === GAMBE CON MUSCOLI VISIBILI ===
    for i, leg in enumerate(cat.legs):
        attach_x = cx + (leg['thigh'].position.x - body_pos.x)
        attach_y = cy + half_h
        
        knee_x = int(leg['calf'].position.x - camera_x)
        knee_y = int(leg['calf'].position.y)
        
        angle_calf = leg['calf'].angle
        foot_x = int(knee_x + math.sin(angle_calf) * cat.leg_height * 0.9)
        foot_y = int(knee_y + math.cos(angle_calf) * cat.leg_height * 0.9)
        
        # Coscia con highlight muscolare
        pygame.draw.line(screen, dark_color, (attach_x, attach_y), (knee_x, knee_y), 16)
        pygame.draw.line(screen, ORANGE_LIGHT, (attach_x-2, attach_y), (knee_x-2, knee_y), 5)
        pygame.draw.line(screen, BLACK, (attach_x, attach_y), (knee_x, knee_y), 3)
        
        # Polpaccio
        pygame.draw.line(screen, cat_color, (knee_x, knee_y), (foot_x, foot_y), 13)
        pygame.draw.line(screen, ORANGE_LIGHT, (knee_x-1, knee_y), (foot_x-1, foot_y), 4)
        pygame.draw.line(screen, BLACK, (knee_x, knee_y), (foot_x, foot_y), 3)
        
        # Zampa/piede
        pygame.draw.circle(screen, CREAM, (foot_x, foot_y), 8)
        pygame.draw.circle(screen, BLACK, (foot_x, foot_y), 8, 2)
        # Mini "cuscinetti"
        pygame.draw.circle(screen, PINK, (foot_x-2, foot_y+2), 3)
        pygame.draw.circle(screen, PINK, (foot_x+3, foot_y+2), 3)
    
    # === CORPO PRINCIPALE ===
    body_rect = (cx - half_w, cy - half_h, cat.body_width, cat.body_height)
    pygame.draw.ellipse(screen, cat_color, body_rect)
    
    # Strisce tigrate
    stripe_positions = [-35, -18, -2, 15, 30]
    for offset in stripe_positions:
        stripe_x = cx + offset
        if abs(offset) < half_w - 10:
            pygame.draw.line(screen, dark_color, 
                           (stripe_x, cy - half_h + 8), (stripe_x, cy + half_h - 8), 6)
    
    pygame.draw.ellipse(screen, BLACK, body_rect, 4)
    
    # Pancia crema
    chest_rect = (cx - half_w + 12, cy + 6, cat.body_width - 24, half_h - 4)
    pygame.draw.ellipse(screen, CREAM, chest_rect)
    pygame.draw.ellipse(screen, (200, 180, 160), chest_rect, 2)
    
    # === TESTA ===
    head_x = cx + half_w + 15
    head_y = cy - 12
    
    # Collo
    pygame.draw.ellipse(screen, cat_color, (cx + half_w - 10, cy - 14, 28, 26))
    pygame.draw.ellipse(screen, BLACK, (cx + half_w - 10, cy - 14, 28, 26), 3)
    
    # Testa principale
    pygame.draw.circle(screen, cat_color, (head_x, head_y), 28)
    pygame.draw.circle(screen, BLACK, (head_x, head_y), 28, 4)
    
    # Muso
    pygame.draw.ellipse(screen, CREAM, (head_x - 18, head_y + 6, 36, 22))
    pygame.draw.ellipse(screen, BLACK, (head_x - 18, head_y + 6, 36, 22), 2)
    
    # === ORECCHIE ===
    ear_left = [(head_x - 20, head_y - 20), (head_x - 12, head_y - 42), (head_x, head_y - 24)]
    ear_right = [(head_x + 5, head_y - 24), (head_x + 17, head_y - 42), (head_x + 25, head_y - 20)]
    
    pygame.draw.polygon(screen, cat_color, ear_left)
    pygame.draw.polygon(screen, cat_color, ear_right)
    pygame.draw.polygon(screen, BLACK, ear_left, 3)
    pygame.draw.polygon(screen, BLACK, ear_right, 3)
    
    # Interno orecchie (rosa)
    pygame.draw.polygon(screen, PINK, [(head_x-16, head_y-24), (head_x-12, head_y-34), (head_x-8, head_y-24)])
    pygame.draw.polygon(screen, PINK, [(head_x+9, head_y-24), (head_x+13, head_y-34), (head_x+17, head_y-24)])
    
    # === OCCHI ===
    eye_y = head_y - 8
    if cat.alive and cat.steps < SIMULATION_STEPS - 50:
        # Occhi aperti con pupille
        pygame.draw.ellipse(screen, WHITE, (head_x - 18, eye_y - 9, 14, 18))
        pygame.draw.ellipse(screen, WHITE, (head_x + 4, eye_y - 9, 14, 18))
        pygame.draw.ellipse(screen, BLACK, (head_x - 18, eye_y - 9, 14, 18), 2)
        pygame.draw.ellipse(screen, BLACK, (head_x + 4, eye_y - 9, 14, 18), 2)
        
        # Pupille verticali (tipiche dei gatti)
        pygame.draw.ellipse(screen, BLACK, (head_x - 14, eye_y - 2, 6, 14))
        pygame.draw.ellipse(screen, BLACK, (head_x + 8, eye_y - 2, 6, 14))
        
        # Riflessi occhi
        pygame.draw.circle(screen, WHITE, (head_x - 10, eye_y + 1), 2)
        pygame.draw.circle(screen, WHITE, (head_x + 12, eye_y + 1), 2)
    else:
        # Occhi chiusi (gatto esausto o morto)
        pygame.draw.line(screen, BLACK, (head_x - 18, eye_y), (head_x - 4, eye_y), 3)
        pygame.draw.line(screen, BLACK, (head_x + 4, eye_y), (head_x + 18, eye_y), 3)
    
    # === NASO E BOCCA ===
    pygame.draw.circle(screen, PINK, (head_x, head_y + 12), 6)
    pygame.draw.circle(screen, (200, 100, 120), (head_x, head_y + 12), 6, 2)
    pygame.draw.line(screen, BLACK, (head_x, head_y + 18), (head_x, head_y + 22), 2)
    
    # Bocca (sorriso felino)
    pygame.draw.arc(screen, BLACK, (head_x - 8, head_y + 18, 16, 10), math.pi, 2*math.pi, 2)
    
    # Baffi
    whisker_base_y = head_y + 10
    for i, angle in enumerate([-0.3, -0.1, 0.1, 0.3]):
        wx_start = head_x - 18 if i < 2 else head_x + 18
        wx_end = wx_start + math.cos(angle) * 35 * (-1 if i < 2 else 1)
        wy_end = whisker_base_y + math.sin(angle) * 15
        pygame.draw.line(screen, GRAY, (wx_start, whisker_base_y), (int(wx_end), int(wy_end)), 1)


def draw_text_with_shadow(screen, text, pos, size=24, color=BLACK, shadow_color=WHITE):
    """Testo con ombra per leggibilità"""
    font = pygame.font.SysFont("Arial", size, bold=True)
    shadow = font.render(text, True, shadow_color)
    surf = font.render(text, True, color)
    screen.blit(shadow, (pos[0]+2, pos[1]+2))
    screen.blit(surf, pos)


def draw_fitness_graph(screen, history, rect, title, color=(0, 200, 0)):
    """Grafico fitness in tempo reale"""
    if len(history) < 2:
        return
    
    # Background solido senza alpha
    pygame.draw.rect(screen, (240, 240, 240), rect)
    pygame.draw.rect(screen, BLACK, rect, 2)
    
    # Titolo
    font = pygame.font.SysFont("Arial", 14, bold=True)
    title_surf = font.render(title, True, BLACK)
    screen.blit(title_surf, (rect.x + 5, rect.y + 5))
    
    # Calcola range valori
    history_window = history[-HISTORY_LENGTH:]
    max_val = max(history_window) if max(history_window) > 0 else 1
    min_val = min(history_window)
    range_val = max_val - min_val if max_val != min_val else 1
    
    # Disegna linea grafico
    points = []
    for i, val in enumerate(history_window):
        x = rect.x + 10 + (i / len(history_window)) * (rect.width - 20)
        normalized = (val - min_val) / range_val
        y = rect.bottom - 10 - normalized * (rect.height - 30)
        points.append((int(x), int(y)))
    
    if len(points) > 1:
        pygame.draw.lines(screen, color, False, points, 3)
    
    # Valori min/max
    font_small = pygame.font.SysFont("Arial", 11)
    max_text = font_small.render(f"{max_val:.1f}", True, color)
    min_text = font_small.render(f"{min_val:.1f}", True, color)
    screen.blit(max_text, (rect.right - 50, rect.y + 20))
    screen.blit(min_text, (rect.right - 50, rect.bottom - 20))


def draw_slider(screen, value, max_val, rect, dragging):
    """Slider per selezionare numero di gatti da visualizzare"""
    pygame.draw.rect(screen, (220, 220, 220), rect, border_radius=5)
    pygame.draw.rect(screen, BLACK, rect, 2, border_radius=5)
    
    ratio = (value - 1) / (max_val - 1)
    handle_x = rect.x + ratio * rect.width
    handle_y = rect.centery
    
    # Traccia
    pygame.draw.line(screen, (120, 120, 120), (rect.x + 10, handle_y), (rect.right - 10, handle_y), 6)
    
    # Handle
    handle_radius = 14
    color = (80, 200, 80) if dragging else (150, 150, 150)
    pygame.draw.circle(screen, color, (int(handle_x), handle_y), handle_radius)
    pygame.draw.circle(screen, BLACK, (int(handle_x), handle_y), handle_radius, 3)
    pygame.draw.circle(screen, WHITE, (int(handle_x)-3, handle_y-3), 4)
    
    # Label
    font = pygame.font.SysFont("Arial", 18, bold=True)
    text = font.render(f"Gatti mostrati: {value}", True, BLACK)
    screen.blit(text, (rect.x, rect.y - 28))


def clear_space(space):
    """Rimuove tutti i corpi dinamici dallo spazio fisico"""
    for constraint in list(space.constraints):
        space.remove(constraint)
    for body in list(space.bodies):
        if body.body_type == pymunk.Body.DYNAMIC:
            for shape in list(body.shapes):
                space.remove(shape)
            space.remove(body)


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("🐱 Cat Evolution Simulator - Claude Edition | SPACE:Watch R:Restart C:Clear")
    clock = pygame.time.Clock()
    
    # Setup fisica Pymunk
    space = pymunk.Space()
    space.gravity = (0, 980)
    space.damping = 0.92
    
    # Ground statico
    static_body = space.static_body
    ground = pymunk.Segment(static_body, (-2000, GROUND_Y), (10000, GROUND_Y), 5)
    ground.friction = 1.2
    space.add(ground)
    
    # Stato evoluzione
    evo = Evolution()
    new_brains = [NeuralNetwork() for _ in range(POPULATION_SIZE)]
    current_cat_index = 0
    is_seeded_run = False
    
    # Modalità visualizzazione
    mode = "FAST"  # FAST = allenamento veloce, WATCH = osserva top performers
    watch_cats = []
    camera_x = 0
    
    # UI slider
    watch_cat_count = 5
    slider_rect = pygame.Rect(WIDTH - 250, HEIGHT - 90, 200, 35)
    slider_dragging = False
    
    # Grafici fitness
    graph_best_rect = pygame.Rect(10, HEIGHT - 180, 280, 120)
    graph_avg_rect = pygame.Rect(300, HEIGHT - 180, 280, 120)
    
    running = True
    while running:
        mouse_pos = pygame.mouse.get_pos()
        
        # === EVENT HANDLING ===
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            # Slider interaction
            if event.type == pygame.MOUSEBUTTONDOWN:
                if slider_rect.collidepoint(mouse_pos) and mode == "WATCH":
                    slider_dragging = True
            
            if event.type == pygame.MOUSEBUTTONUP:
                slider_dragging = False
            
            if event.type == pygame.MOUSEMOTION and slider_dragging:
                rel_x = max(0, min(mouse_pos[0] - slider_rect.x, slider_rect.width))
                ratio = rel_x / slider_rect.width
                new_count = int(1 + ratio * (MAX_WATCH_CATS - 1))
                if new_count != watch_cat_count:
                    watch_cat_count = new_count
                    watch_cats = []
                    clear_space(space)
            
            # Keyboard controls
            if event.type == pygame.KEYDOWN:
                # SPACE: Toggle tra modalità FAST e WATCH
                if event.key == pygame.K_SPACE:
                    if mode == "FAST" and len(evo.top_brains) > 0:
                        mode = "WATCH"
                        watch_cats = []
                    else:
                        mode = "FAST"
                        watch_cats = []
                        clear_space(space)
                
                # R: Restart con seeding dal miglior cervello
                if event.key == pygame.K_r:
                    if mode == "WATCH" and watch_cats:
                        watch_cats = []
                        clear_space(space)
                    else:
                        if evo.best_brain:
                            is_seeded_run = True
                            new_brains = []
                            for _ in range(POPULATION_SIZE):
                                seeded = NeuralNetwork(weights=evo.best_brain.get_weights())
                                seeded.mutate(rate=SEED_MUTATION_RATE, strength=SEED_MUTATION_STRENGTH)
                                new_brains.append(seeded)
                        else:
                            is_seeded_run = False
                            new_brains = [NeuralNetwork() for _ in range(POPULATION_SIZE)]
                        
                        # Preserva record e top brains
                        old_best = evo.best_brain
                        old_record = evo.best_fitness_ever
                        old_top = evo.top_brains
                        evo = Evolution()
                        evo.best_brain = old_best
                        evo.best_fitness_ever = old_record
                        evo.top_brains = old_top
                        current_cat_index = 0
                        mode = "FAST"
                        watch_cats = []
                        camera_x = 0
                        clear_space(space)
                
                # C: Clear completo (reset totale)
                if event.key == pygame.K_c:
                    evo = Evolution()
                    new_brains = [NeuralNetwork() for _ in range(POPULATION_SIZE)]
                    current_cat_index = 0
                    is_seeded_run = False
                    mode = "FAST"
                    watch_cats = []
                    watch_cat_count = 5
                    camera_x = 0
                    clear_space(space)
        
        # === SIMULATION LOGIC ===
        if mode == "FAST":
            # Modalità allenamento veloce: simula un gatto alla volta
            if current_cat_index < POPULATION_SIZE:
                cat = Cat(space, 200, new_brains[current_cat_index], color_hue=current_cat_index % 3)
                for step in range(SIMULATION_STEPS):
                    if not cat.update():
                        break
                    space.step(1/FPS)
                
                evo.population.append(cat)
                clear_space(space)
                current_cat_index += 1
            else:
                # Generazione completa: evolvi popolazione
                new_brains = evo.evolve()
                current_cat_index = 0
                is_seeded_run = False
                evo.population = []
        else:
            # Modalità WATCH: mostra i top performer in tempo reale
            if len(watch_cats) == 0 and len(evo.top_brains) > 0:
                count = min(watch_cat_count, len(evo.top_brains))
                for i in range(count):
                    cat = Cat(space, 200, evo.top_brains[i], color_hue=i)
                    watch_cats.append(cat)
            
            # Aggiorna fisica per tutti i gatti vivi
            any_alive = False
            for cat in watch_cats:
                if cat.alive and cat.steps < SIMULATION_STEPS:
                    cat.update()
                    any_alive = True
            
            if any_alive:
                space.step(1/FPS)
            
            # Camera smooth follow del leader
            if watch_cats:
                leader = max(watch_cats, key=lambda c: c.body.position.x)
                target_cam = max(0, leader.body.position.x - WIDTH/2.5)
                camera_x += (target_cam - camera_x) * 0.08  # Smooth lerp
        
        # === RENDERING ===
        draw_gradient_background(screen, camera_x)
        
        if mode == "WATCH" and watch_cats:
            # Disegna tutti i gatti con grafica migliorata
            for cat in watch_cats:
                draw_cat_improved(screen, cat, camera_x)
            
            # Info leader e distanze
            leader = max(watch_cats, key=lambda c: c.body.position.x)
            leader_dist = leader.body.position.x - 200
            
            alive_count = sum(1 for cat in watch_cats if cat.alive)
            color = (0, 160, 0) if alive_count > 0 else (200, 0, 0)
            
            font = pygame.font.SysFont("Arial", 26, bold=True)
            text_surf = font.render(f"WATCH MODE - Top {len(watch_cats)} Gatti | Gen {evo.generation}", True, color)
            screen.blit(text_surf, (10, 10))
            
            font2 = pygame.font.SysFont("Arial", 20, bold=True)
            text_surf2 = font2.render(f"Vivi: {alive_count}/{len(watch_cats)} | Leader: {leader_dist:.1f}m | Record: {evo.best_fitness_ever:.1f}m", True, BLACK)
            screen.blit(text_surf2, (10, 45))
            
            # Slider UI
            draw_slider(screen, watch_cat_count, MAX_WATCH_CATS, slider_rect, slider_dragging)
            
        else:
            # Modalità FAST: mostra progresso allenamento
            color = (0, 100, 220) if is_seeded_run else (40, 40, 40)
            mode_text = "SEEDED RUN" if is_seeded_run else "RANDOM RUN"
            
            font = pygame.font.SysFont("Arial", 26, bold=True)
            text_surf = font.render(f"{mode_text} | Gen {evo.generation} | {current_cat_index}/{POPULATION_SIZE}", True, color)
            screen.blit(text_surf, (10, 10))
            
            if evo.best_fitness_history:
                last_best = evo.best_fitness_history[-1]
                avg_last = evo.avg_fitness_history[-1] if evo.avg_fitness_history else 0
                font2 = pygame.font.SysFont("Arial", 20, bold=True)
                text_surf2 = font2.render(f"Best Gen: {last_best:.1f}m | Avg: {avg_last:.1f}m | Record: {evo.best_fitness_ever:.1f}m", True, BLACK)
                screen.blit(text_surf2, (10, 45))
                
                # Indicatore stagnazione
                if evo.stagnation_counter > 10:
                    stag_color = (200, 100, 0)
                    font3 = pygame.font.SysFont("Arial", 18, bold=True)
                    text_surf3 = font3.render(f"Stagnazione: {evo.stagnation_counter} gen", True, stag_color)
                    screen.blit(text_surf3, (10, 75))
            
            # Grafici fitness
            if len(evo.best_fitness_history) > 2:
                draw_fitness_graph(screen, evo.best_fitness_history, graph_best_rect, 
                                  "Best Fitness", color=(0, 200, 0))
                draw_fitness_graph(screen, evo.avg_fitness_history, graph_avg_rect, 
                                  "Avg Fitness", color=(200, 100, 0))
        
        # Footer con controlli
        font_footer = pygame.font.SysFont("Arial", 18, bold=True)
        text_footer = font_footer.render("SPACE:Watch | R:Restart | C:Clear All", True, (60, 60, 60))
        screen.blit(text_footer, (10, HEIGHT-25))
        
        pygame.display.flip()
        clock.tick(FPS)
    
    pygame.quit()


if __name__ == "__main__":
    main()
