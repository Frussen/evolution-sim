import pygame
import pymunk
import numpy as np
import random
import math

# Configurazione
WIDTH, HEIGHT = 1200, 600
FPS = 60
POPULATION_SIZE = 30
SIMULATION_STEPS = 700
MUTATION_RATE = 0.1
MUTATION_STRENGTH = 0.5
ELITE_RATIO = 0.2
SEED_MUTATION_RATE = 0.3
SEED_MUTATION_STRENGTH = 0.8

GROUND_Y = 400

# Colori
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
SKY_TOP = (135, 206, 235)
SKY_BOTTOM = (240, 248, 255)
GRASS_LIGHT = (124, 252, 0)
GRASS_DARK = (34, 139, 34)
ORANGE_CAT = (255, 140, 0)
ORANGE_DARK = (180, 90, 0)
CREAM = (255, 235, 200)
PINK = (255, 182, 193)

MAX_WATCH_CATS = 10


class NeuralNetwork:
    def __init__(self, input_size=20, hidden_size=12, output_size=8, weights=None):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        if weights is None:
            self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0/input_size)
            self.b1 = np.zeros(hidden_size)
            self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0/hidden_size)
            self.b2 = np.zeros(output_size)
        else:
            self.set_weights(weights)
    
    def forward(self, x):
        x = np.array(x, dtype=np.float32)
        h = np.tanh(np.dot(x, self.W1) + self.b1)
        o = np.tanh(np.dot(h, self.W2) + self.b2)
        return o
    
    def get_weights(self):
        return [self.W1.copy(), self.b1.copy(), self.W2.copy(), self.b2.copy()]
    
    def set_weights(self, weights):
        self.W1, self.b1, self.W2, self.b2 = weights
    
    def mutate(self, rate=MUTATION_RATE, strength=MUTATION_STRENGTH):
        for W in [self.W1, self.W2]:
            mask = np.random.random(W.shape) < rate
            W += mask * np.random.randn(*W.shape) * strength
        for b in [self.b1, self.b2]:
            mask = np.random.random(b.shape) < rate
            b += mask * np.random.randn(*b.shape) * strength


class Cat:
    def __init__(self, space, x_start, brain=None):
        self.space = space
        self.start_x = x_start
        self.fitness = 0
        self.alive = True
        self.steps = 0
        self.brain = brain if brain else NeuralNetwork()
        
        self.body_width, self.body_height = 80, 32
        self.leg_width, self.leg_height = 12, 30
        
        mass = 7
        moment = pymunk.moment_for_box(mass, (self.body_width, self.body_height))
        self.body = pymunk.Body(mass, moment)
        self.body.position = (x_start, 330)
        
        self.body_shape = pymunk.Poly.create_box(self.body, (self.body_width, self.body_height))
        self.body_shape.friction = 0.9
        self.body_shape.filter = pymunk.ShapeFilter(group=1)
        self.space.add(self.body, self.body_shape)
        
        self.legs = []
        half_w = self.body_width/2
        half_h = self.body_height/2
        
        leg_positions = [
            (half_w - 10, half_h),
            (half_w - 10, half_h),
            (-half_w + 10, half_h),
            (-half_w + 10, half_h)
        ]
        
        for i, (offset_x, offset_y) in enumerate(leg_positions):
            is_front = i < 2
            
            thigh_mass = 2.2
            thigh_moment = pymunk.moment_for_segment(thigh_mass, (0,0), (0, self.leg_height), self.leg_width/2)
            thigh = pymunk.Body(thigh_mass, thigh_moment)
            
            thigh_x = x_start + offset_x
            thigh_y = 330 + offset_y
            thigh.position = (thigh_x, thigh_y)
            
            if is_front:
                thigh.angle = -0.15 if i == 0 else 0.15
            else:
                thigh.angle = 0.15 if i == 2 else -0.15
            
            thigh_shape = pymunk.Segment(thigh, (0, 0), (0, self.leg_height), self.leg_width/2)
            thigh_shape.friction = 1.2
            thigh_shape.filter = pymunk.ShapeFilter(group=1)
            self.space.add(thigh, thigh_shape)
            
            joint = pymunk.PivotJoint(self.body, thigh, (thigh_x, thigh_y))
            self.space.add(joint)
            
            if is_front:
                limit = pymunk.RotaryLimitJoint(self.body, thigh, -math.pi/4, math.pi/4)
            else:
                limit = pymunk.RotaryLimitJoint(self.body, thigh, -math.pi/3, math.pi/3)
            self.space.add(limit)
            
            motor_thigh = pymunk.SimpleMotor(self.body, thigh, 0)
            motor_thigh.max_force = 3000000
            self.space.add(motor_thigh)
            
            calf_mass = 1.5
            calf_len = self.leg_height * 0.85
            calf_moment = pymunk.moment_for_segment(calf_mass, (0,0), (0, calf_len), self.leg_width/2)
            calf = pymunk.Body(calf_mass, calf_moment)
            
            calf_x = thigh_x + math.sin(thigh.angle) * self.leg_height * 0.2
            calf_y = thigh_y + math.cos(thigh.angle) * self.leg_height
            calf.position = (calf_x, calf_y)
            calf.angle = thigh.angle
            
            calf_shape = pymunk.Segment(calf, (0, 0), (0, calf_len), self.leg_width/2)
            calf_shape.friction = 1.5
            calf_shape.filter = pymunk.ShapeFilter(group=1)
            self.space.add(calf, calf_shape)
            
            knee_joint = pymunk.PivotJoint(thigh, calf, (calf_x, calf_y))
            self.space.add(knee_joint)
            knee_limit = pymunk.RotaryLimitJoint(thigh, calf, -math.pi/2.2, 0.15)
            self.space.add(knee_limit)
            
            motor_calf = pymunk.SimpleMotor(thigh, calf, 0)
            motor_calf.max_force = 2000000
            self.space.add(motor_calf)
            
            self.legs.append({
                'thigh': thigh, 'calf': calf,
                'thigh_motor': motor_thigh, 'calf_motor': motor_calf,
                'is_front': is_front
            })
        
        tail_mass = 1.0
        tail_moment = pymunk.moment_for_segment(tail_mass, (0,0), (-35, -15), 5)
        self.tail = pymunk.Body(tail_mass, tail_moment)
        tail_x = x_start - half_w
        tail_y = 330 - 5
        self.tail.position = (tail_x, tail_y)
        tail_shape = pymunk.Segment(self.tail, (0, 0), (-35, -15), 5)
        tail_shape.friction = 0.3
        tail_shape.filter = pymunk.ShapeFilter(group=1)
        self.space.add(self.tail, tail_shape)
        
        tail_joint = pymunk.PivotJoint(self.body, self.tail, (tail_x, tail_y))
        self.space.add(tail_joint)
        tail_limit = pymunk.RotaryLimitJoint(self.body, self.tail, -math.pi/3, math.pi/3)
        self.space.add(tail_limit)
        
        for leg in self.legs:
            leg['thigh_motor'].rate = 0
            leg['calf_motor'].rate = 0
        
        for _ in range(15):
            space.step(1/FPS)
    
    def get_nn_input(self):
        inputs = []
        angle_body = self.body.angle % (2*math.pi)
        if angle_body > math.pi:
            angle_body -= 2*math.pi
        inputs.append(angle_body / math.pi)
        
        vx, vy = self.body.velocity
        inputs.append(vx / 100.0)
        inputs.append(vy / 100.0)
        inputs.append((GROUND_Y - self.body.position.y) / 200.0)
        
        for leg in self.legs:
            thigh_angle = (leg['thigh'].angle - self.body.angle) % (2*math.pi)
            if thigh_angle > math.pi:
                thigh_angle -= 2*math.pi
            inputs.append(thigh_angle / math.pi)
            
            calf_angle = (leg['calf'].angle - leg['thigh'].angle) % (2*math.pi)
            if calf_angle > math.pi:
                calf_angle -= 2*math.pi
            inputs.append(calf_angle / math.pi)
            
            inputs.append(leg['thigh'].angular_velocity / 10.0)
            inputs.append(leg['calf'].angular_velocity / 10.0)
        
        return inputs
    
    def act(self):
        inputs = self.get_nn_input()
        outputs = self.brain.forward(inputs)
        
        for i, leg in enumerate(self.legs):
            thigh_speed = outputs[i*2] * 3.5
            calf_speed = outputs[i*2 + 1] * 3.5
            leg['thigh_motor'].rate = thigh_speed
            leg['calf_motor'].rate = calf_speed
    
    def update(self):
        self.act()
        self.steps += 1
        current_x = self.body.position.x
        self.fitness = current_x - self.start_x
        
        if self.steps > 80:
            if self.body.position.y > GROUND_Y + 30 or abs(self.body.angle) > 1.8:
                self.alive = False
        return self.alive


class Evolution:
    def __init__(self):
        self.population = []
        self.generation = 1
        self.best_fitness_history = []
        self.best_brain = None
        self.best_fitness_ever = -float('inf')
        self.top_brains = []
        
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
        w1 = brain1.get_weights()
        w2 = brain2.get_weights()
        new_weights = []
        for a, b in zip(w1, w2):
            mask = np.random.random(a.shape) < 0.5
            new_w = np.where(mask, a, b)
            new_weights.append(new_w)
        child = NeuralNetwork(weights=new_weights)
        child.mutate()
        return child
    
    def evolve(self):
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        best_fitness = self.population[0].fitness
        self.best_fitness_history.append(best_fitness)
        
        if best_fitness > self.best_fitness_ever:
            self.best_fitness_ever = best_fitness
            self.best_brain = NeuralNetwork(weights=self.population[0].brain.get_weights())
        
        self.top_brains = []
        for i in range(min(MAX_WATCH_CATS, len(self.population))):
            self.top_brains.append(NeuralNetwork(weights=self.population[i].brain.get_weights()))
        
        elite_count = max(2, int(POPULATION_SIZE * ELITE_RATIO))
        elite_cats = self.population[:elite_count]
        elite_brains = [cat.brain for cat in elite_cats]
        new_population = elite_brains.copy()
        
        while len(new_population) < POPULATION_SIZE:
            parent1 = random.choice(elite_brains)
            parent2 = random.choice(elite_brains)
            child = self.crossover(parent1, parent2)
            new_population.append(child)
        
        self.generation += 1
        return new_population


def draw_background(screen, camera_x):
    for y in range(0, GROUND_Y):
        ratio = y / GROUND_Y
        r = int(SKY_TOP[0] + (SKY_BOTTOM[0] - SKY_TOP[0]) * ratio)
        g = int(SKY_TOP[1] + (SKY_BOTTOM[1] - SKY_TOP[1]) * ratio)
        b = int(SKY_TOP[2] + (SKY_BOTTOM[2] - SKY_TOP[2]) * ratio)
        pygame.draw.line(screen, (r,g,b), (0, y), (WIDTH, y))
    
    cloud_offset = -(camera_x * 0.15) % (WIDTH + 400) - 200
    for i in range(5):
        x = (i * 300 + int(cloud_offset)) % (WIDTH + 400) - 200
        y = 60 + i * 25
        pygame.draw.circle(screen, WHITE, (x, y), 35)
        pygame.draw.circle(screen, WHITE, (x+30, y-12), 40)
        pygame.draw.circle(screen, WHITE, (x+60, y), 35)
    
    grass_height = HEIGHT - GROUND_Y
    for y in range(GROUND_Y, HEIGHT):
        ratio = (y - GROUND_Y) / grass_height
        r = int(GRASS_LIGHT[0] + (GRASS_DARK[0] - GRASS_LIGHT[0]) * ratio)
        g = int(GRASS_LIGHT[1] + (GRASS_DARK[1] - GRASS_LIGHT[1]) * ratio)
        b = int(GRASS_LIGHT[2] + (GRASS_DARK[2] - GRASS_LIGHT[2]) * ratio)
        pygame.draw.line(screen, (r,g,b), (0, y), (WIDTH, y))
    
    pygame.draw.line(screen, (30, 100, 30), (0, GROUND_Y), (WIDTH, GROUND_Y), 3)
    
    stripe_offset = -(camera_x * 0.8) % 80
    for i in range(-2, WIDTH//80 + 3):
        x = i * 80 + int(stripe_offset)
        h = 15 + (i % 3) * 5
        pygame.draw.line(screen, (20, 90, 20), (x, GROUND_Y), (x-8, GROUND_Y+h), 2)


def draw_cat(screen, cat, camera_x):
    """Disegna un gatto con colori standard (tutti uguali)"""
    if not cat:
        return
    
    body_pos = cat.body.position
    half_w = cat.body_width/2
    half_h = cat.body_height/2
    
    cx = int(body_pos.x - camera_x)
    cy = int(body_pos.y)
    
    # Coda
    tail_start_x = cx - half_w + 8
    tail_start_y = cy - 2
    tail_mid_x = int((cat.tail.position.x - camera_x) * 0.6 + tail_start_x * 0.4)
    tail_mid_y = int((cat.tail.position.y) * 0.6 + tail_start_y * 0.4) - 8
    tail_end_x = int(cat.tail.position.x - camera_x)
    tail_end_y = int(cat.tail.position.y)
    
    pygame.draw.line(screen, ORANGE_DARK, (tail_start_x, tail_start_y), (tail_mid_x, tail_mid_y), 10)
    pygame.draw.line(screen, ORANGE_DARK, (tail_mid_x, tail_mid_y), (tail_end_x, tail_end_y), 8)
    pygame.draw.line(screen, BLACK, (tail_start_x, tail_start_y), (tail_mid_x, tail_mid_y), 3)
    pygame.draw.line(screen, BLACK, (tail_mid_x, tail_mid_y), (tail_end_x, tail_end_y), 3)
    
    # Gambe
    for i, leg in enumerate(cat.legs):
        attach_x = cx + (leg['thigh'].position.x - body_pos.x)
        attach_y = cy + half_h
        
        knee_x = int(leg['calf'].position.x - camera_x)
        knee_y = int(leg['calf'].position.y)
        
        angle_calf = leg['calf'].angle
        foot_x = int(knee_x + math.sin(angle_calf) * cat.leg_height * 0.85)
        foot_y = int(knee_y + math.cos(angle_calf) * cat.leg_height * 0.85)
        
        pygame.draw.line(screen, ORANGE_DARK, (attach_x, attach_y), (knee_x, knee_y), 14)
        pygame.draw.line(screen, BLACK, (attach_x, attach_y), (knee_x, knee_y), 3)
        
        pygame.draw.line(screen, ORANGE_CAT, (knee_x, knee_y), (foot_x, foot_y), 12)
        pygame.draw.line(screen, BLACK, (knee_x, knee_y), (foot_x, foot_y), 3)
        
        pygame.draw.circle(screen, CREAM, (foot_x, foot_y), 7)
        pygame.draw.circle(screen, BLACK, (foot_x, foot_y), 7, 2)
    
    # Corpo
    body_rect = (cx - half_w, cy - half_h, cat.body_width, cat.body_height)
    pygame.draw.ellipse(screen, ORANGE_CAT, body_rect)
    
    for offset in [-30, -10, 10, 25]:
        stripe_x = cx + offset
        if abs(offset) < half_w - 8:
            pygame.draw.line(screen, ORANGE_DARK, 
                           (stripe_x, cy - half_h + 6), (stripe_x, cy + half_h - 6), 5)
    
    pygame.draw.ellipse(screen, BLACK, body_rect, 4)
    
    chest_rect = (cx - half_w + 10, cy + 4, cat.body_width - 20, half_h - 2)
    pygame.draw.ellipse(screen, CREAM, chest_rect)
    pygame.draw.ellipse(screen, BLACK, chest_rect, 2)
    
    # Testa
    head_x = cx + half_w + 10
    head_y = cy - 10
    
    pygame.draw.ellipse(screen, ORANGE_CAT, (cx + half_w - 8, cy - 12, 24, 24))
    pygame.draw.ellipse(screen, BLACK, (cx + half_w - 8, cy - 12, 24, 24), 3)
    
    pygame.draw.circle(screen, ORANGE_CAT, (head_x, head_y), 25)
    pygame.draw.circle(screen, BLACK, (head_x, head_y), 25, 4)
    
    pygame.draw.ellipse(screen, CREAM, (head_x - 15, head_y + 5, 30, 18))
    pygame.draw.ellipse(screen, BLACK, (head_x - 15, head_y + 5, 30, 18), 2)
    
    # Orecchie
    ear_left = [(head_x - 18, head_y - 18), (head_x - 10, head_y - 38), (head_x + 2, head_y - 22)]
    ear_right = [(head_x + 5, head_y - 22), (head_x + 15, head_y - 38), (head_x + 23, head_y - 18)]
    
    pygame.draw.polygon(screen, ORANGE_CAT, ear_left)
    pygame.draw.polygon(screen, ORANGE_CAT, ear_right)
    pygame.draw.polygon(screen, BLACK, ear_left, 3)
    pygame.draw.polygon(screen, BLACK, ear_right, 3)
    
    pygame.draw.polygon(screen, PINK, [(head_x-14, head_y-22), (head_x-10, head_y-30), (head_x-6, head_y-22)])
    pygame.draw.polygon(screen, PINK, [(head_x+9, head_y-22), (head_x+13, head_y-30), (head_x+17, head_y-22)])
    
    # Occhi
    eye_y = head_y - 8
    if cat.alive and cat.steps < SIMULATION_STEPS:
        pygame.draw.ellipse(screen, WHITE, (head_x - 16, eye_y - 8, 12, 16))
        pygame.draw.ellipse(screen, WHITE, (head_x + 4, eye_y - 8, 12, 16))
        pygame.draw.ellipse(screen, BLACK, (head_x - 16, eye_y - 8, 12, 16), 2)
        pygame.draw.ellipse(screen, BLACK, (head_x + 4, eye_y - 8, 12, 16), 2)
        pygame.draw.circle(screen, BLACK, (head_x - 10, eye_y + 2), 5)
        pygame.draw.circle(screen, BLACK, (head_x + 10, eye_y + 2), 5)
        pygame.draw.circle(screen, WHITE, (head_x - 8, eye_y), 2)
        pygame.draw.circle(screen, WHITE, (head_x + 12, eye_y), 2)
    else:
        pygame.draw.line(screen, BLACK, (head_x - 16, eye_y), (head_x - 4, eye_y), 3)
        pygame.draw.line(screen, BLACK, (head_x + 4, eye_y), (head_x + 16, eye_y), 3)
    
    # Naso
    pygame.draw.circle(screen, PINK, (head_x, head_y + 10), 5)
    pygame.draw.circle(screen, BLACK, (head_x, head_y + 10), 5, 2)
    pygame.draw.line(screen, BLACK, (head_x, head_y + 15), (head_x, head_y + 20), 2)


def draw_text(screen, text, pos, size=24, color=BLACK):
    font = pygame.font.SysFont("monospace", size, bold=True)
    surf = font.render(text, True, color)
    shadow = font.render(text, True, WHITE)
    screen.blit(shadow, (pos[0]+1, pos[1]+1))
    screen.blit(surf, pos)


def draw_slider(screen, value, max_val, rect, dragging):
    """Disegna uno slider orizzontale"""
    pygame.draw.rect(screen, (200, 200, 200), rect)
    pygame.draw.rect(screen, BLACK, rect, 2)
    
    ratio = (value - 1) / (max_val - 1)
    handle_x = rect.x + ratio * rect.width
    handle_y = rect.centery
    
    pygame.draw.line(screen, (100, 100, 100), (rect.x, handle_y), (rect.right, handle_y), 4)
    
    handle_radius = 12
    color = (100, 200, 100) if dragging else (150, 150, 150)
    pygame.draw.circle(screen, color, (int(handle_x), handle_y), handle_radius)
    pygame.draw.circle(screen, BLACK, (int(handle_x), handle_y), handle_radius, 2)
    
    font = pygame.font.SysFont("monospace", 20, bold=True)
    text = font.render(f"Gatti: {value}", True, BLACK)
    screen.blit(text, (rect.x, rect.y - 25))


def clear_space(space):
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
    pygame.display.set_caption("Gatto Evoluto - SPAZIO:toggle R:newrun C:clear")
    clock = pygame.time.Clock()
    
    space = pymunk.Space()
    space.gravity = (0, 900)
    space.damping = 0.9
    
    static_body = space.static_body
    ground = pymunk.Segment(static_body, (-1000, GROUND_Y), (5000, GROUND_Y), 5)
    ground.friction = 1.0
    space.add(ground)
    
    evo = Evolution()
    new_brains = [NeuralNetwork() for _ in range(POPULATION_SIZE)]
    current_cat_index = 0
    is_seeded_run = False
    
    mode = "FAST"
    watch_cats = []
    camera_x = 0
    
    watch_cat_count = 5
    slider_rect = pygame.Rect(WIDTH - 220, HEIGHT - 80, 180, 30)
    slider_dragging = False
    
    running = True
    while running:
        mouse_pos = pygame.mouse.get_pos()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
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
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    if mode == "FAST" and len(evo.top_brains) > 0:
                        mode = "WATCH"
                        watch_cats = []
                    else:
                        mode = "FAST"
                        watch_cats = []
                        clear_space(space)
                
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
        
        if mode == "FAST":
            if current_cat_index < POPULATION_SIZE:
                cat = Cat(space, 200, new_brains[current_cat_index])
                for step in range(SIMULATION_STEPS):
                    if not cat.update():
                        break
                    space.step(1/FPS)
                
                cat.fitness = cat.body.position.x - 200
                evo.population.append(cat)
                clear_space(space)
                current_cat_index += 1
            else:
                new_brains = evo.evolve()
                current_cat_index = 0
                is_seeded_run = False
                evo.population = []
        else:
            # Modalità WATCH
            if len(watch_cats) == 0 and len(evo.top_brains) > 0:
                count = min(watch_cat_count, len(evo.top_brains))
                for i in range(count):
                    cat = Cat(space, 200, evo.top_brains[i])
                    watch_cats.append(cat)
            
            # Aggiorna tutti i gatti vivi
            any_alive = False
            for cat in watch_cats:
                if cat.alive and cat.steps < SIMULATION_STEPS:
                    cat.update()
                    any_alive = True
            
            if any_alive:
                space.step(1/FPS)
            
            # CAMERA DINAMICA: segue sempre il gatto in prima posizione
            if watch_cats:
                # Trova il leader (quello più a destra)
                leader = max(watch_cats, key=lambda c: c.body.position.x)
                target_cam = leader.body.position.x - WIDTH/2
                camera_x += (target_cam - camera_x) * 0.05  # Smooth follow
        
        draw_background(screen, camera_x)
        
        if mode == "WATCH" and watch_cats:
            # Disegna tutti i gatti con colori IDENTICI (nessuna distinzione)
            for cat in watch_cats:
                draw_cat(screen, cat, camera_x)
            
            # Mostra chi è il leader (per debug/info)
            leader = max(watch_cats, key=lambda c: c.body.position.x)
            leader_dist = leader.body.position.x - 200
            
            color = (0, 160, 0) if any(cat.alive for cat in watch_cats) else (160, 0, 0)
            draw_text(screen, f"WATCH TOP {len(watch_cats)} - Gen:{evo.generation}", (10, 10), color=color)
            draw_text(screen, f"Record:{evo.best_fitness_ever:.1f} Leader:{leader_dist:.1f}", (10, 40))
            
            draw_slider(screen, watch_cat_count, MAX_WATCH_CATS, slider_rect, slider_dragging)
            
        else:
            color = (0, 100, 200) if is_seeded_run else BLACK
            txt = "SEEDED" if is_seeded_run else "RANDOM"
            draw_text(screen, f"FAST [{txt}] Gen{evo.generation} {current_cat_index}/{POPULATION_SIZE}", (10, 10), color=color)
            if evo.best_fitness_history:
                draw_text(screen, f"Best:{evo.best_fitness_history[-1]:.1f} All:{evo.best_fitness_ever:.1f}", (10, 40))
        
        draw_text(screen, "SPAZIO:Toggle R:NewRun C:Clear", (10, HEIGHT-30), size=16)
        
        pygame.display.flip()
        clock.tick(FPS)
    
    pygame.quit()


if __name__ == "__main__":
    main()