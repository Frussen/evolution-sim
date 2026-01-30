import pygame
import math
import random
from random import uniform
import copy

# Initialize pygame
pygame.init()
info = pygame.display.Info()
screen_w, screen_h = info.current_w, info.current_h

# Display configuration
_MAX_DESIRED_W, _MAX_DESIRED_H = 1500, 900
WIDTH = min(_MAX_DESIRED_W, int(screen_w * 1))
HEIGHT = min(_MAX_DESIRED_H, int(screen_h * 0.85))
FPS = 60
MARGIN = 15

# ================= SIMULATION PARAMETERS =================
NUM_INPUTS = 6          # Energy, time_in_gen, leg phase feedback (4 legs)
NUM_HIDDEN = 16
NUM_OUTPUTS = 4         # Control each leg: [front_left, front_right, back_left, back_right]

START_CATS = 50
MAX_ENERGY_CAT = 100
MOVEMENT_COST_PER_FRAME = 0.2
GENERATION_TIME = 900  # frames per generation (~15 seconds at 60 FPS)

# Evolution parameters
SELECTION_TOP_PERCENT = 0.3  # Top 30% survive and reproduce
CHILD_ENERGY_VAL = 50
MUTATION_PROB = 0.5
MUTATION_SIGMA = 1.0

# ================= VISUALIZATION =================
COLOR_BG = (15, 15, 22)
COLOR_SKY = (135, 206, 235)
COLOR_GRASS = (34, 139, 34)
COLOR_GROUND = (101, 84, 65)
COLOR_CAT = (200, 150, 100)
COLOR_CAT_DEAD = (60, 60, 60)
COLOR_WALL = (50, 50, 65)
COLOR_FINISH = (60, 220, 80)

GROUND_Y = HEIGHT * 0.65  # Where cats walk
HORIZON_Y = HEIGHT * 0.35  # Sky/grass boundary

show_stats = True
generation = 0

# ================= NEURAL NETWORK =================
class NeuralNet:
    def __init__(self):
        def rw(): 
            return random.gauss(0, 0.5)
        # Input → Hidden
        self.w_ih = [[rw() for _ in range(NUM_INPUTS)] for _ in range(NUM_HIDDEN)]
        self.b_h = [rw() * 0.1 for _ in range(NUM_HIDDEN)]
        # Hidden → Output (4 legs)
        self.w_ho = [[rw() for _ in range(NUM_HIDDEN)] for _ in range(NUM_OUTPUTS)]
        self.b_o = [rw() * 0.1 for _ in range(NUM_OUTPUTS)]

    def forward(self, inputs):
        # Hidden layer with tanh
        h = [math.tanh(self.b_h[j] + sum(inputs[i] * self.w_ih[j][i] for i in range(NUM_INPUTS))) 
             for j in range(NUM_HIDDEN)]
        # Output layer: tanh for leg control [-1, 1]
        out = [math.tanh(self.b_o[j] + sum(h[i] * self.w_ho[j][i] for i in range(NUM_HIDDEN))) 
               for j in range(NUM_OUTPUTS)]
        return out

    def mutate(self):
        for layer in (self.w_ih, self.w_ho):
            for row in layer:
                for i in range(len(row)):
                    if random.random() < MUTATION_PROB:
                        row[i] += random.gauss(0, MUTATION_SIGMA)


# ================= CAT CLASS =================
class Cat:
    def __init__(self, x, y, brain=None):
        self.x = x
        self.y = GROUND_Y  # Always on ground level
        self.vx = 0
        self.vy = 0
        
        # Body dimensions
        self.body_radius = 8
        self.body_length = 25
        self.head_radius = 6
        
        # Leg angles (radians) - current phase
        self.leg_angles = [0, math.pi, 0, math.pi]  # Front-left, Front-right, Back-left, Back-right
        self.leg_speeds = [0, 0, 0, 0]  # Current angular velocity for each leg
        
        # Heading direction
        self.heading = 0
        
        # Energy
        self.energy = MAX_ENERGY_CAT
        self.age_in_generation = 0
        
        # Brain
        self.brain = brain if brain else NeuralNet()
        
        # Tracking
        self.start_x = x
        self.distance_traveled = 0
        self.alive = True
    
    def get_leg_joints(self, leg_idx):
        """Get hip and paw positions for realistic leg articulation (side view)"""
        # Leg parameters
        thigh_length = 10
        shin_length = 10
        
        # Hip positions relative to body center (horizontal spacing)
        # Front legs further forward, back legs further back
        hip_x_offsets = [-12, -6, 6, 12]  # FL, FR, BL, BR
        
        hip_x = self.x + hip_x_offsets[leg_idx]
        hip_y = self.y
        
        # Leg angle: positive = leg down, negative = leg up (for walking)
        # Front and back legs should alternate (left/right pairs)
        leg_phase = self.leg_angles[leg_idx]
        
        # Thigh goes down/up
        thigh_vertical = thigh_length * math.sin(leg_phase)
        knee_x = hip_x
        knee_y = hip_y + thigh_vertical
        
        # Shin angle adds compliance
        shin_phase = leg_phase + 0.5 * math.cos(leg_phase * 2)
        shin_vertical = shin_length * math.sin(shin_phase)
        
        paw_x = knee_x
        paw_y = knee_y + shin_vertical
        
        # Clamp paw to ground or slightly below
        paw_y = max(self.y, paw_y)
        
        return (hip_x, hip_y), (knee_x, knee_y), (paw_x, paw_y)
    
    def update(self, frame_in_gen):
        # Inputs to neural network
        energy_norm = min(1.0, self.energy / MAX_ENERGY_CAT)
        time_norm = min(1.0, frame_in_gen / GENERATION_TIME)
        
        # Phase feedback from legs
        leg_phases = [math.sin(angle) for angle in self.leg_angles]
        
        inputs = [energy_norm, time_norm] + leg_phases[:4]
        
        # Forward pass
        outputs = self.brain.forward(inputs)
        
        # Update leg angles based on network output
        # outputs range [-1, 1], map to angular velocity
        for i in range(4):
            self.leg_speeds[i] = outputs[i] * 0.3  # Angular velocity
            self.leg_angles[i] += self.leg_speeds[i]
        
        # Simple locomotion model: average leg motion creates forward movement
        avg_leg_motion = sum([math.sin(self.leg_angles[i]) for i in range(4)]) / 4.0
        
        # Forward movement (positive leg motion pushes forward) - slowed down
        self.vx = avg_leg_motion * 1.2
        self.vy = 0  # Keep movement along x-axis mainly
        
        # Apply movement
        self.x += self.vx
        self.y += self.vy
        
        # Energy cost
        locomotion_cost = MOVEMENT_COST_PER_FRAME * (abs(self.vx) + abs(self.vy))
        self.energy -= locomotion_cost
        
        # Track distance
        self.distance_traveled = self.x - self.start_x
        
        # Death condition
        if self.energy <= 0:
            self.alive = False
        
        # Boundary constraint (keep on screen)
        self.x = max(MARGIN, min(WIDTH - MARGIN, self.x))
        self.y = HEIGHT // 2  # Keep cats on ground level
        
        self.age_in_generation += 1
    
    def draw(self, screen):
        if not self.alive:
            color = COLOR_CAT_DEAD
        else:
            color = COLOR_CAT
        
        # Body (horizontal, side view)
        body_length = 35
        body_height = 12
        
        # Main torso (horizontal ellipse)
        pygame.draw.ellipse(screen, color, (int(self.x - body_length/2), int(self.y - body_height/2), 
                                            int(body_length), int(body_height)))
        
        # Head (at right side)
        head_offset = body_length * 0.4
        head_x = self.x + head_offset
        head_y = self.y - 2
        pygame.draw.circle(screen, color, (int(head_x), int(head_y)), 8)
        
        # Ears (on top of head)
        ear_left_x = head_x - 4
        ear_right_x = head_x + 4
        ear_top_y = head_y - 8
        pygame.draw.polygon(screen, color, [
            (int(ear_left_x), int(head_y - 2)),
            (int(ear_left_x - 2), int(ear_top_y)),
            (int(ear_left_x + 2), int(head_y - 2))
        ])
        pygame.draw.polygon(screen, color, [
            (int(ear_right_x), int(head_y - 2)),
            (int(ear_right_x - 2), int(ear_top_y)),
            (int(ear_right_x + 2), int(head_y - 2))
        ])
        
        # Eyes
        pygame.draw.circle(screen, (50, 50, 50), (int(head_x + 1), int(head_y - 3)), 2)
        
        # Nose
        pygame.draw.circle(screen, (150, 100, 80), (int(head_x + 4), int(head_y)), 2)
        
        # Tail (wagging, curves upward)
        tail_start_x = self.x - body_length * 0.4
        tail_start_y = self.y
        tail_wave = 10 * math.sin(self.age_in_generation * 0.15)
        tail_end_x = tail_start_x - 20
        tail_end_y = tail_start_y - 8 + tail_wave
        pygame.draw.line(screen, color, (int(tail_start_x), int(tail_start_y)), 
                        (int(tail_end_x), int(tail_end_y)), 3)
        
        # Draw legs with articulation
        for i in range(4):
            hip, knee, paw = self.get_leg_joints(i)
            
            # Thigh
            pygame.draw.line(screen, color, (int(hip[0]), int(hip[1])), 
                           (int(knee[0]), int(knee[1])), 4)
            
            # Shin
            pygame.draw.line(screen, color, (int(knee[0]), int(knee[1])), 
                           (int(paw[0]), int(paw[1])), 3)
            
            # Paw
            pygame.draw.circle(screen, color, (int(paw[0]), int(paw[1])), 3)
        
        # Energy bar above cat
        bar_width = 30
        bar_height = 4
        energy_ratio = min(1.0, self.energy / MAX_ENERGY_CAT)
        pygame.draw.rect(screen, (100, 100, 100), (int(self.x - bar_width/2), int(self.y - 22), bar_width, bar_height))
        pygame.draw.rect(screen, (100, 220, 100), (int(self.x - bar_width/2), int(self.y - 22), int(bar_width * energy_ratio), bar_height))


# ================= SETUP =================
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
font = pygame.font.SysFont("Consolas", 14, bold=True)
font_small = pygame.font.SysFont("Consolas", 12)

# Initialize population
spawn_x = 100
spawn_y = GROUND_Y
cats = [Cat(spawn_x, spawn_y) for _ in range(START_CATS)]

running = True
frame = 0

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                show_stats = not show_stats
    
    # Update frame counter and check for generation end
    frame_in_generation = frame % GENERATION_TIME
    
    if frame_in_generation == 0 and frame > 0:
        # Selection and reproduction
        # Sort by distance traveled
        cats.sort(key=lambda c: c.distance_traveled, reverse=True)
        
        # Keep top performers
        num_survivors = max(1, int(len(cats) * SELECTION_TOP_PERCENT))
        survivors = cats[:num_survivors]
        
        # Reproduction: each survivor creates offspring
        new_gen = []
        for survivor in survivors:
            num_offspring = max(1, int(START_CATS / num_survivors))
            for _ in range(num_offspring):
                child_brain = copy.deepcopy(survivor.brain)
                child_brain.mutate()
                child = Cat(spawn_x, spawn_y, child_brain)
                new_gen.append(child)
        
        # Ensure we have roughly the same population
        while len(new_gen) < START_CATS:
            parent = random.choice(survivors)
            child_brain = copy.deepcopy(parent.brain)
            child_brain.mutate()
            child = Cat(spawn_x, spawn_y, child_brain)
            new_gen.append(child)
        
        cats = new_gen[:START_CATS]
        generation += 1
        
        # Reset distance tracking for next generation
        for cat in cats:
            cat.start_x = spawn_x
            cat.distance_traveled = 0
    
    # Update all cats
    for cat in cats:
        cat.update(frame_in_generation)
    
    # Draw
    screen.fill(COLOR_BG)
    
    # Sky and grass background
    pygame.draw.rect(screen, COLOR_SKY, (0, 0, WIDTH, int(HORIZON_Y)))
    pygame.draw.rect(screen, COLOR_GRASS, (int(HORIZON_Y), int(HORIZON_Y), WIDTH, HEIGHT - int(HORIZON_Y)))
    
    # Ground line (darker)
    pygame.draw.line(screen, COLOR_GROUND, (0, int(GROUND_Y)), (WIDTH, int(GROUND_Y)), 2)
    
    # Draw starting line
    pygame.draw.line(screen, (100, 100, 150), (spawn_x, int(GROUND_Y) - 50), (spawn_x, int(GROUND_Y) + 20), 3)
    
    # Draw cats
    for cat in cats:
        cat.draw(screen)
    
    # Draw finish line at max distance seen
    max_dist = max([c.distance_traveled for c in cats]) if cats else 0
    finish_x = spawn_x + max_dist
    if MARGIN < finish_x < WIDTH - MARGIN:
        pygame.draw.line(screen, COLOR_FINISH, (int(finish_x), int(GROUND_Y) - 50), (int(finish_x), int(GROUND_Y) + 20), 3)
    
    # Stats display
    if show_stats:
        alive_count = sum(1 for c in cats if c.alive)
        avg_distance = sum([c.distance_traveled for c in cats]) / len(cats) if cats else 0
        max_distance = max([c.distance_traveled for c in cats]) if cats else 0
        
        stats_lines = [
            f"Generation: {generation}",
            f"Frame: {frame_in_generation}/{GENERATION_TIME}",
            f"Alive: {alive_count}/{len(cats)}",
            f"Avg distance: {avg_distance:.1f}px",
            f"Max distance: {max_distance:.1f}px",
        ]
        
        y_pos = 30
        for line in stats_lines:
            surf = font.render(line, True, (200, 200, 200))
            screen.blit(surf, (20, y_pos))
            y_pos += 24
    
    pygame.display.flip()
    clock.tick(FPS)
    frame += 1

pygame.quit()
