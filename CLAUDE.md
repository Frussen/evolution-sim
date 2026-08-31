# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Interactive AI evolution simulation platform built with Pygame. Virtual creatures (predators, prey, cats) evolve behaviors through neuroevolution — neural networks trained via genetic algorithms across generations.

## Running Simulations

Each simulation is a standalone Python script. No build step or test framework exists.

```bash
pip install pygame numpy pymunk   # pymunk only needed for cats/kimi_cat.py and cats/claude_cat.py

python predators/evolving_replica.py   # Core predator-prey evolution
python predators/savannah.py           # Enhanced ecosystem (terrain, water, day/night cycle)
python preys/prey_food.py              # Prey foraging with interactive grass circle
python preys/prey_circle.py            # Prey with circle-focused neural inputs
python cats/walking_cat.py             # Simple bipedal cat gait evolution
python cats/kimi_cat.py                # Cat with pymunk physics engine
python cats/claude_cat.py              # Dual-layer neural network cat
python snakes/simple_NN.py             # Boids-style predator-prey with numpy
python snakes/test.py                  # Rule-based (no neural network) predator-prey
```

## Architecture

Each script is self-contained with no shared modules. There is no package structure — simulations are organized by creature type into folders: `predators/`, `preys/`, `cats/`, `snakes/`. Code comments and UI labels are in Italian.

### Common Pattern Across Simulations

1. **Creature classes** at the top: position, velocity, heading, energy/health, and a `brain` (neural network as weight matrices)
2. **Neural network**: feed-forward, stored as raw numpy arrays or nested lists — no ML framework. Predator/prey sims use pure-Python math with tanh; `snakes/simple_NN.py` uses numpy with ReLU hidden + tanh output; cat sims use numpy with tanh throughout
3. **NN outputs**: 2 neurons (turn angle + speed) in predator/prey sims; 8 neurons (joint motor forces) in cat sims
4. **Genetic algorithm loop**: creatures with enough energy reproduce; offspring inherit parent brain + Gaussian mutation on weights
5. **Main game loop**: Pygame event handling → creature updates (sense → think → act) → rendering → UI overlays
6. **Configurable constants** at module top (population sizes, mutation rates, energy costs, vision parameters)

### Simulation Complexity Spectrum

- **snakes/test.py**: Pure rule-based behavior (boids flocking), no neural networks, toroidal world
- **snakes/simple_NN.py**: Numpy NN (128 hidden), boids-style nearest-neighbor inputs, toroidal world
- **preys/prey_food.py**: Minimal NN (5 inputs → 6 hidden → 2 output), prey-only foraging, adaptive screen size
- **predators/evolving_replica.py**: Full predator-prey with vision cones (24 NN inputs, 14 hidden), bounded arena
- **predators/savannah.py**: Extends evolving_replica (35 NN inputs) — adds terrain types, water/thirst, carcasses, day/night cycle, creature selection UI
- **cats/claude_cat.py**: Dual hidden layers (24→16→8→8), pymunk physics, crossover-based reproduction

### Key Mechanics

- **Vision system** (predator sims): 5 directional sectors, each reporting distance/type of nearest entity + wall proximity sensors (+ terrain sensors in savannah)
- **World topology**: predator/prey sims use bounded arenas with wall sensors; snake sims use toroidal wrapping (edges connect)
- **Energy model**: Creatures spend energy on movement/metabolism; gain energy by eating; reproduce at threshold
- **Rendering layers**: Background/terrain → entities (color-coded by type/energy) → UI buttons/sliders/stats → optional neural network visualization

## Enhancement Ideas

1. **Save/load evolved brains** — nessuna sim persiste i pesi NN. Serializzare (json/pickle) la popolazione migliore per riprendere l'evoluzione o confrontare brains tra ambienti diversi.
2. **Real-time evolution graphs** — tracciare popolazione, energia media e fitness nel tempo per visualizzare se l'evoluzione funziona e i collapse di popolazione.
3. **Shared neural network module** — ci sono 5+ implementazioni duplicate di NeuralNet; estrarle in un `core/brain.py` configurabile (layers, activation, mutation).
4. **Spatial hash grid** — i `get_inputs()` iterano su tutte le entità (O(n²)); un grid partitioning permetterebbe popolazioni molto più grandi.
5. **Crossover nelle sim predator-prey** — evolving_replica e savannah usano solo mutazione; portare il crossover già presente in claude_cat.py.
6. **Sensi aggiuntivi per savannah** — l'energia non è passata come input alla NN (solo la sete lo è); aggiungere densità di popolazione vicina e una forma di memoria ricorrente.
7. **Arena competitiva** — far evolvere popolazioni separatamente in ambienti diversi e confrontarle in un'arena condivisa.
