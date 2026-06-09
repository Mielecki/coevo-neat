# CoevoNEAT — GPU Coevolution of Predator–Prey Agents

A high-performance artificial life simulation where prey and predators evolve together through neuroevolution. The entire runtime is built around **data-oriented design** and **JAX-accelerated batch operations**, with separate NEAT populations for each species running on the GPU via [TensorNEAT](https://github.com/EMI-Group/tensorneat).

Watch hundreds of agents interact in real time, train headlessly at full speed, and inspect the best evolved neural networks through a live web dashboard.

## Highlights

- **Coevolution** — independent NEAT populations for prey and predators, evaluated and evolved every generation
- **Data-oriented runtime** — no per-agent objects; all state lives in flat GPU buffers (`SpeciesBuffers`)
- **Fully batched pipeline** — raycast sensors, neural inference, physics, collisions, reproduction, and food spawning run as vectorized JAX kernels
- **Two run modes** — real-time Pygame visualization or headless training with a `tqdm` progress bar
- **Checkpointing** — save/load network populations, with automatic per-generation, `latest`, and `best` checkpoints
- **Live analytics dashboard** — exports generation stats and champion network topologies to `web/` for interactive inspection

## How It Works

Each generation unfolds inside a fixed-size world (default 3000×3000). Prey must find food and survive; predators hunt prey within a forward-facing attack cone. Both species spend energy on movement and metabolism, and can reproduce when they accumulate enough energy.

Agents perceive the world through **raycast sensors** (distance + object type per ray, plus normalized energy). A small neural network outputs throttle and turn rate. At the end of each generation, fitness is aggregated per genome and both populations take an evolutionary step (`tell` → `ask`).

```text
┌─────────────────┐     ┌──────────────────┐     ┌───────────────────┐
│  SensorSystem     │──▶│  EvolutionManager  │──▶│    Environment      │
│  (raycasting)     │     │  (TensorNEAT)     │     │  physics/collisions │
└─────────────────┘     └──────────────────┘     │  reproduction/food  │
         ▲                        ▲                  └─────────┬─────────┘
         │                        │                             │
         └──────── SimulationManager (tick loop) ─────────────┘
                              │
                    ┌─────────┴─────────┐
                    ▼                     ▼
              Pygame (visual)     web/ exports (dashboard)
```

## Project Structure

```text
src/
  config.py              # Simulation and evolution parameters
  enums.py               # Agent species definitions
  environment.py         # World state, physics, collisions, rendering
  sensor_system.py       # Batched raycast perception
  evolution_manager.py   # TensorNEAT integration and checkpoint I/O
  simulation_manager.py  # Main loop, HUD, generation lifecycle
  utils.py               # Async writers for tick/stats logs
  main.py                # CLI entry point

web/
  index.html             # Live dashboard (network graphs + population charts)
```

Runtime exports (`stats.json`, `history/`, `data/ticks/`, checkpoints) are generated locally and excluded from version control.

## Requirements

- Python 3.11+
- CUDA-capable GPU recommended (JAX GPU backend; CPU works but is slower)
- Dependencies listed in `requirements.txt`:

```text
pygame, numpy, jax, tqdm, tensorneat (from GitHub)
```

## Installation

```bash
git clone https://github.com/Mielecki/coevo-neat.git
cd coevo-neat
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Install the JAX build that matches your hardware ([JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html)).

## Quick Start

**Visual mode** — watch the simulation in a Pygame window:

```bash
python -m src.main --fps 60 --seed 1337
```

Press `ESC` or `Q` to quit.

**Headless training** — evolve without rendering:

```bash
python -m src.main --headless --steps 50000 --generations 200 --seed 1337
```

A progress bar appears when `--steps` is greater than zero. Use `--steps 0` or `--generations 0` for no limit on that axis.

## Usage

### Visualization options

```bash
# Custom viewport size (world size is configured separately in Config)
python -m src.main --viewport-width 1920 --viewport-height 1080
```

### Training and checkpoints

```bash
# Train and save networks on exit
python -m src.main --headless --steps 100000 --save-networks checkpoints/latest.pkl

# Resume visualization from a trained checkpoint
python -m src.main --load-networks checkpoints/latest.pkl

# Automatic checkpointing every N generations
python -m src.main --headless --generations 500 \
  --checkpoint-dir checkpoints \
  --checkpoint-every 25 \
  --checkpoint-keep-last 5
```

Checkpoint flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint-dir` | `checkpoints` | Output directory |
| `--checkpoint-every` | `25` | Autosave interval in generations (`0` disables) |
| `--checkpoint-keep-last` | `5` | Number of `gen_*.pkl` files to retain |
| `--no-checkpoint-latest` | off | Skip updating `latest.pkl` |
| `--no-checkpoint-best` | off | Skip updating `best.pkl` on fitness improvement |

### Profiling

```bash
python -m src.main --headless --steps 5000 --profile \
  --profile-output tmp/profile.prof \
  --profile-sort cumulative \
  --profile-top 60
```

Produces a `.prof` file (viewable with [SnakeViz](https://jiffyclub.github.io/snakeviz/)) and prints a top-functions table to stdout.

### Web dashboard

While the simulation runs, it writes data to `web/`:

- `stats.json` — generation-level population counts and mean fitness
- `prey_latest.json` / `predator_latest.json` — champion network topology per species
- `history/` — per-generation network snapshots
- `data/ticks/` — per-tick alive counts within each generation

Serve the folder locally and open the dashboard:

```bash
cd web && python -m http.server 8080
# open http://localhost:8080
```

The dashboard shows evolved network graphs (vis.js) and population dynamics (Chart.js), with a generation slider to browse history.

## Configuration

Simulation parameters live in `src/config.py` as a `Config` dataclass. Key defaults:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `screen_width` / `screen_height` | 3000 | World size |
| `generation_max_ticks` | 8000 | Ticks per generation |
| `tensorneat_prey_population` | 512 | Prey genome pool size |
| `tensorneat_predator_population` | 128 | Predator genome pool size |
| `tensorneat_device` | `"cuda"` | JAX device (`cuda` maps to GPU) |
| `brain_tick_rate` | 4 | Neural network inference every N ticks |
| `prey_sensor_ray_count` | 16 | Raycast count for prey |
| `predator_sensor_ray_count` | 12 | Raycast count for predators |

Edit `src/config.py` directly to tune the simulation.

On startup the application sets `jax_enable_x64=False` for better GPU throughput.

## Architecture Notes

- **`Environment`** holds all prey, predator, and food state in fixed-size JAX arrays. Updates run through a single compiled mega-kernel per tick.
- **`SensorSystem`** performs batched raycasting against all world objects, returning a `[population, sensor_input_size]` matrix per species.
- **`EvolutionManager`** wraps TensorNEAT: batched `forward` for inference, `tell`/`ask` for the evolutionary cycle, and JSON export of champion genomes for the dashboard.
- **`SimulationManager`** coordinates the tick loop, generation boundaries, checkpointing, async logging, and optional Pygame rendering.

Fitness combines survival time and energy gained (weights differ per species so predators are rewarded for successful hunts).

## CLI Reference

```
--headless              Train without a Pygame window
--steps N               Max simulation ticks (0 = unlimited)
--generations N         Max generations (0 = unlimited)
--seed N                Random seed
--fps N                 Target frame rate in visual mode
--draw-fov              Draw agent field-of-view
--viewport-width N      Window width
--viewport-height N     Window height
--verbose-lag-logs      Enable diagnostic logging
--load-networks PATH    Load a checkpoint at startup
--save-networks PATH    Save a checkpoint on shutdown
--profile               Run under cProfile
```
