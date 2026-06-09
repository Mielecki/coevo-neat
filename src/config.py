from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Config:

    screen_width: int = 3000
    screen_height: int = 3000

    viewport_width: int = 1280
    viewport_height: int = 720
    fps: int = 60
    headless: bool = False
    draw_fov: bool = False
    max_headless_steps: int = 0
    max_generations: int = 1000
    fixed_dt: float = 1.0 / 60.0
    verbose_lag_logs: bool = False
    random_seed: int = 1337

    initial_prey_count: int = 512
    initial_predator_count: int = 128
    initial_food_count: int = 500

    max_food_items: int = 1024
    food_spawn_per_second: float = 60.0
    food_energy_gain: float = 400.0
    predator_kill_energy_gain: float = 300.0

    prey_initial_energy: float = 400.0
    predator_initial_energy: float = 800.0

    prey_base_metabolism: float = 40.0
    predator_base_metabolism: float = 30.0
    movement_energy_cost: float = 0.25
    rotation_energy_cost: float = 0.08

    prey_max_speed: float = 110.0
    predator_max_speed: float = 170.0
    prey_max_turn_rate_deg: float = 240.0
    predator_max_turn_rate_deg: float = 120.0

    reproduction_energy_factor: float = 2.0
    reproduction_energy_transfer_ratio: float = 0.5

    generation_max_ticks: int = 8000

    prey_sensor_ray_count: int = 16
    predator_sensor_ray_count: int = 12
    prey_sensor_fov_deg: float = 270.0
    predator_sensor_fov_deg: float = 120.0
    prey_sensor_max_range: float = 400
    predator_sensor_max_range: float = 600

    prey_radius: float = 6.0
    predator_radius: float = 8.0
    food_radius: float = 8.0
    predator_attack_fov_deg: float = 60.0
    predator_attack_requires_forward_motion: bool = True

    brain_tick_rate: int = 4

    max_prey_capacity: int = 1024
    max_predator_capacity: int = 512

    tensorneat_device: str = "cuda"
    tensorneat_prey_population: int = 512
    tensorneat_predator_population: int = 128
    tensorneat_species_size: int = 13
    tensorneat_batch_size: int = 128
    tensorneat_output_size: int = 2
    tensorneat_output_activation: str = "tanh"
    load_networks_path: str | None = None
    save_networks_path: str | None = None
    checkpoint_dir: str = "checkpoints"
    checkpoint_every_generations: int = 25
    checkpoint_keep_last: int = 5
    checkpoint_save_latest: bool = True
    checkpoint_save_best: bool = True

    @property
    def prey_sensor_input_size(self) -> int:
        return max(1, int(self.prey_sensor_ray_count)) * 4 + 1

    @property
    def predator_sensor_input_size(self) -> int:
        return max(1, int(self.predator_sensor_ray_count)) * 4 + 1
