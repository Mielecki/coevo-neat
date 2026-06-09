from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import jax

jax.config.update("jax_enable_x64", False)

import jax.numpy as jnp
import numpy as np
import pygame
from pygame import Surface

from .config import Config
from .enums import AgentSpecies

if TYPE_CHECKING:
    from .evolution_manager import EvolutionManager
    from .sensor_system import SensorSystem


def _jit_if_available(
    func: Any = None, *, static_argnums=(0,), static_argnames=()
) -> Any:
    if func is None:
        return lambda f: _jit_if_available(
            f, static_argnums=static_argnums, static_argnames=static_argnames
        )

    return jax.jit(func, static_argnums=static_argnums, static_argnames=static_argnames)


import flax.struct


@flax.struct.dataclass
class SpeciesBuffers:

    positions: jnp.ndarray
    headings_deg: jnp.ndarray
    velocities: jnp.ndarray
    energies: jnp.ndarray
    is_alive: jnp.ndarray
    age_ticks: jnp.ndarray
    total_energy_gained: jnp.ndarray
    genome_indices: jnp.ndarray
    last_throttle: jnp.ndarray
    last_turn: jnp.ndarray


class Environment:

    def __init__(self, config: Config) -> None:
        self.config = config
        self.prey_capacity = max(1, int(config.max_prey_capacity))
        self.predator_capacity = max(1, int(config.max_predator_capacity))
        self.food_capacity = max(1, int(config.max_food_items))

        self.prey_population_size = max(1, int(config.tensorneat_prey_population))
        self.predator_population_size = max(
            1, int(config.tensorneat_predator_population)
        )

        self._fitness_banks = {
            AgentSpecies.PREY: jnp.zeros(
                (self.prey_population_size,), dtype=jnp.float32
            ),
            AgentSpecies.PREDATOR: jnp.zeros(
                (self.predator_population_size,), dtype=jnp.float32
            ),
        }

        self._species_buffers: dict[AgentSpecies, SpeciesBuffers] = {
            AgentSpecies.PREY: self._create_species_buffers(self.prey_capacity),
            AgentSpecies.PREDATOR: self._create_species_buffers(self.predator_capacity),
        }

        self.food_positions = jnp.zeros((self.food_capacity, 2), dtype=jnp.float32)
        self.food_energy_values = jnp.full(
            (self.food_capacity,), float(config.food_energy_gain), dtype=jnp.float32
        )
        self.food_is_alive = jnp.zeros((self.food_capacity,), dtype=jnp.float32)

        self._sensor_object_radii = jnp.concatenate(
            (
                jnp.full(
                    (self.food_capacity,), float(config.food_radius), dtype=jnp.float32
                ),
                jnp.full(
                    (self.prey_capacity,), float(config.prey_radius), dtype=jnp.float32
                ),
                jnp.full(
                    (self.predator_capacity,),
                    float(config.predator_radius),
                    dtype=jnp.float32,
                ),
            )
        )

        self._sensor_object_types = jnp.concatenate(
            (
                jnp.full((self.food_capacity,), 2, dtype=jnp.int32),
                jnp.full((self.prey_capacity,), 3, dtype=jnp.int32),
                jnp.full((self.predator_capacity,), 4, dtype=jnp.int32),
            )
        )

        self._sensor_object_indices = jnp.arange(
            self.food_capacity + self.prey_capacity + self.predator_capacity,
            dtype=jnp.int32,
        )
        self._prey_owner_object_indices = (
            jnp.arange(self.prey_capacity, dtype=jnp.int32) + self.food_capacity
        )
        self._predator_owner_object_indices = (
            jnp.arange(self.predator_capacity, dtype=jnp.int32)
            + self.food_capacity
            + self.prey_capacity
        )

        self._food_spawn_accumulator = 0.0
        self._jax_food_key: Any | None = None

    def _create_species_buffers(self, capacity: int) -> SpeciesBuffers:
        return SpeciesBuffers(
            positions=jnp.zeros((capacity, 2), dtype=jnp.float32),
            headings_deg=jnp.zeros((capacity,), dtype=jnp.float32),
            velocities=jnp.zeros((capacity, 2), dtype=jnp.float32),
            energies=jnp.zeros((capacity,), dtype=jnp.float32),
            is_alive=jnp.zeros((capacity,), dtype=jnp.float32),
            age_ticks=jnp.zeros((capacity,), dtype=jnp.int32),
            total_energy_gained=jnp.zeros((capacity,), dtype=jnp.float32),
            genome_indices=jnp.full((capacity,), -1, dtype=jnp.int32),
            last_throttle=jnp.zeros((capacity,), dtype=jnp.float32),
            last_turn=jnp.zeros((capacity,), dtype=jnp.float32),
        )

    def reset(self) -> None:

        self._reset_species_buffers(AgentSpecies.PREY)
        self._reset_species_buffers(AgentSpecies.PREDATOR)

        self.food_positions = jnp.zeros_like(self.food_positions, dtype=jnp.float32)
        self.food_energy_values = jnp.full_like(
            self.food_energy_values,
            float(self.config.food_energy_gain),
            dtype=jnp.float32,
        )
        self.food_is_alive = jnp.zeros_like(self.food_is_alive, dtype=jnp.float32)

        for species in (AgentSpecies.PREY, AgentSpecies.PREDATOR):
            self._fitness_banks[species] = jnp.zeros_like(self._fitness_banks[species])

        self._food_spawn_accumulator = 0.0

    def _reset_species_buffers(self, species: AgentSpecies) -> None:
        buffers = self._buffers(species)

        self._species_buffers[species] = buffers.replace(
            positions=jnp.zeros_like(buffers.positions),
            headings_deg=jnp.zeros_like(buffers.headings_deg),
            velocities=jnp.zeros_like(buffers.velocities),
            energies=jnp.zeros_like(buffers.energies),
            is_alive=jnp.zeros_like(buffers.is_alive),
            age_ticks=jnp.zeros_like(buffers.age_ticks),
            total_energy_gained=jnp.zeros_like(buffers.total_energy_gained),
            genome_indices=jnp.full_like(buffers.genome_indices, -1),
            last_throttle=jnp.zeros_like(buffers.last_throttle),
            last_turn=jnp.zeros_like(buffers.last_turn),
        )

    def seed_generation(self, generation_seed: int) -> None:

        self.reset()
        rng = np.random.default_rng(int(generation_seed))

        self._seed_species(AgentSpecies.PREY, rng)
        self._seed_species(AgentSpecies.PREDATOR, rng)

        initial_food = min(self.config.initial_food_count, self.food_capacity)

        self._jax_food_key = jax.random.PRNGKey(int(generation_seed))

        (
            self.food_positions,
            self.food_energy_values,
            self.food_is_alive,
            self._jax_food_key,
        ) = self._spawn_food_kernel(
            food_positions=self.food_positions,
            food_energy_values=self.food_energy_values,
            food_is_alive=self.food_is_alive,
            spawn_count=jnp.asarray(initial_food, dtype=jnp.int32),
            key=self._jax_food_key,
            screen_width=float(self.config.screen_width),
            screen_height=float(self.config.screen_height),
            food_energy_gain=float(self.config.food_energy_gain),
        )

    def _seed_species(self, species: AgentSpecies, rng: np.random.Generator) -> None:
        buffers = self._buffers(species)
        capacity = self.get_species_capacity(species)
        if species == AgentSpecies.PREY:
            initial_count = max(0, int(self.config.initial_prey_count))
            population_size = max(1, int(self.config.tensorneat_prey_population))
            initial_energy = float(self.config.prey_initial_energy)
        else:
            initial_count = max(0, int(self.config.initial_predator_count))
            population_size = max(1, int(self.config.tensorneat_predator_population))
            initial_energy = float(self.config.predator_initial_energy)
        active_count = min(capacity, initial_count)

        positions = np.zeros((capacity, 2), dtype=np.float32)
        headings = np.zeros((capacity,), dtype=np.float32)
        velocities = np.zeros((capacity, 2), dtype=np.float32)
        energies = np.zeros((capacity,), dtype=np.float32)
        is_alive = np.zeros((capacity,), dtype=np.float32)
        age_ticks = np.zeros((capacity,), dtype=np.int32)
        total_energy_gained = np.zeros((capacity,), dtype=np.float32)
        genome_indices = np.full((capacity,), -1, dtype=np.int32)
        last_throttle = np.zeros((capacity,), dtype=np.float32)
        last_turn = np.zeros((capacity,), dtype=np.float32)

        if active_count > 0:
            positions[:active_count, 0] = rng.uniform(
                0.0, float(self.config.screen_width), size=active_count
            ).astype(np.float32)
            positions[:active_count, 1] = rng.uniform(
                0.0, float(self.config.screen_height), size=active_count
            ).astype(np.float32)
            headings[:active_count] = rng.uniform(0.0, 360.0, size=active_count).astype(
                np.float32
            )
            energies[:active_count] = initial_energy
            is_alive[:active_count] = 1.0
            genome_indices[:active_count] = (
                np.arange(active_count, dtype=np.int32) % population_size
            ).astype(np.int32)

        self._species_buffers[species] = buffers.replace(
            positions=jnp.asarray(positions, dtype=jnp.float32),
            headings_deg=jnp.asarray(headings, dtype=jnp.float32),
            velocities=jnp.asarray(velocities, dtype=jnp.float32),
            energies=jnp.asarray(energies, dtype=jnp.float32),
            is_alive=jnp.asarray(is_alive, dtype=jnp.float32),
            age_ticks=jnp.asarray(age_ticks, dtype=jnp.int32),
            total_energy_gained=jnp.asarray(total_energy_gained, dtype=jnp.float32),
            genome_indices=jnp.asarray(genome_indices, dtype=jnp.int32),
            last_throttle=jnp.asarray(last_throttle, dtype=jnp.float32),
            last_turn=jnp.asarray(last_turn, dtype=jnp.float32),
        )

    def update(
        self,
        dt: float,
        tick_index: int,
        sensor_system: "SensorSystem",
        evolution_manager: "EvolutionManager | None" = None,
    ) -> None:

        self.run_decision_phase(tick_index, sensor_system, evolution_manager)

        prey = self._buffers(AgentSpecies.PREY)
        pred = self._buffers(AgentSpecies.PREDATOR)

        (
            new_prey,
            new_pred,
            self.food_positions,
            self.food_energy_values,
            self.food_is_alive,
            self._food_spawn_accumulator,
            self._jax_food_key,
            prey_fitness_delta,
            pred_fitness_delta,
        ) = self._mega_step_kernel(
            float(dt),
            self._food_spawn_accumulator,
            self._jax_food_key,
            prey,
            pred,
            self.food_positions,
            self.food_energy_values,
            self.food_is_alive,
            float(self.config.prey_initial_energy),
            float(self.config.predator_initial_energy),
            float(self.config.reproduction_energy_factor),
            float(self.config.reproduction_energy_transfer_ratio),
            1.0,
            1.0,
            0.05,
            15.0,
        )

        self._species_buffers[AgentSpecies.PREY] = new_prey
        self._species_buffers[AgentSpecies.PREDATOR] = new_pred
        self._fitness_banks[AgentSpecies.PREY] += prey_fitness_delta
        self._fitness_banks[AgentSpecies.PREDATOR] += pred_fitness_delta

    def run_decision_phase(
        self,
        tick_index: int,
        sensor_system: "SensorSystem",
        evolution_manager: "EvolutionManager | None" = None,
    ) -> None:

        brain_tick_rate = max(1, int(self.config.brain_tick_rate))
        if tick_index % brain_tick_rate != 0:
            return

        sensor_system.begin_frame(self)
        try:
            self._update_species_decisions(
                AgentSpecies.PREY, sensor_system, evolution_manager
            )
            self._update_species_decisions(
                AgentSpecies.PREDATOR, sensor_system, evolution_manager
            )
        finally:
            sensor_system.end_frame()

    def _update_species_decisions(
        self,
        species: AgentSpecies,
        sensor_system: "SensorSystem",
        evolution_manager: "EvolutionManager | None",
    ) -> None:
        buffers = self._buffers(species)

        sensor_batch = sensor_system.sample_for_species(species, self)
        if evolution_manager is None:
            controls_raw = jnp.zeros(
                (self.get_species_capacity(species), 2), dtype=jnp.float32
            )
        else:
            controls_raw = evolution_manager.activate_species_for_bodies(
                species=species,
                sensor_batch=sensor_batch,
                genome_indices=buffers.genome_indices,
                is_alive=buffers.is_alive,
            )

        new_throttle, new_turn = self._normalize_and_apply_controls_kernel(
            controls_raw, buffers.is_alive
        )

        self._species_buffers[species] = buffers.replace(
            last_throttle=new_throttle, last_turn=new_turn
        )

    @_jit_if_available
    def _normalize_and_apply_controls_kernel(self, controls_raw, alive):
        throttle = jnp.clip((controls_raw[:, 0] + 1.0) * 0.5, 0.0, 1.0)
        turn = jnp.clip(controls_raw[:, 1], -1.0, 1.0)
        return (throttle * alive).astype(jnp.float32), (turn * alive).astype(
            jnp.float32
        )

    @_jit_if_available
    def _physics_step_kernel(
        self,
        positions: Any,
        headings_deg: Any,
        velocities: Any,
        energies: Any,
        is_alive: Any,
        age_ticks: Any,
        throttle: Any,
        turn: Any,
        dt: float,
        max_speed: float,
        max_turn_rate: float,
        base_metabolism: float,
    ) -> tuple[Any, Any, Any, Any, Any, Any]:

        alive = jnp.clip(is_alive, 0.0, 1.0)

        new_headings = (headings_deg + (turn * max_turn_rate * dt)) % 360.0
        headings_rad = jnp.deg2rad(new_headings)

        speed = throttle * max_speed
        direction = jnp.stack((jnp.cos(headings_rad), jnp.sin(headings_rad)), axis=-1)
        new_velocities = direction * speed[:, None] * alive[:, None]

        new_positions = positions + new_velocities * dt

        new_positions = jnp.stack(
            (
                jnp.clip(new_positions[:, 0], 0.0, float(self.config.screen_width)),
                jnp.clip(new_positions[:, 1], 0.0, float(self.config.screen_height)),
            ),
            axis=-1,
        )

        movement_cost = float(self.config.movement_energy_cost) * speed
        rotation_cost = float(self.config.rotation_energy_cost) * jnp.abs(turn)
        energy_cost = (base_metabolism + movement_cost + rotation_cost) * dt

        new_energies = energies - (energy_cost * alive)
        new_is_alive = jnp.where(new_energies > 0.0, alive, 0.0)
        new_energies = jnp.where(new_is_alive > 0.5, new_energies, 0.0)
        new_velocities = new_velocities * new_is_alive[:, None]

        new_age_ticks = age_ticks + alive.astype(age_ticks.dtype)
        return (
            new_positions,
            new_headings,
            new_velocities,
            new_energies,
            new_is_alive,
            new_age_ticks,
        )

    @_jit_if_available
    def _resolve_prey_food_collisions_kernel(
        self,
        prey_positions: Any,
        prey_energies: Any,
        prey_is_alive: Any,
        prey_total_gain: Any,
        food_positions: Any,
        food_energy_values: Any,
        food_is_alive: Any,
    ) -> tuple[Any, Any, Any]:
        collision_distance_sq = float(
            (self.config.prey_radius + self.config.food_radius) ** 2
        )

        prey_sq = jnp.sum(prey_positions * prey_positions, axis=1)[:, None]
        food_sq = jnp.sum(food_positions * food_positions, axis=1)[None, :]
        dot_product = jnp.dot(prey_positions, food_positions.T)

        dist_sq = jnp.maximum(prey_sq + food_sq - 2.0 * dot_product, 0.0)

        candidate = (prey_is_alive[:, None] > 0.5) & (food_is_alive[None, :] > 0.5)
        candidate = candidate & (dist_sq <= collision_distance_sq)

        masked_dist = jnp.where(candidate, dist_sq, jnp.inf)
        prey_choice_food = jnp.argmin(masked_dist, axis=1)
        prey_choice_dist = masked_dist[
            jnp.arange(masked_dist.shape[0]), prey_choice_food
        ]
        prey_has_choice = jnp.isfinite(prey_choice_dist)

        food_indices = jnp.arange(masked_dist.shape[1], dtype=jnp.int32)
        selected_matrix = prey_has_choice[:, None] & (
            prey_choice_food[:, None] == food_indices[None, :]
        )

        dist_for_food = jnp.where(selected_matrix, prey_choice_dist[:, None], jnp.inf)
        best_dist_per_food = jnp.min(dist_for_food, axis=0)

        prey_indices = jnp.arange(masked_dist.shape[0], dtype=jnp.int32)
        invalid_prey_index = jnp.asarray(masked_dist.shape[0], dtype=jnp.int32)
        tied_best = selected_matrix & (dist_for_food == best_dist_per_food[None, :])
        winner_prey_per_food = jnp.min(
            jnp.where(tied_best, prey_indices[:, None], invalid_prey_index),
            axis=0,
        )
        food_consumed = winner_prey_per_food < invalid_prey_index

        safe_winner_indices = jnp.where(food_consumed, winner_prey_per_food, 0)
        consumed_energy = food_energy_values * food_consumed.astype(jnp.float32)
        gain_per_prey = (
            jnp.zeros_like(prey_energies).at[safe_winner_indices].add(consumed_energy)
        )

        new_prey_energies = prey_energies + gain_per_prey
        new_prey_total_gain = prey_total_gain + gain_per_prey
        new_food_is_alive = jnp.where(food_consumed, 0.0, food_is_alive)
        return new_prey_energies, new_prey_total_gain, new_food_is_alive

    @_jit_if_available
    def _resolve_predator_prey_collisions_kernel(
        self,
        predator_positions: Any,
        predator_headings_deg: Any,
        predator_velocities: Any,
        predator_energies: Any,
        predator_is_alive: Any,
        predator_total_gain: Any,
        prey_positions: Any,
        prey_energies: Any,
        prey_is_alive: Any,
    ) -> tuple[Any, Any, Any, Any]:
        kill_distance_sq = float(
            (self.config.prey_radius + self.config.predator_radius) ** 2
        )

        delta = prey_positions[None, :, :] - predator_positions[:, None, :]

        pred_sq = jnp.sum(predator_positions**2, axis=1)[:, None]
        prey_sq = jnp.sum(prey_positions**2, axis=1)[None, :]
        dot_prod = jnp.dot(predator_positions, prey_positions.T)
        dist_sq = jnp.maximum(pred_sq + prey_sq - 2.0 * dot_prod, 0.0)

        candidate = (predator_is_alive[:, None] > 0.5) & (prey_is_alive[None, :] > 0.5)
        candidate = candidate & (dist_sq <= kill_distance_sq)

        dist = jnp.sqrt(jnp.maximum(dist_sq, 1e-9))
        prey_dir = delta / dist[:, :, None]

        heading_rad = jnp.deg2rad(predator_headings_deg)
        heading_dir = jnp.stack((jnp.cos(heading_rad), jnp.sin(heading_rad)), axis=1)
        attack_cos_threshold = float(
            math.cos(math.radians(self.config.predator_attack_fov_deg * 0.5))
        )
        heading_alignment = jnp.sum(heading_dir[:, None, :] * prey_dir, axis=2)
        candidate = candidate & (heading_alignment >= attack_cos_threshold)

        if self.config.predator_attack_requires_forward_motion:
            speed = jnp.linalg.norm(predator_velocities, axis=1)

            min_speed = float(self.config.predator_max_speed) * 0.25
            has_motion = speed > min_speed
            safe_speed = jnp.where(has_motion, speed, 1.0)
            motion_dir = predator_velocities / safe_speed[:, None]
            motion_alignment = jnp.sum(motion_dir[:, None, :] * prey_dir, axis=2)
            candidate = candidate & has_motion[:, None] & (motion_alignment > 0.0)

        masked_dist = jnp.where(candidate, dist_sq, jnp.inf)
        predator_choice_prey = jnp.argmin(masked_dist, axis=1)
        predator_choice_dist = masked_dist[
            jnp.arange(masked_dist.shape[0]), predator_choice_prey
        ]
        predator_has_choice = jnp.isfinite(predator_choice_dist)

        prey_indices = jnp.arange(masked_dist.shape[1], dtype=jnp.int32)
        selected_matrix = predator_has_choice[:, None] & (
            predator_choice_prey[:, None] == prey_indices[None, :]
        )

        dist_for_prey = jnp.where(
            selected_matrix, predator_choice_dist[:, None], jnp.inf
        )
        best_dist_per_prey = jnp.min(dist_for_prey, axis=0)

        predator_indices = jnp.arange(masked_dist.shape[0], dtype=jnp.int32)
        invalid_predator_index = jnp.asarray(masked_dist.shape[0], dtype=jnp.int32)
        tied_best = selected_matrix & (dist_for_prey == best_dist_per_prey[None, :])
        winner_predator_per_prey = jnp.min(
            jnp.where(tied_best, predator_indices[:, None], invalid_predator_index),
            axis=0,
        )
        prey_killed = winner_predator_per_prey < invalid_predator_index

        safe_winner_indices = jnp.where(prey_killed, winner_predator_per_prey, 0)
        gained_energy = prey_killed.astype(jnp.float32) * float(
            self.config.predator_kill_energy_gain
        )
        gain_per_predator = (
            jnp.zeros_like(predator_energies).at[safe_winner_indices].add(gained_energy)
        )

        new_predator_energies = predator_energies + gain_per_predator
        new_predator_total_gain = predator_total_gain + gain_per_predator
        new_prey_energies = jnp.where(prey_killed, 0.0, prey_energies)
        new_prey_is_alive = jnp.where(prey_killed, 0.0, prey_is_alive)
        return (
            new_predator_energies,
            new_predator_total_gain,
            new_prey_energies,
            new_prey_is_alive,
        )

    @_jit_if_available(static_argnames=["population_size"])
    def _resolve_reproduction_kernel(
        self,
        buffers: SpeciesBuffers,
        initial_energy: float,
        reproduction_energy_factor: float,
        reproduction_transfer_ratio: float,
        population_size: int,
        age_weight: float,
        gain_weight: float,
    ) -> tuple[SpeciesBuffers, Any]:
        threshold = jnp.asarray(
            initial_energy * reproduction_energy_factor, dtype=jnp.float32
        )
        transfer_ratio = jnp.asarray(reproduction_transfer_ratio, dtype=jnp.float32)
        capacity = int(buffers.energies.shape[0])

        wants = (
            (buffers.is_alive > 0.5)
            & (buffers.energies >= threshold)
            & (buffers.genome_indices >= 0)
        )
        dead_slots = buffers.is_alive <= 0.5

        sorted_parent_indices = jnp.argsort(~wants)
        sorted_dead_indices = jnp.argsort(~dead_slots)

        num_parents = jnp.sum(wants.astype(jnp.int32))
        num_dead = jnp.sum(dead_slots.astype(jnp.int32))
        valid_reproductions = jnp.minimum(num_parents, num_dead)

        pair_rank = jnp.arange(capacity, dtype=jnp.int32)
        pair_mask = pair_rank < valid_reproductions

        parent_idx = sorted_parent_indices
        child_idx = sorted_dead_indices

        overwritten_age = buffers.age_ticks[child_idx].astype(jnp.float32)
        overwritten_gain = buffers.total_energy_gained[child_idx].astype(jnp.float32)
        overwritten_fitness = (
            overwritten_age * age_weight + overwritten_gain * gain_weight
        )
        overwritten_genomes = buffers.genome_indices[child_idx]

        valid_bank = (
            pair_mask
            & (overwritten_genomes >= 0)
            & (overwritten_genomes < population_size)
        )

        safe_indices = jnp.where(valid_bank, overwritten_genomes, 0).astype(jnp.int32)
        safe_fitness = jnp.where(valid_bank, overwritten_fitness, 0.0)

        banked_delta = jnp.bincount(
            safe_indices, weights=safe_fitness, length=population_size
        )

        parent_energy = buffers.energies[parent_idx]
        transfer_energy = jnp.where(
            pair_mask, parent_energy * transfer_ratio, 0.0
        ).astype(jnp.float32)

        parent_energy_delta = (
            jnp.zeros_like(buffers.energies).at[parent_idx].add(-transfer_energy)
        )
        child_energy_delta = (
            jnp.zeros_like(buffers.energies).at[child_idx].add(transfer_energy)
        )
        new_energies = buffers.energies + parent_energy_delta + child_energy_delta

        parent_headings = buffers.headings_deg[parent_idx]
        offset_x = jnp.cos(jnp.deg2rad(parent_headings)) * jnp.asarray(
            2.0, dtype=jnp.float32
        )
        offset_y = jnp.sin(jnp.deg2rad(parent_headings)) * jnp.asarray(
            2.0, dtype=jnp.float32
        )
        child_positions = buffers.positions[parent_idx] + jnp.stack(
            (offset_x, offset_y), axis=1
        )
        child_positions = jnp.stack(
            (
                jnp.clip(child_positions[:, 0], 0.0, float(self.config.screen_width)),
                jnp.clip(child_positions[:, 1], 0.0, float(self.config.screen_height)),
            ),
            axis=-1,
        )

        child_headings = parent_headings
        child_genomes = buffers.genome_indices[parent_idx]
        child_alive_values = jnp.where(pair_mask, 1.0, buffers.is_alive[child_idx])
        child_age_values = jnp.where(pair_mask, 0, buffers.age_ticks[child_idx]).astype(
            buffers.age_ticks.dtype
        )
        child_gain_values = jnp.where(
            pair_mask, 0.0, buffers.total_energy_gained[child_idx]
        ).astype(jnp.float32)

        new_positions = buffers.positions.at[child_idx].set(
            jnp.where(pair_mask[:, None], child_positions, buffers.positions[child_idx])
        )
        new_headings = buffers.headings_deg.at[child_idx].set(
            jnp.where(pair_mask, child_headings, buffers.headings_deg[child_idx])
        )
        new_velocities = buffers.velocities.at[child_idx].set(
            jnp.where(
                pair_mask[:, None],
                jnp.zeros_like(buffers.velocities[child_idx], dtype=jnp.float32),
                buffers.velocities[child_idx],
            )
        )
        new_is_alive = buffers.is_alive.at[child_idx].set(child_alive_values)
        new_age_ticks = buffers.age_ticks.at[child_idx].set(child_age_values)
        new_total_energy_gained = buffers.total_energy_gained.at[child_idx].set(
            child_gain_values
        )
        new_genome_indices = buffers.genome_indices.at[child_idx].set(
            jnp.where(pair_mask, child_genomes, buffers.genome_indices[child_idx])
        )
        new_last_throttle = buffers.last_throttle.at[child_idx].set(
            jnp.where(
                pair_mask,
                jnp.zeros_like(buffers.last_throttle[child_idx], dtype=jnp.float32),
                buffers.last_throttle[child_idx],
            )
        )
        new_last_turn = buffers.last_turn.at[child_idx].set(
            jnp.where(
                pair_mask,
                jnp.zeros_like(buffers.last_turn[child_idx], dtype=jnp.float32),
                buffers.last_turn[child_idx],
            )
        )

        new_buffers = buffers.replace(
            positions=new_positions,
            headings_deg=new_headings,
            velocities=new_velocities,
            energies=new_energies,
            is_alive=new_is_alive,
            age_ticks=new_age_ticks,
            total_energy_gained=new_total_energy_gained,
            genome_indices=new_genome_indices,
            last_throttle=new_last_throttle,
            last_turn=new_last_turn,
        )

        return new_buffers, banked_delta

    @_jit_if_available
    def _spawn_food_kernel(
        self,
        food_positions: Any,
        food_energy_values: Any,
        food_is_alive: Any,
        spawn_count: Any,
        key: Any,
        screen_width: float,
        screen_height: float,
        food_energy_gain: float,
    ) -> tuple[Any, Any, Any, Any]:
        capacity = int(food_is_alive.shape[0])
        dead_slots = food_is_alive <= 0.5
        dead_count = jnp.sum(dead_slots.astype(jnp.int32))
        actual_spawn_count = jnp.minimum(
            jnp.asarray(spawn_count, dtype=jnp.int32), dead_count
        )

        sorted_dead_indices = jnp.argsort(~dead_slots)
        rank = jnp.arange(capacity, dtype=jnp.int32)
        spawn_mask = rank < actual_spawn_count

        key, key_x, key_y = jax.random.split(key, 3)
        rand_x = jax.random.uniform(
            key_x,
            shape=(capacity,),
            minval=0.0,
            maxval=screen_width,
            dtype=jnp.float32,
        )
        rand_y = jax.random.uniform(
            key_y,
            shape=(capacity,),
            minval=0.0,
            maxval=screen_height,
            dtype=jnp.float32,
        )
        sampled_positions = jnp.stack((rand_x, rand_y), axis=1)
        slot_positions = food_positions[sorted_dead_indices]
        slot_energy = food_energy_values[sorted_dead_indices]
        slot_alive = food_is_alive[sorted_dead_indices]

        new_slot_positions = jnp.where(
            spawn_mask[:, None], sampled_positions, slot_positions
        )
        new_slot_energy = jnp.where(
            spawn_mask, jnp.asarray(food_energy_gain, dtype=jnp.float32), slot_energy
        )
        new_slot_alive = jnp.where(spawn_mask, 1.0, slot_alive)

        new_food_positions = food_positions.at[sorted_dead_indices].set(
            new_slot_positions
        )
        new_food_energy_values = food_energy_values.at[sorted_dead_indices].set(
            new_slot_energy
        )
        new_food_is_alive = food_is_alive.at[sorted_dead_indices].set(new_slot_alive)
        return new_food_positions, new_food_energy_values, new_food_is_alive, key

    def get_species_capacity(self, species: AgentSpecies) -> int:
        if species == AgentSpecies.PREY:
            return self.prey_capacity
        return self.predator_capacity

    def get_alive_count(self, species: AgentSpecies) -> int:
        buffers = self._buffers(species)
        return int(jax.device_get(jnp.count_nonzero(buffers.is_alive > 0.5)))

    def get_food_count(self) -> int:
        return int(jax.device_get(jnp.count_nonzero(self.food_is_alive > 0.5)))

    def collect_generation_fitness(self, species: AgentSpecies) -> np.ndarray:

        buffers = self._buffers(species)
        population_size = (
            self.prey_population_size
            if species == AgentSpecies.PREY
            else self.predator_population_size
        )

        if species == AgentSpecies.PREY:
            age_w, gain_w = 1.0, 1.0
        else:
            age_w, gain_w = 0.05, 15.0

        current_bodies_fitness = self._collect_generation_fitness_kernel(
            genome_indices=buffers.genome_indices,
            age_ticks=buffers.age_ticks,
            total_energy_gained=buffers.total_energy_gained,
            population_size=population_size,
            age_weight=age_w,
            gain_weight=gain_w,
        )

        total_fitness = self._fitness_banks[species] + current_bodies_fitness
        return total_fitness

    @_jit_if_available(static_argnames=["population_size", "age_weight", "gain_weight"])
    def _collect_generation_fitness_kernel(
        self,
        genome_indices: Any,
        age_ticks: Any,
        total_energy_gained: Any,
        population_size: int,
        age_weight: float = 1.0,
        gain_weight: float = 1.0,
    ) -> Any:
        slot_fitness = (
            age_ticks.astype(jnp.float32) * age_weight
            + total_energy_gained.astype(jnp.float32) * gain_weight
        )
        valid = (genome_indices >= 0) & (genome_indices < int(population_size))
        safe_indices = jnp.where(valid, genome_indices, 0).astype(jnp.int32)
        safe_weights = jnp.where(valid, slot_fitness, 0.0).astype(jnp.float32)
        return jnp.bincount(
            safe_indices, weights=safe_weights, length=int(population_size)
        ).astype(jnp.float32)

    def get_species_state_arrays(
        self, species: AgentSpecies
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        buffers = self._buffers(species)
        return (
            buffers.positions,
            buffers.headings_deg,
            buffers.velocities,
            buffers.energies,
            buffers.is_alive,
        )

    def get_sensor_object_arrays(
        self,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:

        prey_buffers = self._buffers(AgentSpecies.PREY)
        predator_buffers = self._buffers(AgentSpecies.PREDATOR)

        centers = jnp.concatenate(
            (self.food_positions, prey_buffers.positions, predator_buffers.positions),
            axis=0,
        ).astype(jnp.float32)
        alive = jnp.concatenate(
            (self.food_is_alive, prey_buffers.is_alive, predator_buffers.is_alive),
            axis=0,
        ).astype(jnp.float32)

        return (
            centers,
            self._sensor_object_radii,
            self._sensor_object_types,
            alive,
            self._sensor_object_indices,
        )

    def get_owner_object_indices(self, species: AgentSpecies) -> np.ndarray:
        if species == AgentSpecies.PREY:
            return self._prey_owner_object_indices
        return self._predator_owner_object_indices

    def render(self, surface: Surface) -> None:

        surface.fill((18, 26, 32))

        world_width = max(float(self.config.screen_width), 1.0)
        world_height = max(float(self.config.screen_height), 1.0)
        surface_width = max(float(surface.get_width()), 1.0)
        surface_height = max(float(surface.get_height()), 1.0)

        scale = min(surface_width / world_width, surface_height / world_height)
        offset_x = (surface_width - world_width * scale) * 0.5
        offset_y = (surface_height - world_height * scale) * 0.5

        food_radius_px = max(1, int(round(float(self.config.food_radius) * scale)))
        prey_radius_world = float(self.config.prey_radius)
        predator_radius_world = float(self.config.predator_radius)
        draw_fov = bool(self.config.draw_fov)
        prey_fov_deg = float(self.config.prey_sensor_fov_deg)
        predator_fov_deg = float(self.config.predator_sensor_fov_deg)
        prey_fov_range = float(self.config.prey_sensor_max_range)
        predator_fov_range = float(self.config.predator_sensor_max_range)

        (
            food_pos_cpu,
            food_alive_cpu,
            prey_pos_cpu,
            prey_head_cpu,
            prey_alive_cpu,
            predator_pos_cpu,
            predator_head_cpu,
            predator_alive_cpu,
        ) = self._get_render_cpu_snapshot()
        for food_slot in np.flatnonzero(food_alive_cpu > 0.5):
            x, y = food_pos_cpu[int(food_slot)]
            sx = int(x * scale + offset_x)
            sy = int(y * scale + offset_y)
            pygame.draw.circle(surface, (90, 200, 120), (sx, sy), food_radius_px)

        for slot in np.flatnonzero(prey_alive_cpu > 0.5):
            idx = int(slot)
            if draw_fov:
                self._draw_fov_wedge(
                    surface=surface,
                    x=float(prey_pos_cpu[idx, 0]),
                    y=float(prey_pos_cpu[idx, 1]),
                    heading_deg=float(prey_head_cpu[idx]),
                    fov_deg=prey_fov_deg,
                    max_range=prey_fov_range,
                    color=(90, 190, 255),
                    scale=scale,
                    offset_x=offset_x,
                    offset_y=offset_y,
                )
            self._draw_agent(
                surface=surface,
                x=float(prey_pos_cpu[idx, 0]),
                y=float(prey_pos_cpu[idx, 1]),
                heading_deg=float(prey_head_cpu[idx]),
                radius=prey_radius_world,
                color=(90, 190, 255),
                scale=scale,
                offset_x=offset_x,
                offset_y=offset_y,
            )

        for slot in np.flatnonzero(predator_alive_cpu > 0.5):
            idx = int(slot)
            if draw_fov:
                self._draw_fov_wedge(
                    surface=surface,
                    x=float(predator_pos_cpu[idx, 0]),
                    y=float(predator_pos_cpu[idx, 1]),
                    heading_deg=float(predator_head_cpu[idx]),
                    fov_deg=predator_fov_deg,
                    max_range=predator_fov_range,
                    color=(230, 95, 75),
                    scale=scale,
                    offset_x=offset_x,
                    offset_y=offset_y,
                )
            self._draw_agent(
                surface=surface,
                x=float(predator_pos_cpu[idx, 0]),
                y=float(predator_pos_cpu[idx, 1]),
                heading_deg=float(predator_head_cpu[idx]),
                radius=predator_radius_world,
                color=(230, 95, 75),
                scale=scale,
                offset_x=offset_x,
                offset_y=offset_y,
            )

    def _get_render_cpu_snapshot(self) -> tuple[np.ndarray, ...]:
        prey_buffers = self._buffers(AgentSpecies.PREY)
        predator_buffers = self._buffers(AgentSpecies.PREDATOR)
        values = (
            self.food_positions,
            self.food_is_alive,
            prey_buffers.positions,
            prey_buffers.headings_deg,
            prey_buffers.is_alive,
            predator_buffers.positions,
            predator_buffers.headings_deg,
            predator_buffers.is_alive,
        )
        return tuple(np.asarray(item) for item in jax.device_get(values))

    def _draw_agent(
        self,
        surface: Surface,
        x: float,
        y: float,
        heading_deg: float,
        radius: float,
        color: tuple[int, int, int],
        scale: float,
        offset_x: float,
        offset_y: float,
    ) -> None:
        heading_rad = math.radians(heading_deg)
        side_left_rad = heading_rad + math.radians(140.0)
        side_right_rad = heading_rad - math.radians(140.0)

        tip_x = x + math.cos(heading_rad) * (radius * 1.6)
        tip_y = y + math.sin(heading_rad) * (radius * 1.6)
        left_x = x + math.cos(side_left_rad) * radius
        left_y = y + math.sin(side_left_rad) * radius
        right_x = x + math.cos(side_right_rad) * radius
        right_y = y + math.sin(side_right_rad) * radius

        pygame.draw.polygon(
            surface,
            color,
            (
                (int(tip_x * scale + offset_x), int(tip_y * scale + offset_y)),
                (int(left_x * scale + offset_x), int(left_y * scale + offset_y)),
                (int(right_x * scale + offset_x), int(right_y * scale + offset_y)),
            ),
        )

    def _draw_fov_wedge(
        self,
        surface: Surface,
        x: float,
        y: float,
        heading_deg: float,
        fov_deg: float,
        max_range: float,
        color: tuple[int, int, int],
        scale: float,
        offset_x: float,
        offset_y: float,
    ) -> None:
        radius_px = max(1, int(round(max_range * scale)))
        center = (int(x * scale + offset_x), int(y * scale + offset_y))
        start_angle = math.radians(heading_deg - (fov_deg * 0.5))
        end_angle = math.radians(heading_deg + (fov_deg * 0.5))
        arc_rect = pygame.Rect(0, 0, radius_px * 2, radius_px * 2)
        arc_rect.center = center

        edge_start = (
            int((x + math.cos(start_angle) * max_range) * scale + offset_x),
            int((y + math.sin(start_angle) * max_range) * scale + offset_y),
        )
        edge_end = (
            int((x + math.cos(end_angle) * max_range) * scale + offset_x),
            int((y + math.sin(end_angle) * max_range) * scale + offset_y),
        )

        fov_color = (color[0], color[1], color[2], 70)
        pygame.draw.line(surface, fov_color, center, edge_start, width=1)
        pygame.draw.line(surface, fov_color, center, edge_end, width=1)
        pygame.draw.arc(surface, fov_color, arc_rect, start_angle, end_angle, width=1)

    def _buffers(self, species: AgentSpecies) -> SpeciesBuffers:
        return self._species_buffers[species]

    @_jit_if_available(
        static_argnames=["prey_age_w", "prey_gain_w", "pred_age_w", "pred_gain_w"]
    )
    def _mega_step_kernel(
        self,
        dt: float,
        food_spawn_accumulator: float,
        food_key,
        prey: SpeciesBuffers,
        pred: SpeciesBuffers,
        food_pos,
        food_vals,
        food_alive,
        prey_init_e: float,
        pred_init_e: float,
        repr_fac: float,
        repr_ratio: float,
        prey_age_w: float,
        prey_gain_w: float,
        pred_age_w: float,
        pred_gain_w: float,
    ):

        p_pos, p_head, p_vel, p_energies, p_alive, p_age = self._physics_step_kernel(
            prey.positions,
            prey.headings_deg,
            prey.velocities,
            prey.energies,
            prey.is_alive,
            prey.age_ticks,
            prey.last_throttle,
            prey.last_turn,
            dt,
            float(self.config.prey_max_speed),
            float(self.config.prey_max_turn_rate_deg),
            float(self.config.prey_base_metabolism),
        )
        prey = prey.replace(
            positions=p_pos,
            headings_deg=p_head,
            velocities=p_vel,
            energies=p_energies,
            is_alive=p_alive,
            age_ticks=p_age,
        )

        pr_pos, pr_head, pr_vel, pr_energies, pr_alive, pr_age = (
            self._physics_step_kernel(
                pred.positions,
                pred.headings_deg,
                pred.velocities,
                pred.energies,
                pred.is_alive,
                pred.age_ticks,
                pred.last_throttle,
                pred.last_turn,
                dt,
                float(self.config.predator_max_speed),
                float(self.config.predator_max_turn_rate_deg),
                float(self.config.predator_base_metabolism),
            )
        )
        pred = pred.replace(
            positions=pr_pos,
            headings_deg=pr_head,
            velocities=pr_vel,
            energies=pr_energies,
            is_alive=pr_alive,
            age_ticks=pr_age,
        )

        p_energies, p_gain, food_alive = self._resolve_prey_food_collisions_kernel(
            prey.positions,
            prey.energies,
            prey.is_alive,
            prey.total_energy_gained,
            food_pos,
            food_vals,
            food_alive,
        )
        prey = prey.replace(energies=p_energies, total_energy_gained=p_gain)

        pr_energies, pr_gain, p_energies, p_alive = (
            self._resolve_predator_prey_collisions_kernel(
                pred.positions,
                pred.headings_deg,
                pred.velocities,
                pred.energies,
                pred.is_alive,
                pred.total_energy_gained,
                prey.positions,
                prey.energies,
                prey.is_alive,
            )
        )
        pred = pred.replace(energies=pr_energies, total_energy_gained=pr_gain)
        prey = prey.replace(energies=p_energies, is_alive=p_alive)

        prey, prey_bank = self._resolve_reproduction_kernel(
            prey,
            prey_init_e,
            repr_fac,
            repr_ratio,
            population_size=self.prey_population_size,
            age_weight=prey_age_w,
            gain_weight=prey_gain_w,
        )
        pred, pred_bank = self._resolve_reproduction_kernel(
            pred,
            pred_init_e,
            repr_fac,
            repr_ratio,
            population_size=self.predator_population_size,
            age_weight=pred_age_w,
            gain_weight=pred_gain_w,
        )

        spawn_rate = float(self.config.food_spawn_per_second)
        new_acc = food_spawn_accumulator + (dt * spawn_rate)
        spawn_cnt = jnp.floor(new_acc).astype(jnp.int32)
        new_acc = new_acc - spawn_cnt.astype(jnp.float32)

        food_pos, food_vals, food_alive, new_food_key = self._spawn_food_kernel(
            food_pos,
            food_vals,
            food_alive,
            spawn_cnt,
            food_key,
            float(self.config.screen_width),
            float(self.config.screen_height),
            float(self.config.food_energy_gain),
        )

        return (
            prey,
            pred,
            food_pos,
            food_vals,
            food_alive,
            new_acc,
            new_food_key,
            prey_bank,
            pred_bank,
        )
