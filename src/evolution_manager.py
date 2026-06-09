from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Callable

import jax
import numpy as np

jax.config.update("jax_enable_x64", False)

import jax.numpy as jnp
from tensorneat.algorithm import NEAT
from tensorneat.common import ACT, State
from tensorneat.genome import DefaultGenome

from .config import Config
from .enums import AgentSpecies

logger = logging.getLogger(__name__)


class EvolutionManager:

    def __init__(self, config: Config) -> None:
        self.config = config
        self.backend: Any | None = None
        self.current_generation: int = 0
        self._tensorneat_enabled = False
        self._fitness_buffer: dict[AgentSpecies, list[float]] = {
            AgentSpecies.PREY: [0.0] * max(1, self.config.tensorneat_prey_population),
            AgentSpecies.PREDATOR: [0.0]
            * max(1, self.config.tensorneat_predator_population),
        }
        self._algorithms: dict[AgentSpecies, Any] = {}
        self._states: dict[AgentSpecies, Any] = {}
        self._generation_population: dict[AgentSpecies, Any] = {}
        self._transformed_population: dict[AgentSpecies, Any] = {}
        self._batched_forward_fns: dict[
            AgentSpecies, Callable[[Any, Any, Any, Any], Any]
        ] = {}
        self._batched_transform_fns: dict[
            AgentSpecies, Callable[[Any, Any, Any], Any]
        ] = {}
        self._batched_evolve_fns: dict[
            AgentSpecies, Callable[[Any, Any], tuple[Any, Any]]
        ] = {}
        self._last_generation_stats: dict[AgentSpecies, dict[str, float]] = {}

    def initialize_backend(self) -> None:

        requested_device = self._resolve_jax_device(self.config.tensorneat_device)

        available_devices = {device.platform for device in jax.devices()}
        if requested_device not in available_devices:
            raise RuntimeError(
                f"Requested JAX device '{requested_device}' is not available for TensorNEAT"
            )
        logger.info(
            "Selected JAX device '%s' (requested '%s')",
            requested_device,
            self.config.tensorneat_device,
        )
        self.backend = {
            "name": "tensorneat",
            "device": requested_device,
        }
        self._tensorneat_enabled = True

    def create_populations(self) -> None:

        self._algorithms.clear()
        self._states.clear()
        self._generation_population.clear()
        self._transformed_population.clear()
        self._batched_forward_fns.clear()
        self._batched_transform_fns.clear()
        self._batched_evolve_fns.clear()

        if not self._tensorneat_enabled:
            raise RuntimeError("TensorNEAT backend is not initialized")

        activation = self._resolve_activation(self.config.tensorneat_output_activation)

        for species in (AgentSpecies.PREY, AgentSpecies.PREDATOR):
            num_inputs = self._sensor_input_size_for_species(species)
            num_outputs = self.config.tensorneat_output_size
            pop_size = self._population_size_for_species(species)
            genome = DefaultGenome(
                num_inputs=self._sensor_input_size_for_species(species),
                num_outputs=self.config.tensorneat_output_size,
                max_nodes=num_inputs + num_outputs + 64,
                max_conns=256,
                output_transform=activation,
            )

            algorithm = NEAT(
                genome=genome,
                pop_size=pop_size,
                species_size=max(2, self.config.tensorneat_species_size),
                pop_batch_size=self._resolve_pop_batch_size(pop_size),
            )

            species_seed = self.config.random_seed + (
                0 if species == AgentSpecies.PREY else 10_000
            )
            state = algorithm.setup(State(randkey=jax.random.PRNGKey(species_seed)))
            population = algorithm.ask(state)

            self._algorithms[species] = algorithm
            self._states[species] = state
            self._generation_population[species] = population
            self._refresh_transformed_population(species)
            self._fitness_buffer[species] = [0.0] * pop_size

        self.current_generation = 1

    def activate_species_for_bodies(
        self,
        species: AgentSpecies,
        sensor_batch: Any,
        genome_indices: Any,
        is_alive: Any,
    ) -> Any:
        algorithm = self._algorithms.get(species)
        state = self._states.get(species)
        transformed_population = self._transformed_population.get(species)

        batched_forward = self._get_batched_forward_fn(species, algorithm)

        return batched_forward(
            state, transformed_population, genome_indices, is_alive, sensor_batch
        )

    def set_generation_fitness(
        self, species: AgentSpecies, fitness_values: Any
    ) -> None:

        self._fitness_buffer[species] = fitness_values

    def save_checkpoint(self, file_path: str) -> bool:

        if not self._tensorneat_enabled:
            logger.warning("Checkpoint save skipped: tensorneat backend inactive")
            return False

        checkpoint = self._build_checkpoint_payload()
        target = Path(file_path)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("wb") as file_handle:
                pickle.dump(checkpoint, file_handle)
            return True
        except Exception as exc:
            logger.warning(
                "Failed to save checkpoint %s: %s", file_path, type(exc).__name__
            )
            return False

    def load_checkpoint(self, file_path: str) -> bool:

        if not self._tensorneat_enabled:
            logger.warning("Checkpoint load skipped: tensorneat backend inactive")
            return False

        source = Path(file_path)
        if not source.exists():
            logger.warning("Checkpoint file does not exist: %s", file_path)
            return False

        try:
            with source.open("rb") as file_handle:
                checkpoint = pickle.load(file_handle)
        except Exception as exc:
            logger.warning(
                "Failed to load checkpoint %s: %s", file_path, type(exc).__name__
            )
            return False

        try:
            return self._apply_checkpoint_payload(checkpoint)
        except Exception as exc:
            logger.warning(
                "Failed to apply checkpoint %s: %s", file_path, type(exc).__name__
            )
            return False

    def evolve_step(self) -> None:

        if self.config.verbose_lag_logs:
            backend_name = (self.backend or {}).get("name", "unknown")
            backend_device = (self.backend or {}).get("device", "unknown")
            logger.info(
                "evolve_step start gen=%d backend=%s device=%s",
                self.current_generation,
                backend_name,
                backend_device,
            )

        for species in (AgentSpecies.PREY, AgentSpecies.PREDATOR):
            self._last_generation_stats[species] = self._compute_fitness_stats(
                self._fitness_buffer[species]
            )

        if not self._tensorneat_enabled:
            raise RuntimeError("TensorNEAT backend is not initialized")

        for species in (AgentSpecies.PREY, AgentSpecies.PREDATOR):
            algorithm = self._algorithms[species]
            state = self._states[species]
            if self.config.verbose_lag_logs:
                population = self._generation_population.get(species)
                population_size = (
                    int(population[0].shape[0]) if population is not None else 0
                )
                logger.info(
                    "evolve_step species start gen=%d species=%s population=%d",
                    self.current_generation,
                    species.value,
                    population_size,
                )

            fitness_array = jnp.asarray(
                self._fitness_buffer[species], dtype=jnp.float32
            )
            evolve_fn = self._get_batched_evolve_fn(species, algorithm)
            next_state, next_population = evolve_fn(state, fitness_array)

            self._states[species] = next_state
            self._generation_population[species] = next_population
            self._refresh_transformed_population(species)
            self._fitness_buffer[species] = [0.0] * int(next_population[0].shape[0])

        self.current_generation += 1

    def get_last_generation_stats(self) -> dict[AgentSpecies, dict[str, float]]:

        return self._last_generation_stats

    def _population_size_for_species(self, species: AgentSpecies) -> int:

        if self._tensorneat_enabled and species in self._generation_population:
            population = self._generation_population[species]
            return int(population[0].shape[0])

        if species == AgentSpecies.PREY:
            return max(1, self.config.tensorneat_prey_population)
        return max(1, self.config.tensorneat_predator_population)

    def _sensor_input_size_for_species(self, species: AgentSpecies) -> int:
        if species == AgentSpecies.PREY:
            return self.config.prey_sensor_input_size
        return self.config.predator_sensor_input_size

    def _resolve_activation(self, activation_name: str) -> Any:

        if ACT is None:
            return None

        normalized = activation_name.strip().lower()
        activation = getattr(ACT, normalized, None)
        if activation is not None:
            return activation
        return ACT.tanh

    def _resolve_jax_device(self, requested: str) -> str:

        normalized = requested.strip().lower()
        if normalized == "cuda":
            return "gpu"
        if normalized in {"gpu", "tpu"}:
            return normalized
        raise ValueError(f"Unsupported TensorNEAT device '{requested}'")

    def _resolve_pop_batch_size(self, pop_size: int) -> int | None:

        configured = max(1, self.config.tensorneat_batch_size)
        candidate = min(pop_size, configured)
        if pop_size % candidate == 0:
            return candidate
        return pop_size

    def _compute_fitness_stats(self, values: Any) -> dict[str, float]:

        return {
            "min": float(jnp.min(values)),
            "max": float(jnp.max(values)),
            "mean": float(jnp.mean(values)),
        }

    def _refresh_transformed_population(self, species: AgentSpecies) -> None:
        if not self._tensorneat_enabled:
            return

        algorithm = self._algorithms.get(species)
        state = self._states.get(species)
        population = self._generation_population.get(species)

        if algorithm is None or state is None or population is None:
            return

        pop_nodes, pop_conns = population
        transform_fn = self._get_batched_transform_fn(species, algorithm)
        self._transformed_population[species] = transform_fn(
            state, pop_nodes, pop_conns
        )

    def _build_checkpoint_payload(self) -> dict[str, Any]:

        species_payload: dict[str, Any] = {}
        for species in (AgentSpecies.PREY, AgentSpecies.PREDATOR):
            population = self._generation_population.get(species)
            state = self._states.get(species)
            species_payload[species.value] = {
                "population": self._to_host(population),
                "state": self._to_host(state),
                "fitness_buffer": list(self._fitness_buffer.get(species, [])),
            }

        return {
            "version": 1,
            "current_generation": self.current_generation,
            "species": species_payload,
            "backend": self.backend,
            "tensorneat_enabled": self._tensorneat_enabled,
        }

    def _apply_checkpoint_payload(self, checkpoint: dict[str, Any]) -> bool:

        species_payload_raw = checkpoint.get("species")
        if not isinstance(species_payload_raw, dict):
            return False

        loaded_any = False
        for species in (AgentSpecies.PREY, AgentSpecies.PREDATOR):
            raw_entry = species_payload_raw.get(species.value)
            if not isinstance(raw_entry, dict):
                continue

            population_raw = raw_entry.get("population")
            if population_raw is None:
                continue
            if (
                not isinstance(population_raw, (tuple, list))
                or len(population_raw) != 2
            ):
                continue

            population = (np.asarray(population_raw[0]), np.asarray(population_raw[1]))
            if population[0].ndim == 0:
                continue

            self._generation_population[species] = population

            state_raw = raw_entry.get("state")
            if state_raw is not None and species in self._states:
                self._states[species] = state_raw

            population_size = int(population[0].shape[0])
            self._fitness_buffer[species] = [0.0] * max(1, population_size)
            self._refresh_transformed_population(species)
            loaded_any = True

        if not loaded_any:
            return False

        try:
            loaded_generation = int(
                checkpoint.get("current_generation", self.current_generation)
            )
            self.current_generation = max(1, loaded_generation)
        except (TypeError, ValueError):
            pass

        return True

    def _to_host(self, value: Any) -> Any:

        return value

    def _get_batched_forward_fn(
        self,
        species: AgentSpecies,
        algorithm: Any,
    ) -> Callable[[Any, Any, Any, Any, Any], Any]:
        cached = self._batched_forward_fns.get(species)
        if cached is not None:
            return cached

        pop_size = self._population_size_for_species(species)

        def _forward_batch(state, transformed_pop, indices, alive, input_batch):
            valid_mask = (indices >= 0) & (indices < pop_size) & (alive > 0.5)
            safe_idx = jnp.where(valid_mask, indices, 0)
            gathered = jax.tree_util.tree_map(lambda x: x[safe_idx], transformed_pop)
            controls = jax.vmap(lambda t, i: algorithm.forward(state, t, i))(
                gathered, input_batch
            )
            return jnp.where(valid_mask[:, None], controls, jnp.zeros_like(controls))

        compiled = jax.jit(_forward_batch)
        self._batched_forward_fns[species] = compiled
        return compiled

    def _get_batched_transform_fn(
        self,
        species: AgentSpecies,
        algorithm: Any,
    ) -> Callable[[Any, Any, Any], Any]:

        cached = self._batched_transform_fns.get(species)
        if cached is not None:
            return cached

        def _transform_batch(state: Any, pop_nodes: Any, pop_conns: Any) -> Any:
            return jax.vmap(
                lambda nodes, conns: algorithm.transform(state, (nodes, conns)),
                in_axes=(0, 0),
            )(pop_nodes, pop_conns)

        compiled = jax.jit(_transform_batch)

        self._batched_transform_fns[species] = compiled
        return compiled

    def _get_batched_evolve_fn(
        self,
        species: AgentSpecies,
        algorithm: Any,
    ) -> Callable[[Any, Any], tuple[Any, Any]]:

        cached = self._batched_evolve_fns.get(species)
        if cached is not None:
            return cached

        def _evolve_step(state: Any, fitness: Any) -> tuple[Any, Any]:
            next_state = algorithm.tell(state, fitness)
            next_population = algorithm.ask(next_state)
            return next_state, next_population

        compiled = jax.jit(_evolve_step)

        self._batched_evolve_fns[species] = compiled
        return compiled

    def export_best_genome_to_json(self, species: AgentSpecies, filepath: str):
        if not self._tensorneat_enabled:
            return

        fitness_array = jnp.asarray(self._fitness_buffer[species])
        best_idx = int(jnp.argmax(fitness_array))
        pop_nodes, pop_conns = self._generation_population[species]

        nodes = jax.device_get(pop_nodes[best_idx])
        conns = jax.device_get(pop_conns[best_idx])
        state = self._states[species]

        algorithm = self._algorithms.get(species)
        if algorithm is None or not hasattr(algorithm, "genome"):
            logger.error(
                f"Cannot export network: Genome not found for species {species.value}"
            )
            return

        genome = algorithm.genome

        net_dict = genome.network_dict(state, nodes, conns)

        export_data = {
            "generation": self.current_generation,
            "species": species.value,
            "nodes": [],
            "edges": [],
            "layers": [],
        }

        for layer_idx, layer_nodes in enumerate(net_dict["topo_layers"]):
            for node_id in layer_nodes:

                node_type = "hidden"
                if node_id in genome.input_idx:
                    node_type = "input"
                elif node_id in genome.output_idx:
                    node_type = "output"

                export_data["nodes"].append(
                    {
                        "id": int(node_id),
                        "label": f"N{node_id}",
                        "layer": layer_idx,
                        "type": node_type,
                    }
                )

        for (src, dest), conn_gene in net_dict["conns"].items():

            if isinstance(conn_gene, dict):
                weight = float(conn_gene.get("weight", 1.0))
                enabled_val = conn_gene.get("enabled", True)
            else:

                weight = float(conn_gene[0])
                enabled_val = conn_gene[3] if len(conn_gene) > 3 else 1.0

            is_enabled = (
                bool(enabled_val)
                if isinstance(enabled_val, bool)
                else float(enabled_val) > 0.5
            )

            if is_enabled:
                export_data["edges"].append(
                    {
                        "from": int(src),
                        "to": int(dest),
                        "weight": weight,
                        "enabled": True,
                    }
                )

        target = Path(filepath)
        target.parent.mkdir(parents=True, exist_ok=True)

        with open(target, "w") as f:
            json.dump(export_data, f, indent=2)
