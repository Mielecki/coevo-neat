from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax

jax.config.update("jax_enable_x64", False)
import jax.numpy as jnp
import numpy as np

from .config import Config
from .enums import AgentSpecies

if TYPE_CHECKING:
    from .environment import Environment


def _jit_if_available(func: Any) -> Any:
    return jax.jit(func, static_argnums=(0,))


class SensorSystem:

    def __init__(self, config: Config) -> None:
        self.config = config
        self._relative_angles_cache: dict[tuple[int, float], np.ndarray] = {}
        self._cached_environment_id: int | None = None
        self._cached_detection_arrays: (
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None
        ) = None

    def begin_frame(self, environment: "Environment") -> None:

        self._cached_environment_id = id(environment)
        self._cached_detection_arrays = environment.get_sensor_object_arrays()

    def end_frame(self) -> None:

        self._cached_environment_id = None
        self._cached_detection_arrays = None

    def sample_for_species(
        self, species: AgentSpecies, environment: "Environment"
    ) -> Any:

        positions, headings_deg, _, energies, owner_alive = (
            environment.get_species_state_arrays(species)
        )
        owner_object_indices = environment.get_owner_object_indices(species)
        object_centers, object_radii, object_types, object_alive, object_indices = (
            self._get_detection_arrays(environment)
        )

        fov_deg, max_range = self._get_cone_params_for_species(species)
        relative_angles_deg = self._build_relative_angles_for_species(species, fov_deg)

        if species == AgentSpecies.PREY:
            max_energy = float(
                self.config.prey_initial_energy * self.config.reproduction_energy_factor
            )
        else:
            max_energy = float(
                self.config.predator_initial_energy
                * self.config.reproduction_energy_factor
            )

        return self._raycast_kernel(
            positions.astype(jnp.float32),
            headings_deg.astype(jnp.float32),
            owner_object_indices.astype(jnp.int32),
            owner_alive.astype(jnp.float32),
            object_centers.astype(jnp.float32),
            object_radii.astype(jnp.float32),
            object_types.astype(jnp.int32),
            object_alive.astype(jnp.float32),
            object_indices.astype(jnp.int32),
            jnp.asarray(relative_angles_deg, dtype=jnp.float32),
            float(fov_deg),
            float(max_range),
            energies.astype(jnp.float32),
            max_energy,
        )

    @_jit_if_available
    def _raycast_kernel(
        self,
        positions,
        headings_deg,
        owner_object_indices,
        owner_alive,
        object_centers,
        object_radii,
        object_types,
        object_alive,
        object_indices,
        relative_angles_deg,
        fov_deg,
        max_range,
        energies,
        max_energy,
    ) -> Any:
        abs_angles_rad = jnp.deg2rad(
            headings_deg[:, None] + relative_angles_deg[None, :]
        )
        directions = jnp.stack(
            (jnp.cos(abs_angles_rad), jnp.sin(abs_angles_rad)), axis=-1
        )

        pos_sq = jnp.sum(positions * positions, axis=1)[:, None]
        obj_sq = jnp.sum(object_centers * object_centers, axis=1)[None, :]
        dot_prod = jnp.dot(positions, object_centers.T)

        dist_sq = jnp.maximum(pos_sq + obj_sq - 2.0 * dot_prod, 0.0)

        max_dist_plus_r = max_range + jnp.max(object_radii)
        is_candidate = dist_sq < (max_dist_plus_r**2)

        dir_dot_obj = jnp.dot(directions, object_centers.T)
        dir_dot_pos = jnp.einsum("ard,ad->ar", directions, positions)
        t_proj = dir_dot_obj - dir_dot_pos[:, :, None]

        h_sq = dist_sq[:, None, :] - (t_proj * t_proj)
        r_sq = (object_radii * object_radii)[None, None, :]

        intersect_mask = (h_sq <= r_sq) & (t_proj > 0) & is_candidate[:, None, :]
        intersect_mask &= object_alive[None, None, :] > 0.5
        intersect_mask &= (
            object_indices[None, None, :] != owner_object_indices[:, None, None]
        )

        delta_t = jnp.sqrt(jnp.maximum(r_sq - h_sq, 0.0))
        t_final = t_proj - delta_t

        final_hits = jnp.where(
            (t_final > 0) & (t_final <= max_range) & intersect_mask, t_final, jnp.inf
        )
        min_obj_dist = jnp.min(final_hits, axis=2)
        nearest_idx = jnp.argmin(final_hits, axis=2)
        detected_obj_types = object_types[nearest_idx]

        wall_distances = self._wall_distance_batch(positions, directions, max_range)
        hit_wall = wall_distances < min_obj_dist
        final_min_dist = jnp.where(hit_wall, wall_distances, min_obj_dist)

        wall_type = jnp.asarray(1, dtype=jnp.int32)
        final_types = jnp.where(hit_wall, wall_type, detected_obj_types)

        detected_valid = jnp.isfinite(final_min_dist)
        distance_norm = jnp.where(
            detected_valid, jnp.clip(final_min_dist / max_range, 0.0, 1.0), 1.0
        )

        alive_mask = owner_alive[:, None] > 0.5
        distance_norm = jnp.where(alive_mask, distance_norm, 1.0)

        is_wall = (final_types == 1) & detected_valid
        is_food = (final_types == 2) & detected_valid
        is_prey = (final_types == 3) & detected_valid
        is_pred = (final_types == 4) & detected_valid

        dist_wall = jnp.where(is_wall, distance_norm, 1.0)
        dist_food = jnp.where(is_food, distance_norm, 1.0)
        dist_prey = jnp.where(is_prey, distance_norm, 1.0)
        dist_pred = jnp.where(is_pred, distance_norm, 1.0)

        dist_wall = jnp.where(alive_mask, dist_wall, 1.0)
        dist_food = jnp.where(alive_mask, dist_food, 1.0)
        dist_prey = jnp.where(alive_mask, dist_prey, 1.0)
        dist_pred = jnp.where(alive_mask, dist_pred, 1.0)

        features = jnp.stack((dist_wall, dist_food, dist_prey, dist_pred), axis=-1)

        ray_features = features.reshape(positions.shape[0], -1)

        alive_mask = owner_alive[:, None] > 0.5
        energy_norm = jnp.clip(energies[:, None] / max_energy, 0.0, 1.0)

        energy_norm = jnp.where(alive_mask, energy_norm, 0.0)

        return jnp.concatenate((ray_features, energy_norm), axis=-1)

    def _wall_distance_batch(
        self, positions: Any, directions: Any, max_range: Any
    ) -> Any:

        width = float(self.config.screen_width)
        height = float(self.config.screen_height)

        ox = positions[:, None, 0]
        oy = positions[:, None, 1]
        dx = directions[:, :, 0]
        dy = directions[:, :, 1]

        inf = jnp.asarray(jnp.inf, dtype=jnp.float32)
        eps = jnp.asarray(1e-8, dtype=jnp.float32)

        tx_pos = jnp.where(dx > eps, (width - ox) / dx, inf)
        y_tx_pos = oy + tx_pos * dy
        tx_pos = jnp.where(
            (tx_pos >= 0.0)
            & (tx_pos <= max_range)
            & (y_tx_pos >= 0.0)
            & (y_tx_pos <= height),
            tx_pos,
            inf,
        )

        tx_neg = jnp.where(dx < -eps, (0.0 - ox) / dx, inf)
        y_tx_neg = oy + tx_neg * dy
        tx_neg = jnp.where(
            (tx_neg >= 0.0)
            & (tx_neg <= max_range)
            & (y_tx_neg >= 0.0)
            & (y_tx_neg <= height),
            tx_neg,
            inf,
        )

        ty_pos = jnp.where(dy > eps, (height - oy) / dy, inf)
        x_ty_pos = ox + ty_pos * dx
        ty_pos = jnp.where(
            (ty_pos >= 0.0)
            & (ty_pos <= max_range)
            & (x_ty_pos >= 0.0)
            & (x_ty_pos <= width),
            ty_pos,
            inf,
        )

        ty_neg = jnp.where(dy < -eps, (0.0 - oy) / dy, inf)
        x_ty_neg = ox + ty_neg * dx
        ty_neg = jnp.where(
            (ty_neg >= 0.0)
            & (ty_neg <= max_range)
            & (x_ty_neg >= 0.0)
            & (x_ty_neg <= width),
            ty_neg,
            inf,
        )

        return jnp.minimum(
            jnp.minimum(tx_pos, tx_neg), jnp.minimum(ty_pos, ty_neg)
        ).astype(jnp.float32)

    def _build_relative_angles_for_species(
        self, species: AgentSpecies, cone_fov_deg: float
    ) -> np.ndarray:
        ray_count = self._get_ray_count_for_species(species)
        cache_key = (ray_count, float(cone_fov_deg))
        cached = self._relative_angles_cache.get(cache_key)
        if cached is not None:
            return cached

        if ray_count == 1:
            angles = np.asarray([0.0], dtype=np.float32)
        else:
            angles = np.linspace(
                -cone_fov_deg * 0.5, cone_fov_deg * 0.5, num=ray_count, dtype=np.float32
            )

        self._relative_angles_cache[cache_key] = jnp.array(angles, dtype=jnp.float32)
        return angles

    def _get_detection_arrays(
        self,
        environment: "Environment",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if (
            self._cached_environment_id == id(environment)
            and self._cached_detection_arrays is not None
        ):
            return self._cached_detection_arrays
        return environment.get_sensor_object_arrays()

    def _get_cone_params_for_species(
        self, species: AgentSpecies
    ) -> tuple[float, float]:
        if species == AgentSpecies.PREY:
            fov = self.config.prey_sensor_fov_deg
            max_range = self.config.prey_sensor_max_range
        else:
            fov = self.config.predator_sensor_fov_deg
            max_range = self.config.predator_sensor_max_range

        return max(float(fov), 1.0), max(float(max_range), 1.0)

    def _get_ray_count_for_species(self, species: AgentSpecies) -> int:
        if species == AgentSpecies.PREY:
            return max(1, int(self.config.prey_sensor_ray_count))
        return max(1, int(self.config.predator_sensor_ray_count))
