from __future__ import annotations

import logging
from pathlib import Path

import jax.numpy as jnp
import pygame
from tqdm import tqdm

from src.utils import AsyncStatsWriter, AsyncTickLogger

from .config import Config
from .enums import AgentSpecies
from .environment import Environment
from .evolution_manager import EvolutionManager
from .sensor_system import SensorSystem

logger = logging.getLogger(__name__)


class SimulationManager:

    def __init__(self, config: Config) -> None:
        self.config = config
        self.environment = Environment(config)
        self.sensor_system = SensorSystem(config)
        self.evolution_manager = EvolutionManager(config)
        self.screen: pygame.Surface | None = None
        self.clock: pygame.time.Clock | None = None
        self.font: pygame.font.Font | None = None
        self.is_running = False
        self.tick_index = 0
        self.generation_tick_index = 0
        self.completed_generations = 0
        self._headless_progress: tqdm | None = None
        self._best_generation_score: float | None = None

        self.tick_logger = AsyncTickLogger("web/data/ticks")
        self.stats_writer = AsyncStatsWriter("web/stats.json")

    def setup(self) -> None:

        if not self.config.headless:
            pygame.init()
            self.screen = pygame.display.set_mode(
                (self.config.viewport_width, self.config.viewport_height)
            )
            pygame.display.set_caption("Predator-Prey ALife")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 24)

        self.environment.reset()
        self.evolution_manager.initialize_backend()
        self.evolution_manager.create_populations()
        if self.config.load_networks_path:
            loaded = self.evolution_manager.load_checkpoint(
                self.config.load_networks_path
            )
            if loaded:
                logger.warning(
                    "Loaded networks from %s", self.config.load_networks_path
                )
            else:
                logger.warning(
                    "Failed to load networks from %s", self.config.load_networks_path
                )

        self.completed_generations = 0
        self._start_generation()

        self.tick_index = 0
        self.is_running = True

    def run(self) -> None:

        self.setup()
        render_skip = 1
        self._init_headless_progress_bar()
        dt = self.config.fixed_dt
        try:
            while self.is_running:
                if not self.config.headless:
                    self.handle_events()
                    self.clock.tick(self.config.fps)

                self.tick_index += 1
                self.tick(dt=dt, tick_index=self.tick_index)
                self._advance_headless_progress_bar()

                if self.config.verbose_lag_logs and (self.tick_index % 100) == 0:
                    logger.info(
                        f"tick {self.tick_index} gen {self.evolution_manager.current_generation} gen_tick {self.generation_tick_index} prey {self.environment.get_alive_count(AgentSpecies.PREY)} predators {self.environment.get_alive_count(AgentSpecies.PREDATOR)} food {self.environment.get_food_count()}"
                    )

                if not self.config.headless and (self.tick_index % render_skip == 0):
                    self.render()

                if self.config.headless and self.config.max_headless_steps > 0:
                    if self.tick_index >= self.config.max_headless_steps:
                        self.is_running = False
        finally:
            self._close_headless_progress_bar()
            self.tick_logger.stop()
            self.shutdown()

    def handle_events(self) -> None:

        if self.config.headless:
            return

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.is_running = False
            elif event.type == pygame.KEYDOWN and event.key in (
                pygame.K_ESCAPE,
                pygame.K_q,
            ):
                self.is_running = False

    def tick(self, dt: float, tick_index: int) -> None:

        self.environment.update(
            dt=dt,
            tick_index=tick_index,
            sensor_system=self.sensor_system,
            evolution_manager=self.evolution_manager,
        )

        self.generation_tick_index += 1

        prey_cnt = int(self.environment.get_alive_count(AgentSpecies.PREY))
        pred_cnt = int(self.environment.get_alive_count(AgentSpecies.PREDATOR))
        gen = self.evolution_manager.current_generation

        self.tick_logger.log(gen, tick_index, prey_cnt, pred_cnt)

        if self._is_generation_finished():
            self._finalize_generation_and_advance()

    def render(self) -> None:

        if self.config.headless or self.screen is None:
            return

        self.environment.render(self.screen)
        self._draw_hud()
        pygame.display.flip()

    def shutdown(self) -> None:

        self.is_running = False
        if self.config.save_networks_path:
            saved = self.evolution_manager.save_checkpoint(
                self.config.save_networks_path
            )
            if saved:
                logger.warning("Saved networks to %s", self.config.save_networks_path)
            else:
                logger.warning(
                    "Failed to save networks to %s", self.config.save_networks_path
                )

        if not self.config.headless:
            pygame.quit()

    def _start_generation(self) -> None:

        self.generation_tick_index = 0
        generation_seed = (
            self.config.random_seed
            + max(0, self.evolution_manager.current_generation) * 1009
        )
        self.environment.seed_generation(generation_seed)

    def _is_generation_finished(self) -> bool:

        if self.generation_tick_index >= max(1, self.config.generation_max_ticks):
            return True

        if self.generation_tick_index % 100 != 0:
            return False

        if self.environment.get_alive_count(AgentSpecies.PREDATOR) == 0:
            return True

        if self.environment.get_alive_count(AgentSpecies.PREY) == 0:
            return True

        return False

    def _finalize_generation_and_advance(self) -> None:

        prey_fitness = self.environment.collect_generation_fitness(AgentSpecies.PREY)
        predator_fitness = self.environment.collect_generation_fitness(
            AgentSpecies.PREDATOR
        )

        prey_alive = int(self.environment.get_alive_count(AgentSpecies.PREY))
        pred_alive = int(self.environment.get_alive_count(AgentSpecies.PREDATOR))

        prey_mean = float(jnp.mean(jnp.asarray(prey_fitness)))
        pred_mean = float(jnp.mean(jnp.asarray(predator_fitness)))

        gen = self.evolution_manager.current_generation

        self.stats_writer.log(
            {
                "generation": gen,
                "prey_alive": prey_alive,
                "predator_alive": pred_alive,
                "prey_fitness": prey_mean,
                "predator_fitness": pred_mean,
            }
        )

        self.evolution_manager.set_generation_fitness(AgentSpecies.PREY, prey_fitness)
        self.evolution_manager.set_generation_fitness(
            AgentSpecies.PREDATOR, predator_fitness
        )

        self.evolution_manager.export_best_genome_to_json(
            AgentSpecies.PREDATOR, "web/predator_latest.json"
        )
        self.evolution_manager.export_best_genome_to_json(
            AgentSpecies.PREY, "web/prey_latest.json"
        )

        self.evolution_manager.export_best_genome_to_json(
            AgentSpecies.PREDATOR, f"web/history/predator_gen_{gen:06d}.json"
        )
        self.evolution_manager.export_best_genome_to_json(
            AgentSpecies.PREY, f"web/history/prey_gen_{gen:06d}.json"
        )

        self.evolution_manager.evolve_step()

        self.completed_generations += 1
        self._print_generation_fitness()
        self._maybe_save_generation_checkpoint()

        if (
            self.config.max_generations > 0
            and self.completed_generations >= self.config.max_generations
        ):
            self.is_running = False
            return

        self._start_generation()

    def _maybe_save_generation_checkpoint(self) -> None:
        every = max(0, int(self.config.checkpoint_every_generations))
        if every <= 0 or (self.completed_generations % every) != 0:
            return

        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        generation = int(self.evolution_manager.current_generation)

        generation_path = checkpoint_dir / f"gen_{generation:06d}.pkl"
        if self.evolution_manager.save_checkpoint(str(generation_path)):
            logger.warning("Autosaved checkpoint: %s", generation_path)
            self._prune_generation_checkpoints(checkpoint_dir)
        else:
            logger.warning("Autosave failed for checkpoint: %s", generation_path)
            return

        if self.config.checkpoint_save_latest:
            latest_path = checkpoint_dir / "latest.pkl"
            if self.evolution_manager.save_checkpoint(str(latest_path)):
                logger.warning("Updated latest checkpoint: %s", latest_path)

        if self.config.checkpoint_save_best:
            score = self._current_generation_score()
            if (
                self._best_generation_score is None
                or score > self._best_generation_score
            ):
                best_path = checkpoint_dir / "best.pkl"
                if self.evolution_manager.save_checkpoint(str(best_path)):
                    self._best_generation_score = score
                    logger.warning(
                        "Updated best checkpoint: %s (score=%.3f)", best_path, score
                    )

    def _current_generation_score(self) -> float:
        stats = self.evolution_manager.get_last_generation_stats()
        prey_max = float((stats.get(AgentSpecies.PREY) or {}).get("max", 0.0))
        predator_max = float((stats.get(AgentSpecies.PREDATOR) or {}).get("max", 0.0))
        return prey_max + predator_max

    def _prune_generation_checkpoints(self, checkpoint_dir: Path) -> None:
        keep_last = max(0, int(self.config.checkpoint_keep_last))
        if keep_last <= 0:
            return

        generation_files = sorted(checkpoint_dir.glob("gen_*.pkl"))
        if len(generation_files) <= keep_last:
            return

        to_delete = generation_files[: len(generation_files) - keep_last]
        for file_path in to_delete:
            try:
                file_path.unlink()
            except OSError:
                logger.warning("Failed to remove old checkpoint: %s", file_path)

    def _print_generation_fitness(self) -> None:

        stats = self.evolution_manager.get_last_generation_stats()
        prey = stats.get(AgentSpecies.PREY, {"min": 0.0, "mean": 0.0, "max": 0.0})
        predator = stats.get(
            AgentSpecies.PREDATOR, {"min": 0.0, "mean": 0.0, "max": 0.0}
        )
        print(
            (
                f"[gen {self.evolution_manager.current_generation}] "
                f"prey fit min/mean/max: {prey['min']:.2f}/{prey['mean']:.2f}/{prey['max']:.2f} | "
                f"pred fit min/mean/max: {predator['min']:.2f}/{predator['mean']:.2f}/{predator['max']:.2f}"
            )
        )

    def _draw_hud(self) -> None:

        if self.screen is None or self.font is None:
            return

        lines = [
            f"tick: {self.tick_index}",
            f"generation: {self.evolution_manager.current_generation}",
            f"gen_tick: {self.generation_tick_index}/{self.config.generation_max_ticks}",
            f"prey: {self.environment.get_alive_count(AgentSpecies.PREY)}",
            f"predators: {self.environment.get_alive_count(AgentSpecies.PREDATOR)}",
            f"food: {self.environment.get_food_count()}",
            "mode: visual",
        ]

        stats = self.evolution_manager.get_last_generation_stats()
        prey_stats = stats.get(AgentSpecies.PREY)
        predator_stats = stats.get(AgentSpecies.PREDATOR)
        if prey_stats is not None:
            lines.append(f"prey_fit max: {prey_stats['max']:.1f}")
        if predator_stats is not None:
            lines.append(f"pred_fit max: {predator_stats['max']:.1f}")

        panel_height = 12 + len(lines) * 22
        pygame.draw.rect(
            self.screen,
            (0, 0, 0),
            pygame.Rect(12, 12, 220, panel_height),
            border_radius=8,
        )
        pygame.draw.rect(
            self.screen,
            (40, 60, 70),
            pygame.Rect(12, 12, 220, panel_height),
            width=1,
            border_radius=8,
        )

        y = 20
        for line in lines:
            text_surface = self.font.render(line, True, (225, 235, 240))
            self.screen.blit(text_surface, (20, y))
            y += 22

    def _init_headless_progress_bar(self) -> None:
        if not self.config.headless or self.config.max_headless_steps <= 0:
            self._headless_progress = None
            return

        self._headless_progress = tqdm(
            total=self.config.max_headless_steps,
            desc="headless",
            unit="tick",
            dynamic_ncols=True,
            leave=True,
        )

    def _advance_headless_progress_bar(self) -> None:
        if self._headless_progress is None:
            return

        self._headless_progress.update(1)

        if self.tick_index % 100 == 0:
            self._headless_progress.set_postfix(
                gen=self.evolution_manager.current_generation,
                prey=self.environment.get_alive_count(AgentSpecies.PREY),
                pred=self.environment.get_alive_count(AgentSpecies.PREDATOR),
                food=self.environment.get_food_count(),
                refresh=False,
            )

    def _close_headless_progress_bar(self) -> None:
        if self._headless_progress is None:
            return

        self._headless_progress.close()
        self._headless_progress = None
