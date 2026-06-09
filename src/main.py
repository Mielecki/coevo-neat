from __future__ import annotations

import argparse
import cProfile
import importlib
import logging
import pstats
from pathlib import Path
from typing import Sequence

from .config import Config
from .simulation_manager import SimulationManager


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:

    parser = argparse.ArgumentParser(description="Predator-prey ALife simulation")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Uruchamia trening bez okna Pygame (render wylaczony).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="Liczba krokow w trybie headless; 0 oznacza brak limitu.",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=0,
        help="Liczba generacji treningu; 0 oznacza brak limitu.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Seed losowosci dla pozycji i kontrolerow.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=60,
        help="Docelowy FPS dla trybu wizualnego.",
    )
    parser.add_argument(
        "--draw-fov",
        action="store_true",
        help="Rysuje kliny FOV agentow w trybie wizualnym.",
    )
    parser.add_argument(
        "--viewport-width",
        type=int,
        default=1280,
        help="Szerokosc okna wizualizacji (nie rozmiar swiata).",
    )
    parser.add_argument(
        "--viewport-height",
        type=int,
        default=720,
        help="Wysokosc okna wizualizacji (nie rozmiar swiata).",
    )
    parser.add_argument(
        "--verbose-lag-logs",
        action="store_true",
        help="Wlacza szczegolowe logi diagnostyczne (smierci i etapy generacji).",
    )
    parser.add_argument(
        "--load-networks",
        type=str,
        default=None,
        help="Sciezka do checkpointu sieci (np. z treningu headless), ktory ma zostac zaladowany.",
    )
    parser.add_argument(
        "--save-networks",
        type=str,
        default=None,
        help="Sciezka zapisu checkpointu sieci przy zamknieciu aplikacji.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Katalog autosave checkpointow.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=25,
        help="Co ile generacji wykonywac autosave (0 wylacza).",
    )
    parser.add_argument(
        "--checkpoint-keep-last",
        type=int,
        default=5,
        help="Ile ostatnich checkpointow gen_* zachowac.",
    )
    parser.add_argument(
        "--no-checkpoint-latest",
        action="store_true",
        help="Wylacza zapis pliku latest.pkl przy autosave.",
    )
    parser.add_argument(
        "--no-checkpoint-best",
        action="store_true",
        help="Wylacza zapis pliku best.pkl przy poprawie fitness.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Wlacza profilowanie cProfile calego przebiegu.",
    )
    parser.add_argument(
        "--profile-output",
        type=str,
        default="tmp/profile.prof",
        help="Sciezka pliku .prof zapisywanego przez cProfile.",
    )
    parser.add_argument(
        "--profile-sort",
        type=str,
        default="cumulative",
        choices=("cumulative", "time", "calls"),
        help="Klucz sortowania raportu profilera wypisywanego na stdout.",
    )
    parser.add_argument(
        "--profile-top",
        type=int,
        default=40,
        help="Liczba wpisow raportu profilera wypisywanych na stdout.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:

    jax_module = importlib.import_module("jax")
    jax_module.config.update("jax_enable_x64", False)

    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose_lag_logs else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = Config(
        headless=args.headless,
        draw_fov=args.draw_fov,
        max_headless_steps=max(args.steps, 0),
        max_generations=max(args.generations, 0),
        random_seed=args.seed,
        fps=max(args.fps, 1),
        viewport_width=max(args.viewport_width, 320),
        viewport_height=max(args.viewport_height, 240),
        verbose_lag_logs=args.verbose_lag_logs,
        load_networks_path=args.load_networks,
        save_networks_path=args.save_networks,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_every_generations=max(args.checkpoint_every, 0),
        checkpoint_keep_last=max(args.checkpoint_keep_last, 0),
        checkpoint_save_latest=not args.no_checkpoint_latest,
        checkpoint_save_best=not args.no_checkpoint_best,
    )
    manager = SimulationManager(config)
    if not args.profile:
        manager.run()
        return

    profiler = cProfile.Profile()
    profiler.enable()
    try:
        manager.run()
    finally:
        profiler.disable()
        output_path = Path(args.profile_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        profiler.dump_stats(str(output_path))

        stats = pstats.Stats(profiler).sort_stats(args.profile_sort)
        stats.print_stats(max(1, args.profile_top))
        print(f"[profile] saved: {output_path}")


if __name__ == "__main__":
    main()
