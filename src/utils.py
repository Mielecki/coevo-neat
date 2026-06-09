import queue
import threading
from pathlib import Path


class AsyncTickLogger:
    def __init__(self, base_dir="web/ticks"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._queue = queue.Queue()
        self._current_gen = -1
        self._file_handle = None

        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        while True:
            task = self._queue.get()
            if task is None:
                break

            gen, tick, prey, pred = task

            if gen != self._current_gen:
                if self._file_handle:
                    self._file_handle.close()

                self._current_gen = gen
                file_path = self.base_dir / f"ticks_gen_{gen:06d}.csv"

                self._file_handle = open(file_path, "w")
                self._file_handle.write("tick,prey_alive,predator_alive\n")

            self._file_handle.write(f"{tick},{prey},{pred}\n")

            if tick % 100 == 0:
                self._file_handle.flush()

            self._queue.task_done()

        if self._file_handle:
            self._file_handle.close()

    def log(self, gen: int, tick: int, prey: int, pred: int):
        self._queue.put((gen, tick, prey, pred))

    def stop(self):
        self._queue.put(None)
        self._thread.join()


import json


class AsyncStatsWriter:
    def __init__(self, filepath="web/stats.json"):
        self.filepath = Path(filepath)
        self._queue = queue.Queue()

        if not self.filepath.exists():
            self.filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(self.filepath, "w") as f:
                json.dump([], f)

        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        while True:
            item = self._queue.get()
            if item is None:
                break

            try:
                with open(self.filepath, "r") as f:
                    data = json.load(f)

                data.append(item)

                with open(self.filepath, "w") as f:
                    json.dump(data, f, indent=2)
            except Exception as e:
                print(f"Błąd zapisu stats.json: {e}")
            finally:
                self._queue.task_done()

    def log(self, data: dict):
        self._queue.put(data)

    def stop(self):
        self._queue.put(None)
        self._thread.join()
