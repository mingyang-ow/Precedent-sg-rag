from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def empty_mode_state() -> dict[str, Any]:
    return {
        "queries_completed": 0,
        "observations": [],
        "warm_observations": [],
        "latencies_ms": [],
        "warm_latencies_ms": [],
        "relevant_counts": [],
        "warm_relevant_counts": [],
    }


def validate_mode_state(state: dict[str, Any]) -> None:
    completed = state.get("queries_completed")
    if not isinstance(completed, int) or completed < 0:
        raise ValueError("checkpoint has an invalid completed-query count")
    aligned_keys = ("observations", "latencies_ms", "relevant_counts")
    if any(not isinstance(state.get(key), list) for key in aligned_keys):
        raise ValueError("checkpoint has invalid all-query observations")
    if any(len(state[key]) != completed for key in aligned_keys):
        raise ValueError("checkpoint all-query observations are misaligned")
    warm_keys = ("warm_observations", "warm_latencies_ms", "warm_relevant_counts")
    if any(not isinstance(state.get(key), list) for key in warm_keys):
        raise ValueError("checkpoint has invalid warm-query observations")
    warm_count = len(state["warm_observations"])
    if any(len(state[key]) != warm_count for key in warm_keys):
        raise ValueError("checkpoint warm-query observations are misaligned")
    if warm_count > completed:
        raise ValueError("checkpoint has more warm queries than completed queries")


@dataclass
class BenchmarkCheckpoint:
    path: Path
    signature: str
    every_queries: int
    modes: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path, signature: str, every_queries: int) -> BenchmarkCheckpoint:
        if every_queries < 1:
            raise ValueError("checkpoint interval must be positive")
        checkpoint = cls(path=path, signature=signature, every_queries=every_queries)
        if not path.exists():
            return checkpoint
        with path.open("r", encoding="utf-8") as stream:
            raw = json.load(stream)
        if raw.get("version") != 1:
            raise ValueError(f"unsupported checkpoint version in {path}")
        if raw.get("signature") != signature:
            raise ValueError(
                f"checkpoint signature mismatch: {path}; remove it or choose another path"
            )
        raw_modes = raw.get("modes")
        if not isinstance(raw_modes, dict):
            raise TypeError("checkpoint modes must be an object")
        for mode, state in raw_modes.items():
            if not isinstance(mode, str) or not isinstance(state, dict):
                raise TypeError("checkpoint contains an invalid mode state")
            validate_mode_state(state)
        checkpoint.modes = raw_modes
        return checkpoint

    def state_for(self, mode: str) -> dict[str, Any]:
        state = self.modes.setdefault(mode, empty_mode_state())
        validate_mode_state(state)
        return state

    def save(self) -> None:
        for state in self.modes.values():
            validate_mode_state(state)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "signature": self.signature,
            "modes": self.modes,
        }
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=self.path.parent,
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            encoding="utf-8",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            try:
                json.dump(payload, temporary, separators=(",", ":"), sort_keys=True)
                temporary.write("\n")
                temporary.flush()
                temporary_path.replace(self.path)
            except BaseException:
                temporary_path.unlink(missing_ok=True)
                raise

    def save_progress(self, mode: str, total_queries: int, *, force: bool = False) -> None:
        completed = int(self.state_for(mode)["queries_completed"])
        if force or completed % self.every_queries == 0:
            self.save()
            print(f"checkpointed {mode}: {completed}/{total_queries}", flush=True)

    def clear(self) -> None:
        self.path.unlink(missing_ok=True)
