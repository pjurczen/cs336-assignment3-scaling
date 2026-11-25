"""Local client for self-study training runs on Modal (stands in for the course API).

Usage, e.g. from a notebook:

    from cs336_scaling.selfstudy import runner

    eid = runner.submit(config, label="smoke")  # TrainingConfig or API-style dict
    runner.wait()                               # poll until submitted runs finish
    runner.budget()                             # used / reserved / remaining
    df = runner.results_df()                    # one row per run

Runs keep going on Modal if the notebook stops; call refresh() or wait() later
to collect them. Every run is recorded in results/runs.json.

Budget accounting follows the handout (section 3.1), per budget kind:
- a running run reserves its full max_runtime_seconds;
- a completed or crashed run is charged its training-loop time (as timed by
  run.py), clipped to [1, max_runtime_seconds];
- a timed-out, cancelled or lost run is charged max_runtime_seconds.
"""

from __future__ import annotations

import json
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import modal
import modal.exception
from cs336_scaling.modal_utils import app
# Via the schemas package: importing training_config first hits a circular
# import (training_config -> schemas/__init__ -> training_config).
from cs336_scaling.schemas import TrainingConfig

Kind = Literal["scaling", "final"]

BUDGET_SECONDS: dict[str, float] = {"scaling": 3 * 3600.0, "final": 12 * 3600.0}
LEDGER_PATH = Path(__file__).resolve().parents[2] / "results" / "runs.json"
REMOTE_FUNCTION = "run_experiment"


class BudgetExceededError(RuntimeError):
    pass


class DuplicateConfigError(ValueError):
    pass


def submit(
        config: TrainingConfig | dict[str, Any],
        *,
        label: str | None = None,
        kind: Kind = "scaling",
        allow_duplicate: bool = False,
) -> int:
    """Queue one training run on Modal and return its experiment id."""
    cfg = (
        config
        if isinstance(config, TrainingConfig)
        else TrainingConfig.model_validate(config)
    )
    refresh(verbose=False)
    runs = _load()

    if not allow_duplicate:
        for run in runs:
            if run["unique_id"] == cfg.unique_id:
                raise DuplicateConfigError(
                    f"identical config already submitted as experiment "
                    f"{run['experiment_id']} ({run['status']}); "
                    "pass allow_duplicate=True to run it again"
                )

    remaining = budget(kind)["remaining_seconds"]
    if cfg.max_runtime_seconds > remaining:
        raise BudgetExceededError(
            f"max_runtime_seconds={cfg.max_runtime_seconds:.0f} exceeds the "
            f"remaining {kind} budget of {remaining:.0f}s"
        )

    experiment_id = max((run["experiment_id"] for run in runs), default=0) + 1
    try:
        call = modal.Function.from_name(app.name, REMOTE_FUNCTION).spawn(
            cfg, experiment_id
        )
    except modal.exception.NotFoundError as exc:
        raise RuntimeError(
            f"Modal function {app.name}/{REMOTE_FUNCTION} is not deployed; run "
            "`uv run --extra server modal deploy cs336_scaling/selfstudy/remote.py`"
        ) from exc

    runs.append(
        {
            "experiment_id": experiment_id,
            "label": label,
            "kind": kind,
            "status": "running",
            "unique_id": cfg.unique_id,
            "training_config": cfg.model_dump(mode="json"),
            "function_call_id": call.object_id,
            "submitted_at": _now(),
            "finished_at": None,
            "val_losses": None,
            "final_val_loss": None,
            "training_seconds": None,
            "charged_seconds": None,
            "model_params": None,
            "estimated_memory_bytes": None,
            "error": None,
        }
    )
    _save(runs)
    return experiment_id


def refresh(verbose: bool = True) -> list[dict[str, Any]]:
    """Collect results of finished runs; return the runs that just finished."""
    runs = _load()
    finished: list[dict[str, Any]] = []
    for run in runs:
        if run["status"] != "running":
            continue
        call = modal.FunctionCall.from_id(run["function_call_id"])
        try:
            outcome = call.get(timeout=0)
        except modal.exception.OutputExpiredError:
            outcome = {"status": "lost", "error": "result expired before collection"}
        except modal.exception.FunctionTimeoutError as exc:
            outcome = {"status": "timeout", "error": f"Modal function timeout: {exc}"}
        except modal.exception.InputCancellation:
            outcome = {"status": "cancelled", "error": "cancelled on Modal"}
        except TimeoutError:
            continue  # builtin TimeoutError from get(timeout=0): still running
        except Exception as exc:  # infrastructure failure outside run_experiment
            outcome = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}

        run.update(outcome)
        run["finished_at"] = _now()
        losses = run.get("val_losses")
        run["final_val_loss"] = losses[-1] if losses else None
        run["charged_seconds"] = _charged_seconds(run)
        finished.append(run)
        if verbose:
            print(_describe(run))

    if finished:
        _save(runs)
    return finished


def wait(
        experiment_ids: list[int] | None = None, poll_seconds: float = 30.0
) -> None:
    """Block until the given runs (default: all running runs) have finished."""
    while True:
        refresh(verbose=True)
        pending = [
            run["experiment_id"]
            for run in _load()
            if run["status"] == "running"
               and (experiment_ids is None or run["experiment_id"] in experiment_ids)
        ]
        if not pending:
            return
        time.sleep(poll_seconds)


def cancel(experiment_id: int) -> None:
    """Cancel a running run. It is charged its full max_runtime_seconds."""
    runs = _load()
    run = _find(runs, experiment_id)
    if run["status"] != "running":
        raise ValueError(f"experiment {experiment_id} is already {run['status']}")
    modal.FunctionCall.from_id(run["function_call_id"]).cancel(
        terminate_containers=True
    )
    run.update(status="cancelled", error="cancelled by user", finished_at=_now())
    run["charged_seconds"] = _charged_seconds(run)
    _save(runs)


def budget(kind: Kind = "scaling") -> dict[str, float]:
    """Seconds used by finished runs, reserved by running runs, and remaining."""
    runs = [run for run in _load() if run["kind"] == kind]
    used = sum(run["charged_seconds"] or 0.0 for run in runs if run["status"] != "running")
    reserved = sum(
        run["training_config"]["max_runtime_seconds"]
        for run in runs
        if run["status"] == "running"
    )
    total = BUDGET_SECONDS[kind]
    return {
        "total_seconds": total,
        "used_seconds": used,
        "reserved_seconds": reserved,
        "remaining_seconds": total - used - reserved,
    }


def runs() -> list[dict[str, Any]]:
    """All recorded runs, as stored in results/runs.json."""
    return _load()


def results_df():
    """One row per run with the main config fields flattened (needs pandas)."""
    import pandas as pd

    rows = []
    for run in _load():
        cfg = run["training_config"]
        arch = cfg["architecture_config"]
        rows.append(
            {
                "experiment_id": run["experiment_id"],
                "label": run["label"],
                "kind": run["kind"],
                "status": run["status"],
                "hidden_size": arch["hidden_size"],
                "num_hidden_layers": arch["num_hidden_layers"],
                "num_attention_heads": arch["num_attention_heads"],
                "intermediate_size": arch["intermediate_size"],
                "train_batch_size": cfg["train_batch_size"],
                "peak_lr": cfg["optimizer_config"]["lr_scheduler"]["peak_value"],
                "total_train_tokens": cfg["total_train_tokens"],
                "max_runtime_seconds": cfg["max_runtime_seconds"],
                "model_seed": cfg["model_seed"],
                "model_params": run["model_params"],
                "training_seconds": run["training_seconds"],
                "charged_seconds": run["charged_seconds"],
                "final_val_loss": run["final_val_loss"],
                "val_losses": run["val_losses"],
                "submitted_at": run["submitted_at"],
                "finished_at": run["finished_at"],
                "error": run["error"],
            }
        )
    return pd.DataFrame(rows)


def _charged_seconds(run: dict[str, Any]) -> float:
    max_runtime = float(run["training_config"]["max_runtime_seconds"])
    seconds = run.get("training_seconds")
    if run["status"] in ("completed", "failed") and seconds is not None:
        return min(max(float(seconds), 1.0), max_runtime)
    return max_runtime


def _describe(run: dict[str, Any]) -> str:
    text = f"experiment {run['experiment_id']} {run['status']}"
    if run["final_val_loss"] is not None:
        text += f": final val loss {run['final_val_loss']:.4f}"
    if run["training_seconds"] is not None:
        text += f", trained {run['training_seconds']:.0f}s"
    if run["error"]:
        text += f" ({run['error']})"
    return text + f", charged {run['charged_seconds']:.0f}s"


def _find(runs: list[dict[str, Any]], experiment_id: int) -> dict[str, Any]:
    for run in runs:
        if run["experiment_id"] == experiment_id:
            return run
    raise KeyError(f"no experiment {experiment_id}")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load() -> list[dict[str, Any]]:
    if not LEDGER_PATH.exists():
        return []
    return json.loads(LEDGER_PATH.read_text())


def _save(runs: list[dict[str, Any]]) -> None:
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
            "w", dir=LEDGER_PATH.parent, suffix=".tmp", delete=False
    ) as handle:
        json.dump(runs, handle, indent=2)
    Path(handle.name).replace(LEDGER_PATH)
