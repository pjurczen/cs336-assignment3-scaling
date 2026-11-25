"""Modal GPU function behind the self-study runner.

Deploy once, and again after changing any training code:

    uv run --extra server modal deploy cs336_scaling/selfstudy/remote.py
"""

from typing import Any

from cs336_scaling.modal_utils import SUNET_ID, app
from cs336_scaling.training.run import TRAINING_FUNCTION_KWARGS, run_training
from cs336_scaling.training.training_config import TrainingConfig


@app.function(**TRAINING_FUNCTION_KWARGS)
def run_experiment(
        training_config: TrainingConfig, experiment_id: int
) -> dict[str, Any]:
    """Run one training job; always return a status dict instead of raising.

    Returning (not raising) keeps a training timeout distinguishable from the
    TimeoutError Modal raises locally while a spawned call is still running.
    """
    try:
        result = run_training.local(
            training_config, sunet_id=SUNET_ID, experiment_id=experiment_id
        )
    except TimeoutError as exc:
        return {"status": "timeout", "error": str(exc)}
    except Exception as exc:
        elapsed = getattr(exc, "elapsed_training_seconds", None)
        return {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "training_seconds": None if elapsed is None else float(elapsed),
        }
    return {
        "status": "completed",
        "val_losses": [float(loss) for loss in result.val_losses],
        "training_seconds": float(result.training_time),
        "model_params": int(result.model_params),
        "estimated_memory_bytes": float(result.estimated_memory),
    }
