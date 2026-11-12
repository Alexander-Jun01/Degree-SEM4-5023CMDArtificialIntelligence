"""Write the saved results as TensorBoard logs in the runs folder.

Reads the saved results only; it does not retrain or rerun anything.
Needs tensorboard and tensorboardX as well as numpy:
    python -m pip install tensorboard tensorboardX
    python tensorboard_export.py
    tensorboard --logdir runs --host 127.0.0.1
"""
import csv
import json
import shutil
from pathlib import Path

from tensorboardX import SummaryWriter


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RUNS = ROOT / "runs"
# The report calls the "greedy" controller the queue rule.
CONTROLLER_NAMES = {"fixed": "fixed", "greedy": "queue_rule", "q_learning": "q_learning",
                    "q_learning_safe": "q_learning_safe_mode"}


def read_csv(name):
    with (RESULTS / name).open(newline="") as file:
        return list(csv.DictReader(file))


def training():
    """One run per learning rate: training curves and the validation score."""
    rows = read_csv("training_log.csv")
    validation = {float(r["learning_rate"]): float(r["validation_mean_queue_vehicles"])
                  for r in read_csv("validation.csv")}
    for alpha in sorted({float(r["alpha"]) for r in rows}):
        writer = SummaryWriter(RUNS / "training" / f"alpha_{alpha}")
        for r in (r for r in rows if float(r["alpha"]) == alpha):
            step = int(r["episode"])
            writer.add_scalar("train/mean_queue_vehicles", float(r["mean_queue_vehicles"]), step)
            writer.add_scalar("train/reward", float(r["reward"]), step)
            writer.add_scalar("train/epsilon", float(r["epsilon"]), step)
            writer.add_scalar("train/switches", int(r["switches"]), step)
        # Validation seeds chose the learning rate; shown in the HPARAMS tab.
        writer.add_hparams({"learning_rate": alpha},
                           {"hparam/validation_mean_queue_vehicles": validation[alpha]}, name="hparams")
        writer.close()


def evaluation():
    """One run per controller; x axis is the test seed index (0-29), matched across controllers."""
    rows = read_csv("evaluation.csv")
    first_seed = min(int(r["seed"]) for r in rows)
    for controller in sorted({r["controller"] for r in rows}):
        writer = SummaryWriter(RUNS / "test" / CONTROLLER_NAMES[controller])
        for r in (r for r in rows if r["controller"] == controller):
            writer.add_scalar(f"test/{r['scenario']}/mean_queue_vehicles", float(r["mean_queue_vehicles"]),
                              int(r["seed"]) - first_seed)
        writer.close()


def demo():
    """One run per saved demonstration trace; x axis is the 5-second decision step."""
    for name in ("unapproved", "approved"):
        writer = SummaryWriter(RUNS / "demo" / name)
        with (RESULTS / f"demo_trace_{name}.jsonl").open() as file:
            for line in file:
                r = json.loads(line)
                step = r["step"]
                # Missing sensor readings have no observation or Q-values; leave a gap, not a zero.
                if r["observation"] is not None:
                    writer.add_scalar("demo/observed_queue_north_south", r["observation"][0], step)
                    writer.add_scalar("demo/observed_queue_east_west", r["observation"][1], step)
                if r["q_values"] is not None:
                    writer.add_scalar("demo/q_value_north_south_green", r["q_values"][0], step)
                    writer.add_scalar("demo/q_value_east_west_green", r["q_values"][1], step)
                writer.add_scalar("demo/congestion_probability", r["congestion_probability_assumed_model"], step)
                writer.add_scalar("demo/review_required", int(r["review_required"]), step)
                writer.add_scalar("demo/acknowledged_phase", r["acknowledged_phase"], step)
                writer.add_scalar("demo/api_failure", int(r["feedback"]["api_failure"]), step)
                writer.add_scalar("demo/reward", r["reward"], step)
                writer.add_text("demo/reason", r["reason"], step)
        writer.close()


if __name__ == "__main__":
    # Start clean so a rerun does not duplicate points.
    shutil.rmtree(RUNS, ignore_errors=True)
    training()
    evaluation()
    demo()
    print("Saved", RUNS)
    print("View with: tensorboard --logdir runs --host 127.0.0.1")
