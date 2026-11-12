"""Watch the Q-learning agent learn from zero, like the SmartFactory lab.

The agent starts with an empty Q-table (it knows nothing) and trains on
simulated traffic. Every few episodes its current policy is tested, with
exploration switched off, on the same held-out traffic, so the printed
queue length shows how much it has improved. Each checkpoint also records
what it has learned: which traffic situations changed their chosen action,
example Q-values and one worked Q-value update.

Run from the TrafficAgent folder:  python learning_demo.py
Add --plot to draw the learning curve and policy map live (needs matplotlib).
Output: data/learning_progress.csv and data/learning_snapshots.json
(and data/learning_curve_live.png with --plot).
The saved model and report results in results/ are not changed.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from src.traffic_agent import CONFIG, SCENARIOS, Environment, choose, run


# SETTINGS: change these before running the file, or use the options shown by
# python learning_demo.py --help
EPISODES = 2200          # Training episodes (each is 30 simulated minutes).
CHECK_EVERY = 100        # Test the current policy every this many episodes.
LEARNING_RATE = 0.05     # The rate selected in the report.
PAUSE = 0.0              # Extra real seconds to wait after each progress line.

PROJECT_FOLDER = Path(__file__).resolve().parent
DATA_FOLDER = PROJECT_FOLDER / "data"
# Held-out test traffic: three demand patterns, three validation seeds each.
TEST_SCENARIOS = SCENARIOS[:3]
TEST_SEEDS = CONFIG["validation_seeds"][:3]
PHASES = ("N-S", "E-W")
# Queue-size bins used by the state (src/traffic_agent.py BINS = 3, 8, 15, 30).
BIN_LABELS = ("0-2", "3-7", "8-14", "15-29", "30+")
MAP_AGE = 2              # Policy map shows situations after 10 s of green.
# Example situations: (N-S bin, E-W bin, current green phase, green age).
EXAMPLES = (
    ("N-S queue 30+, E-W 0-2, N-S green for 10 s", (4, 0, 0, MAP_AGE)),
    ("N-S queue 0-2, E-W 30+, N-S green for 10 s", (0, 4, 0, MAP_AGE)),
    ("Both queues 8-14, N-S green for 10 s", (2, 2, 0, MAP_AGE)),
)


def test_policy(q, controller="q_learning"):
    """Mean queued vehicles on the held-out traffic. Exploration is off."""
    scores = [run(q, scenario, seed, controller)["mean_queue_vehicles"]
              for scenario in TEST_SCENARIOS for seed in TEST_SEEDS]
    return float(np.mean(scores))


def greedy_policy(q, visits):
    """Chosen action for every state; -1 where the state has never been visited."""
    policy = np.full(visits.shape, -1, dtype=int)
    for state in zip(*np.nonzero(visits)):
        policy[state] = choose(q, tuple(int(s) for s in state))
    return policy


def bar(value, worst):
    """Text bar: shorter means fewer queued cars."""
    width = int(round(30 * min(value, worst) / worst))
    return "#" * width + "." * (30 - width)


class LivePlot:
    """Chart window: learning curve, policy map at 10 s green, and the map holding the
    checkpoint's worked update (red ring = the one Q-table cell that update changed)."""

    def __init__(self, fixed, queue_rule, episodes, seed):
        import matplotlib.pyplot as plt   # Only needed with --plot.
        from matplotlib.colors import ListedColormap
        from matplotlib.patches import Rectangle
        self.plt = plt
        plt.ion()
        self.fig, (self.ax, self.map_ax, self.upd_ax) = plt.subplots(
            1, 3, figsize=(19, 6), gridspec_kw={"width_ratios": [1.5, 1, 1]})
        self.fig.canvas.manager.set_window_title("Q-learning from zero")
        self.ax.axhline(fixed, color="#d84a3a", linestyle="--", label=f"Fixed timing ({fixed:.2f})")
        self.ax.axhline(queue_rule, color="#e0a100", linestyle="--", label=f"Queue rule ({queue_rule:.2f})")
        (self.line,) = self.ax.plot([], [], color="#2f8f62", marker="o", markersize=4,
                                    linewidth=2, label="Q-learning (test)")
        self.ax.set_xlim(0, episodes)
        self.ax.set_xlabel("Training episodes (each = 30 simulated minutes)")
        self.ax.set_ylabel("Mean queued vehicles on held-out traffic")
        self.ax.set_title(f"Learning from an empty Q-table (seed {seed}). Lower is better")
        self.ax.grid(alpha=0.3)
        self.ax.legend(loc="upper right")
        # Policy map colours: grey = never seen, green = N-S green, amber = E-W green.
        cmap = ListedColormap(["#c9cdd0", "#2f8f62", "#e0a100"])
        self.maps = {}
        for ax in (self.map_ax, self.upd_ax):
            image = ax.imshow(np.full((5, 5), -1), cmap=cmap, vmin=-1, vmax=1, origin="lower")
            ax.set_xticks(range(5), BIN_LABELS)
            ax.set_yticks(range(5), BIN_LABELS)
            ax.set_xlabel("E-W queue (cars)")
            ax.set_ylabel("N-S queue (cars)")
            texts = [[ax.text(e, n, "", ha="center", va="center", fontsize=9, color="white")
                      for e in range(5)] for n in range(5)]
            ring = Rectangle((0, 0), 1, 1, fill=False, edgecolor="#d84a3a", linewidth=4, visible=False)
            ax.add_patch(ring)
            self.maps[ax] = (image, texts, ring)
        self.upd_ax.set_title("Updated cell: none yet (empty Q-table)", fontsize=10)
        self.xs, self.ys = [], []
        self.fixed = fixed
        self.fig.tight_layout()

    def _draw_map(self, ax, grid, labels, cell=None):
        image, texts, ring = self.maps[ax]
        image.set_data(grid)
        for n in range(5):
            for e in range(5):
                texts[n][e].set_text("" if grid[n][e] < 0 else labels[grid[n][e]])
        ring.set_visible(cell is not None)
        if cell is not None:
            ring.set_xy((cell[1] - 0.5, cell[0] - 0.5))

    def add(self, episode, score, grid=None, changed=0, update=None):
        self.xs.append(episode)
        self.ys.append(score)
        self.line.set_data(self.xs, self.ys)
        self.ax.set_ylim(0, max(max(self.ys), self.fixed) * 1.1)
        cell = update["cell"] if update else None
        if grid is not None:
            # Ring here too when the update happened at N-S green for 10 s (this map's slice).
            on_map = cell if cell and cell[2] == 0 and cell[3] == MAP_AGE else None
            self._draw_map(self.map_ax, grid, ("hold N-S", "to E-W"), on_map)
            self.map_ax.set_title(f"Choice when N-S has been green 10 s\n"
                                  f"episode {episode}: {changed} situations changed choice")
        if update:
            self._draw_map(self.upd_ax, update["grid"], ("N-S", "E-W"), cell)
            self.upd_ax.set_title(f"Red ring = cell changed by one update (episode {episode})\n"
                                  f"{update['situation']}\n"
                                  f"Q({update['action']}): {update['old']} -> {update['new']}"
                                  f"  (target {update['target']})", fontsize=10)
        self.fig.canvas.draw_idle()
        self.plt.pause(0.01)

    def finish(self, path):
        self.fig.savefig(path, dpi=150, bbox_inches="tight")
        self.plt.ioff()
        self.plt.show()


def learn(episodes, check_every, alpha, seed, pause, plot=False):
    DATA_FOLDER.mkdir(exist_ok=True)
    rng = np.random.default_rng(seed)
    q = np.zeros((5, 5, 2, 7, 2), dtype=float)   # Starts from zero: no knowledge.
    visits = np.zeros(q.shape[:-1], dtype=np.int64)

    # Baselines on the same held-out traffic, for comparison.
    fixed = test_policy(q, "fixed")
    queue_rule = test_policy(q, "greedy")
    start_score = test_policy(q)

    print("=" * 80)
    print("Q-LEARNING FROM ZERO")
    print(f"Training seed {seed} | learning rate {alpha} | discount {CONFIG['discount']}")
    print(f"Comparison on held-out traffic: fixed timing {fixed:.2f}, queue rule {queue_rule:.2f}")
    print("Lower queue = better. Exploration falls from 1.00 to 0.05 over "
          f"{CONFIG['epsilon_decay_episodes']} episodes.")
    print("'Changed' = traffic situations whose chosen action changed since the last check.")
    print("=" * 80)
    print(f"{'Episode':>7}  {'Explore':>7}  {'Test queue':>10}  {'Changed':>7}  Progress (shorter is better)")

    rows = [{"episode": 0, "epsilon": 1.0, "test_mean_queue": round(start_score, 3)}]
    snapshots = [{"episode": 0, "epsilon": 1.0, "score": round(start_score, 3), "changed": 0,
                  "newly_seen": 0, "seen": 0, "grids": [[[-1] * 5 for _ in range(5)]] * 2,
                  "examples": [[0.0, 0.0, PHASES[choose(q, s)]] for _, s in EXAMPLES], "update": None}]
    worst = max(start_score, fixed) * 1.05
    print(f"{0:>7}  {1.0:>7.2f}  {start_score:>10.2f}  {'-':>7}  {bar(start_score, worst)}  (empty Q-table)")
    chart = LivePlot(fixed, queue_rule, episodes, seed) if plot else None
    if chart:
        chart.add(0, start_score, snapshots[0]["grids"][0])
    previous = greedy_policy(q, visits)
    best = (start_score, 0)
    beat_rule = False
    began = time.monotonic()

    for ep in range(episodes):
        env = Environment(seed + ep, SCENARIOS[ep % 3])
        epsilon = max(CONFIG["epsilon_end"], CONFIG["epsilon_start"] -
                      (CONFIG["epsilon_start"] - CONFIG["epsilon_end"]) * ep / CONFIG["epsilon_decay_episodes"])
        obs = env.observe()
        update = None
        for step in range(env.steps):
            state = env.state(obs)
            action = choose(q, state, rng, epsilon)
            reward, done, _ = env.step(action)
            next_obs = env.observe()
            next_state = env.state(next_obs)
            # Q-learning update (Sutton and Barto, 2018). Training traffic has no
            # missing readings, but a missing state is skipped for safety.
            if state is not None:
                best_next = 0.0 if done or next_state is None else float(np.max(q[next_state]))
                old = float(q[state + (action,)])
                target = reward + CONFIG["discount"] * best_next
                q[state + (action,)] = old + alpha * (target - old)
                visits[state] += 1
                if step == env.steps // 2:   # Keep one mid-episode update as a worked example.
                    update = {"situation": f"N-S {BIN_LABELS[state[0]]}, E-W {BIN_LABELS[state[1]]}, "
                                           f"{PHASES[state[2]]} green for {5 * state[3]} s",
                              "action": f"{PHASES[action]} green", "old": round(old, 2),
                              "reward": round(float(reward), 2), "best_next": round(best_next, 2),
                              "target": round(target, 2), "new": round(float(q[state + (action,)]), 2),
                              "cell": [int(x) for x in state] + [int(action)]}
            obs = next_obs

        if (ep + 1) % check_every == 0 or ep == episodes - 1:
            score = test_policy(q)
            policy = greedy_policy(q, visits)
            changed = int(np.sum((policy != previous) & (previous >= 0)))
            newly_seen = int(np.sum((policy >= 0) & (previous < 0)))
            if update:   # Map slice that holds the worked-example cell (its phase and green age).
                _, _, ph, age, _ = update["cell"]
                update["grid"] = policy[:, :, ph, age].tolist()
                update["prev_grid"] = previous[:, :, ph, age].tolist()
            previous = policy
            note = ""
            if score < best[0]:
                best = (score, ep + 1)
                note = "  new best"
            if score < queue_rule and not beat_rule:
                beat_rule = True
                note += "  beats queue rule"
            rows.append({"episode": ep + 1, "epsilon": round(epsilon, 3), "test_mean_queue": round(score, 3)})
            grids = [policy[:, :, phase, MAP_AGE].tolist() for phase in (0, 1)]
            snapshots.append({
                "episode": ep + 1, "epsilon": round(epsilon, 3), "score": round(score, 3),
                "changed": changed, "newly_seen": newly_seen, "seen": int(np.sum(policy >= 0)),
                "grids": grids,
                "examples": [[round(float(q[s][0]), 1), round(float(q[s][1]), 1), PHASES[choose(q, s)]]
                             for _, s in EXAMPLES],
                "update": update})
            print(f"{ep + 1:>7}  {epsilon:>7.2f}  {score:>10.2f}  {changed:>7}  {bar(score, worst)}{note}", flush=True)
            if chart:
                chart.add(ep + 1, score, grids[0], changed, update)
            time.sleep(pause)

    with (DATA_FOLDER / "learning_progress.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "epsilon", "test_mean_queue",
                                               "fixed_timing", "queue_rule", "seed"])
        writer.writeheader()
        writer.writerows({**r, "fixed_timing": round(fixed, 3), "queue_rule": round(queue_rule, 3),
                          "seed": seed} for r in rows)
    (DATA_FOLDER / "learning_snapshots.json").write_text(json.dumps({
        "seed": seed, "alpha": alpha, "discount": CONFIG["discount"], "fixed": round(fixed, 3),
        "rule": round(queue_rule, 3), "bins": BIN_LABELS, "examples": [name for name, _ in EXAMPLES],
        "snapshots": snapshots}, separators=(",", ":")))

    print("=" * 80)
    print(f"Finished {episodes} episodes in {time.monotonic() - began:.0f} s.")
    print(f"Start {start_score:.2f} -> end {rows[-1]['test_mean_queue']:.2f} queued cars; "
          f"best {best[0]:.2f} at episode {best[1]}.")
    print(f"Fixed timing {fixed:.2f}, queue rule {queue_rule:.2f}.")
    print(f"Saved {DATA_FOLDER / 'learning_progress.csv'} and learning_snapshots.json")
    if chart:
        chart.finish(DATA_FOLDER / "learning_curve_live.png")
        print(f"Saved {DATA_FOLDER / 'learning_curve_live.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--episodes", type=int, default=EPISODES)
    parser.add_argument("--check-every", type=int, default=CHECK_EVERY)
    parser.add_argument("--alpha", type=float, default=LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=None,
                        help="Training seed. Omit for a new random seed each run.")
    parser.add_argument("--pause", type=float, default=PAUSE)
    parser.add_argument("--plot", action="store_true",
                        help="Draw the learning curve and policy map live (needs matplotlib).")
    args = parser.parse_args()
    if args.episodes < 1 or args.check_every < 1 or not 0 < args.alpha <= 1:
        parser.error("episodes and check-every must be at least 1; alpha must be in (0, 1]")
    seed = args.seed if args.seed is not None else int(time.time()) % 100000
    learn(args.episodes, args.check_every, args.alpha, seed, args.pause, args.plot)
