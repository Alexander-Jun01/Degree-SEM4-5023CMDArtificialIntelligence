"""Make the report's learning curve, policy map, state-visit and robustness charts, and statistics.

Reads the saved results only; it does not retrain or rerun anything.
Needs matplotlib and scipy as well as numpy:
    python -m pip install matplotlib scipy
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from src.traffic_agent import CONFIG


RESULTS = Path(__file__).resolve().parent / "results"
SCENARIOS = ["balanced", "directional", "rush_change", "sensor_fault", "capacity_loss"]
LABELS = {"balanced": "Balanced", "directional": "Directional", "rush_change": "Rush change",
          "sensor_fault": "Sensor fault", "capacity_loss": "Capacity loss"}

# Colours: validated categorical slots 1-3, plus text and grid inks.
ALPHA_COLOURS = {0.05: "#2a78d6", 0.15: "#eb6834", 0.35: "#1baf7a"}
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
ROLLING_WINDOW = 5  # snapshots, i.e. 250 training episodes


def read_csv(name):
    with (RESULTS / name).open(newline="") as file:
        return list(csv.DictReader(file))


def style_axis(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def learning_curve():
    rows = read_csv("training_log.csv")
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(8, 5.6), sharex=True,
                                      gridspec_kw={"height_ratios": [3, 1], "hspace": 0.12})
    for alpha, colour in ALPHA_COLOURS.items():
        selected = [r for r in rows if float(r["alpha"]) == alpha]
        episodes = np.array([int(r["episode"]) for r in selected])
        queues = np.array([float(r["mean_queue_vehicles"]) for r in selected])
        # Each snapshot is one episode from a rotating scenario, so smooth over five.
        smooth = np.convolve(queues, np.ones(ROLLING_WINDOW) / ROLLING_WINDOW, mode="valid")
        smooth_x = episodes[ROLLING_WINDOW - 1:]
        top.scatter(episodes, queues, s=14, color=colour, alpha=0.30, linewidths=0)
        width = 2.6 if alpha == 0.05 else 1.6
        name = f"learning rate α = {alpha}" + (" (selected)" if alpha == 0.05 else "")
        top.plot(smooth_x, smooth, color=colour, linewidth=width, solid_capstyle="round", label=name)
    legend = top.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=INK)
    legend.get_texts()[0].set_fontweight("bold")

    decay_end = CONFIG["epsilon_decay_episodes"]
    for ax in (top, bottom):
        ax.axvline(decay_end, color=INK_2, linewidth=0.8, linestyle=(0, (3, 3)))
    top.text(decay_end + 20, 45, "exploration reaches\nits floor (0.05)",
             fontsize=8.5, color=INK_2, va="bottom")
    top.set_ylabel("Mean queued vehicles\n(one training episode)", fontsize=9.5, color=INK_2)
    top.set_title("Training queue length for three learning rates", loc="left",
                  fontsize=12, color=INK, fontweight="bold", pad=10)

    episodes = np.arange(CONFIG["training_episodes"])
    eps = np.maximum(CONFIG["epsilon_end"], CONFIG["epsilon_start"] -
                     (CONFIG["epsilon_start"] - CONFIG["epsilon_end"]) * episodes / decay_end)
    bottom.plot(episodes + 1, eps, color=INK_2, linewidth=1.6)
    bottom.set_ylabel("Exploration ε", fontsize=9.5, color=INK_2)
    bottom.set_ylim(0, 1.05)
    bottom.set_xlabel("Training episode", fontsize=9.5, color=INK_2)
    for ax in (top, bottom):
        style_axis(ax)
    top.set_xlim(0, CONFIG["training_episodes"] + 50)

    fig.text(0.01, 0.005, "Dots: snapshot every 50 episodes. Lines: rolling mean of 5 snapshots (250 episodes). "
             "Training queues include exploratory actions.", fontsize=8, color=INK_2)
    out = RESULTS / "learning_curve.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def comparison():
    rows = read_csv("evaluation.csv")
    table = []
    for scenario in SCENARIOS:
        def queues(controller):
            selected = sorted((r for r in rows if r["scenario"] == scenario and r["controller"] == controller),
                              key=lambda r: int(r["seed"]))
            return np.array([float(r["mean_queue_vehicles"]) for r in selected])
        q, rule = queues("q_learning"), queues("greedy")
        diff = q - rule  # negative means Q-learning had shorter queues
        n = len(diff)
        se = diff.std(ddof=1) / np.sqrt(n)
        t_crit = stats.t.ppf(0.975, n - 1)
        table.append({
            "scenario": scenario,
            "q_learning_mean": round(q.mean(), 2),
            "queue_rule_mean": round(rule.mean(), 2),
            "mean_difference_q_minus_rule": round(diff.mean(), 2),
            "ci95_low": round(diff.mean() - t_crit * se, 2),
            "ci95_high": round(diff.mean() + t_crit * se, 2),
            "paired_t_p": stats.ttest_rel(q, rule).pvalue,
            "wilcoxon_p": stats.wilcoxon(q, rule).pvalue,
            "q_learning_better_seeds": int((diff < 0).sum()),
            "seeds": n,
        })

    # Holm correction because five scenarios are tested.
    order = np.argsort([r["paired_t_p"] for r in table])
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(table) - rank) * table[index]["paired_t_p"]))
        table[index]["holm_adjusted_p"] = running
    for r in table:
        for key in ("paired_t_p", "wilcoxon_p", "holm_adjusted_p"):
            r[key] = float(f"{r[key]:.3g}")

    out_csv = RESULTS / "qlearning_vs_queue_rule.csv"
    with out_csv.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)

    # Dot-and-interval chart of the paired differences.
    fig, ax = plt.subplots(figsize=(8, 3.6))
    y = np.arange(len(table))[::-1]
    for yi, r in zip(y, table):
        significant = r["holm_adjusted_p"] < 0.05
        # Blue: Q-learning significantly better. Orange: queue rule significantly better.
        if not significant:
            colour = INK_2
        else:
            colour = "#2a78d6" if r["mean_difference_q_minus_rule"] < 0 else "#eb6834"
        ax.plot([r["ci95_low"], r["ci95_high"]], [yi, yi], color=colour, linewidth=2, solid_capstyle="round")
        ax.scatter(r["mean_difference_q_minus_rule"], yi, s=60, color=colour, zorder=3,
                   edgecolors="white", linewidths=2)
        verdict = "significant" if significant else "not significant"
        ax.annotate(f"{r['mean_difference_q_minus_rule']:+.2f}  ({verdict})", (r["ci95_high"], yi),
                    xytext=(8, 0), textcoords="offset points", va="center", fontsize=8.5, color=INK)
    ax.axvline(0, color=INK, linewidth=1)
    ax.set_yticks(y, [LABELS[r["scenario"]] for r in table], fontsize=9.5, color=INK)
    lo = min(r["ci95_low"] for r in table)
    hi = max(r["ci95_high"] for r in table)
    ax.set_xlim(lo - 1, hi + 4.5)
    ax.text(lo - 0.9, len(table) - 0.35, "← Q-learning shorter queues", fontsize=8.5, color=INK_2)
    ax.text(0.15, len(table) - 0.35, "queue rule shorter queues →", fontsize=8.5, color=INK_2)
    ax.set_xlabel("Paired difference in mean queued vehicles (Q-learning − queue rule), 95% CI",
                  fontsize=9.5, color=INK_2)
    ax.set_title("Q-learning vs the queue rule on held-out test seeds", loc="left",
                 fontsize=12, color=INK, fontweight="bold", pad=18)
    style_axis(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_ylim(-0.6, len(table) - 0.1)
    fig.text(0.01, -0.04, "30 matched seeds per scenario. Significant = Holm-adjusted paired t-test p < 0.05 "
             "across the five scenarios.",
             fontsize=8, color=INK_2)
    out_png = RESULTS / "qlearning_vs_queue_rule.png"
    fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return table, out_csv, out_png



BIN_LABELS = ["0-2", "3-7", "8-14", "15-29", "30+"]   # np.digitize with BINS [3, 8, 15, 30]
NS_COLOUR, EW_COLOUR, UNSEEN = "#2a78d6", "#eb6834", "#f1f0ec"


def policy_map(age=3):
    """Greedy action of the saved Q-table for each pair of queue bins, at one green age."""
    from matplotlib.colors import ListedColormap
    q = np.load(RESULTS / "q_table.npy")
    visits = np.load(RESULTS / "state_visits.npy")
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.9))
    for phase, ax in enumerate(axes):
        values = q[:, :, phase, age, :]
        greedy = np.where(values[..., 0] == values[..., 1], phase, np.argmax(values, axis=-1))
        seen = visits[:, :, phase, age] > 0
        grid = np.where(seen, greedy, 2)
        ax.imshow(grid, cmap=ListedColormap([NS_COLOUR, EW_COLOUR, UNSEEN]), vmin=0, vmax=2, origin="lower")
        for ns in range(5):
            for ew in range(5):
                text = ("N-S" if grid[ns, ew] == 0 else "E-W") if seen[ns, ew] else "unseen"
                colour = "white" if seen[ns, ew] else INK_2
                ax.text(ew, ns, text, ha="center", va="center", fontsize=8, color=colour)
        ax.set_xticks(range(5), BIN_LABELS)
        ax.set_yticks(range(5), BIN_LABELS)
        ax.set_xlabel("observed E-W queue (cars)", fontsize=9, color=INK_2)
        ax.set_ylabel("observed N-S queue (cars)", fontsize=9, color=INK_2)
        ax.set_title(f"{'N-S' if phase == 0 else 'E-W'} currently green", fontsize=10, color=INK)
        ax.tick_params(colors=INK_2, labelsize=8, length=0)
        for side in ax.spines.values():
            side.set_visible(False)
        ax.set_xticks(np.arange(-0.5, 5), minor=True)
        ax.set_yticks(np.arange(-0.5, 5), minor=True)
        ax.grid(which="minor", color="white", linewidth=2)
        ax.tick_params(which="minor", length=0)
    note = " Unseen = state never visited in training." if (visits[:, :, :, age] == 0).any() else ""
    fig.text(0.01, -0.02, f"Cell = phase the saved Q-table requests at green age {age} (minimum green met)." + note,
             fontsize=8, color=INK_2)
    fig.tight_layout()
    out = RESULTS / "policy_map.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def state_visits():
    """How often training visited each pair of queue bins (all phases and green ages)."""
    visits = np.load(RESULTS / "state_visits.npy")
    counts = visits.sum(axis=(2, 3))
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    shown = np.log10(counts + 1)
    image = ax.imshow(shown, cmap="Blues", origin="lower", vmin=0, vmax=shown.max())
    for ns in range(5):
        for ew in range(5):
            light = shown[ns, ew] > 0.6 * shown.max()
            ax.text(ew, ns, f"{int(counts[ns, ew]):,}", ha="center", va="center", fontsize=8,
                    color="white" if light else INK)
    ax.set_xticks(range(5), BIN_LABELS)
    ax.set_yticks(range(5), BIN_LABELS)
    ax.set_xlabel("observed E-W queue (cars)", fontsize=9, color=INK_2)
    ax.set_ylabel("observed N-S queue (cars)", fontsize=9, color=INK_2)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    bar.set_label("visits (log10 scale)", fontsize=8, color=INK_2)
    bar.ax.tick_params(labelsize=7, colors=INK_2)
    visited = int((visits > 0).sum())
    fig.text(0.01, -0.02, f"{int(visits.sum()):,} training visits; {visited} of {visits.size} states visited. "
             f"Per full state: fewest {int(visits.min())}, median {int(np.median(visits))}, "
             f"{int((visits < 100).sum())} states under 100 visits.", fontsize=8, color=INK_2)
    out = RESULTS / "state_visits.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out, visited


def seed_robustness():
    """Q-learning test mean queue for five training seeds, against the queue rule."""
    seeds = [r for r in read_csv("training_seeds.csv") if r["controller"] == "q_learning"]
    evaluation = read_csv("evaluation.csv")
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for i, scenario in enumerate(SCENARIOS):
        values = [float(r["mean_queue_vehicles"]) for r in seeds if r["scenario"] == scenario]
        rule = np.mean([float(r["mean_queue_vehicles"]) for r in evaluation
                        if r["scenario"] == scenario and r["controller"] == "greedy"])
        # Relative to the queue rule so all scenarios share one axis.
        diffs = np.array(values) - rule
        ax.scatter(np.full(len(diffs), i) + np.linspace(-0.12, 0.12, len(diffs)), diffs, s=36,
                   color=ALPHA_COLOURS[0.05], zorder=3, edgecolors="white", linewidths=1)
        ax.text(i + 0.2, diffs.mean(), f"SD {np.std(values, ddof=1):.2f}", fontsize=8, color=INK_2, va="center")
    ax.axhline(0, color=INK, linewidth=1)
    ax.text(-0.45, 0.25, "queue rule (0)", fontsize=8, color=INK, ha="left")
    ax.set_xticks(range(len(SCENARIOS)), [LABELS[s] for s in SCENARIOS])
    ax.set_xlim(-0.5, len(SCENARIOS) - 0.3)
    ax.set_ylabel("Q-learning minus queue rule\n(mean queued vehicles)", fontsize=9, color=INK_2)
    style_axis(ax)
    fig.text(0.01, -0.03, "Each dot = one training seed (5023, 8023, 11023, 14023, 17023), tested on the same "
             "30 seeds per scenario. Below 0 = Q-learning better.", fontsize=8, color=INK_2)
    out = RESULTS / "training_seed_robustness.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def sensor_fault_safe_mode():
    """Sensor-fault mean queue with 95% t intervals for the queue rule, Q-learning and safe mode."""
    rows = [r for r in read_csv("evaluation.csv") if r["scenario"] == "sensor_fault"]
    names = [("greedy", "Queue rule", "#9b9a95"), ("q_learning", "Q-learning", ALPHA_COLOURS[0.05]),
             ("q_learning_safe", "Q-learning +\nmemory safe mode", ALPHA_COLOURS[0.35])]
    fixed = np.mean([float(r["mean_queue_vehicles"]) for r in rows if r["controller"] == "fixed"])
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for i, (key, label, colour) in enumerate(names):
        values = np.array([float(r["mean_queue_vehicles"]) for r in rows if r["controller"] == key])
        half = stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
        ax.bar(i, values.mean(), width=0.55, color=colour)
        ax.errorbar(i, values.mean(), yerr=half, color=INK, capsize=5, linewidth=1.2)
        ax.text(i, values.mean() + half + 0.25, f"{values.mean():.2f}", ha="center", fontsize=9, color=INK)
    ax.set_xticks(range(len(names)), [n[1] for n in names])
    ax.set_ylabel("mean queued vehicles", fontsize=9, color=INK_2)
    ax.set_ylim(0, 17)
    style_axis(ax)
    fig.text(0.01, -0.04, f"Sensor-fault scenario, 30 matched test seeds; bars show 95% t intervals. "
             f"Fixed timing (not shown) averages {fixed:.2f}.", fontsize=8, color=INK_2)
    out = RESULTS / "sensor_fault_safe_mode.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out



def _demo_rows():
    import json
    with (RESULTS / "demo_trace_unapproved.jsonl").open() as file:
        return [json.loads(line) for line in file]


def belief_timeline():
    """P(congestion) over the 36 demo steps, with the observed total and review flags."""
    rows = _demo_rows()
    steps = np.array([r["step"] for r in rows])
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(8, 4.6), sharex=True,
                                      gridspec_kw={"height_ratios": [1, 1], "hspace": 0.12})
    seen = [r for r in rows if r["observation"] is not None]
    top.plot([r["step"] for r in seen], [sum(r["observation"]) for r in seen], "o-", color=ALPHA_COLOURS[0.05],
             markersize=4, linewidth=1.2, label="observed total queue")
    missing = [r["step"] for r in rows if r["observation"] is None]
    top.scatter(missing, [2] * len(missing), marker="x", color=INK, s=22, label="missing reading", zorder=3)
    top.axhline(30, color=INK_2, linewidth=0.9, linestyle="--")
    top.text(35.9, 30, "detector\npositive (30)", fontsize=7.5, color=INK_2, va="center", ha="left")
    top.axhline(40, color="#eb6834", linewidth=0.9, linestyle="--")
    top.text(35.9, 40, "review (40)", fontsize=7.5, color="#eb6834", va="center", ha="left")
    top.set_ylabel("cars", fontsize=9, color=INK_2)
    top.legend(loc="upper center", frameon=False, fontsize=8, ncol=2, bbox_to_anchor=(0.5, 1.18))
    prob = np.array([r["congestion_probability_assumed_model"] for r in rows])
    bottom.plot(steps, prob, color=INK, linewidth=1.4)
    bottom.axhline(0.90, color="#eb6834", linewidth=0.9, linestyle="--")
    bottom.text(35.9, 0.9, "review (0.90)", fontsize=7.5, color="#eb6834", va="center", ha="left")
    bottom.set_xlim(-0.8, 35.5)
    flagged = [r["step"] for r in rows if r["review_required"]]
    bottom.scatter(flagged, prob[flagged], color="#eb6834", s=30, zorder=3, label="review flagged")
    bottom.legend(loc="lower left", frameon=False, fontsize=8)
    bottom.set_ylim(0, 1.05)
    bottom.set_ylabel("P(congestion)", fontsize=9, color=INK_2)
    bottom.set_xlabel("demonstration step (5 s each)", fontsize=9, color=INK_2)
    for ax in (top, bottom):
        style_axis(ax)
    fig.text(0.01, -0.03, "Seed 40001, sensor-fault scenario, approval withheld (results/demo_trace_unapproved.jsonl). "
             "Detector rates are assumed.", fontsize=8, color=INK_2)
    out = RESULTS / "belief_timeline.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def demo_timeline():
    """Who decided each of the 36 demo steps, under the true queues."""
    rows = _demo_rows()
    kinds = [("learned action", "#2a78d6"), ("review: queue rule", "#eb6834"),
             ("memory safe mode", "#1baf7a"), ("missing: fixed timing", "#9b9a95")]
    def kind(r):
        if r["observation"] is None:
            return 3
        if r["safe_mode"]:
            return 2
        return 1 if r["review_required"] else 0
    fig, (top, band) = plt.subplots(2, 1, figsize=(8, 3.9), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 0.7], "hspace": 0.08})
    steps = [r["step"] for r in rows]
    true = np.array([r["true_queues_for_simulation_audit"] for r in rows])
    top.plot(steps, true[:, 0], color=INK, linewidth=1.5, label="N-S true queue")
    top.plot(steps, true[:, 1], color=INK_2, linewidth=1.5, linestyle="--", label="E-W true queue")
    fails = [r["step"] for r in rows if r["feedback"]["api_failure"]]
    top.scatter(fails, [true[s].max() + 3 for s in fails], marker="v", color="#d84a3a", s=20, label="command failed")
    top.set_ylabel("cars", fontsize=9, color=INK_2)
    top.legend(loc="upper right", frameon=False, fontsize=8, ncol=3)
    style_axis(top)
    for r in rows:
        band.add_patch(plt.Rectangle((r["step"] - 0.5, 0), 1, 1, facecolor=kinds[kind(r)][1], edgecolor="white", linewidth=1.5))
    band.set_ylim(0, 1)
    band.set_yticks([])
    band.set_xlim(-0.6, 35.6)
    for side in band.spines.values():
        side.set_visible(False)
    band.tick_params(colors=INK_2, labelsize=9, length=0)
    band.set_xlabel("demonstration step (5 s each)", fontsize=9, color=INK_2)
    counts = [sum(1 for r in rows if kind(r) == k) for k in range(4)]
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in kinds]
    fig.legend(handles, [f"{n} ({c})" for (n, _), c in zip(kinds, counts)], loc="lower center", ncol=4, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, -0.07))
    out = RESULTS / "demo_timeline.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out

if __name__ == "__main__":
    print("Saved", learning_curve())
    print("Saved", policy_map())
    print("Saved", state_visits())
    print("Saved", seed_robustness())
    print("Saved", sensor_fault_safe_mode())
    print("Saved", belief_timeline())
    print("Saved", demo_timeline())
    table, out_csv, out_png = comparison()
    print("Saved", out_csv)
    print("Saved", out_png)
    print()
    print(f"{'Scenario':<14}{'Q':>7}{'Rule':>7}{'Diff':>7}{'95% CI':>17}{'t p':>10}{'Holm p':>10}{'Wilcoxon p':>12}{'Q better':>10}")
    for r in table:
        ci = f"[{r['ci95_low']:.2f}, {r['ci95_high']:.2f}]"
        print(f"{LABELS[r['scenario']]:<14}{r['q_learning_mean']:>7.2f}{r['queue_rule_mean']:>7.2f}"
              f"{r['mean_difference_q_minus_rule']:>+7.2f}{ci:>17}{r['paired_t_p']:>10.3g}"
              f"{r['holm_adjusted_p']:>10.3g}{r['wilcoxon_p']:>12.3g}{r['q_learning_better_seeds']:>7}/{r['seeds']}")
