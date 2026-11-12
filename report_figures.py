"""Make the report's learning curve and the Q-learning vs queue rule statistics.

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


if __name__ == "__main__":
    print("Saved", learning_curve())
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
