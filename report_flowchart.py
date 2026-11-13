"""Draw the report's memory-update flowchart (one demo step, as in demo()).

Needs matplotlib:
    python -m pip install matplotlib
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

OUT = Path(__file__).resolve().parent / "results" / "memory_flowchart.png"
INK, INK_2 = "#0b0b0b", "#52514e"
STEP, STEP_EDGE = "#eaf2fc", "#2a78d6"
HUMAN, HUMAN_EDGE = "#fdeee7", "#eb6834"
MEMORY, MEMORY_EDGE = "#e6f6f0", "#138a60"
plt.rcParams["font.family"] = "Liberation Sans"

ROW = 0.62           # vertical distance between rows, inches
W, H = 2.15, 0.40    # process box size
DW, DH = 1.75, 0.50  # decision diamond size
XL, XC, XR = 0.95, 3.2, 5.45  # left, centre and right columns


def y(row):
    return 8.55 - row * ROW


def label(x, yy, text, size=6.9, bold=False, colour=INK):
    plt.gca().text(x, yy, text, ha="center", va="center", fontsize=size, color=colour,
                   fontweight="bold" if bold else "normal", linespacing=1.15)


def process(x, row, text, fill=STEP, edge=STEP_EDGE, w=W):
    plt.gca().add_patch(Rectangle((x - w / 2, y(row) - H / 2), w, H, facecolor=fill,
                                  edgecolor=edge, linewidth=1))
    label(x, y(row), text)


def terminator(x, row, text, w=W):
    plt.gca().add_patch(FancyBboxPatch((x - w / 2, y(row) - H / 2), w, H,
                                       boxstyle=f"round,pad=0,rounding_size={H / 2}",
                                       facecolor="white", edgecolor=INK, linewidth=1))
    label(x, y(row), text, bold=True)


def decision(x, row, text, fill=STEP, edge=STEP_EDGE):
    yy = y(row)
    plt.gca().add_patch(Polygon([(x, yy + DH / 2), (x + DW / 2, yy), (x, yy - DH / 2), (x - DW / 2, yy)],
                                closed=True, facecolor=fill, edgecolor=edge, linewidth=1))
    label(x, yy, text, size=6.6)


def database(x, row, text, w=W):
    yy, cap = y(row), 0.09
    ax = plt.gca()
    ax.add_patch(Rectangle((x - w / 2, yy - H / 2), w, H, facecolor=MEMORY, edgecolor="none"))
    ax.plot([x - w / 2, x - w / 2], [yy - H / 2, yy + H / 2], color=MEMORY_EDGE, linewidth=1)
    ax.plot([x + w / 2, x + w / 2], [yy - H / 2, yy + H / 2], color=MEMORY_EDGE, linewidth=1)
    ax.add_patch(Ellipse((x, yy - H / 2), w, 2 * cap, facecolor=MEMORY, edgecolor=MEMORY_EDGE, linewidth=1))
    ax.add_patch(Rectangle((x - w / 2 + 0.01, yy - H / 2), w - 0.02, H, facecolor=MEMORY, edgecolor="none"))
    ax.add_patch(Ellipse((x, yy + H / 2), w, 2 * cap, facecolor=MEMORY, edgecolor=MEMORY_EDGE, linewidth=1))
    label(x, yy - 0.03, text)


def path(points, text=None, at=0, offset=(0.07, 0.07), ha="left"):
    """Orthogonal connector through the points; arrowhead on the last segment."""
    ax = plt.gca()
    xs, ys = zip(*points)
    ax.plot(xs[:-1], ys[:-1], color=INK, linewidth=0.9, solid_capstyle="butt")
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>", mutation_scale=9,
                                 color=INK, linewidth=0.9, shrinkA=0, shrinkB=0))
    if text:
        ax.text(points[at][0] + offset[0], points[at][1] + offset[1], text, ha=ha, va="center",
                fontsize=6.6, color=INK_2, fontweight="bold")


def main():
    fig = plt.figure(figsize=(6.5, 10.75))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-0.05, 6.6)
    ax.set_ylim(-1.67, 8.9)
    ax.axis("off")

    terminator(XC, 0, "Start: load learned Q-table (long-term memory)", w=3.0)
    process(XC, 1, "Read sensor queues (N-S, E-W)")
    decision(XC, 2, "Sensor reading\nmissing?")
    process(XL, 3, "Bayesian model:\npredict only", w=1.55)
    process(XL, 4, "Propose fixed timing", w=1.55)
    process(XC, 3, "Bayesian update of\nP(congestion)")
    process(XC, 4, "Q-table proposes a phase\n(two Q-values)")
    process(XC, 5, "Retrieve precedent: queue change\nafter each action in similar cases", MEMORY, MEMORY_EDGE, w=2.3)
    database(XR + 0.15, 5, "Experience base\n(indexed SQLite)", w=1.55)
    decision(XC, 6, "≥ 3 faults in last 12\nmemory events?", MEMORY, MEMORY_EDGE)
    process(XR, 6, "Safe mode: queue rule\n(learned action not used)", MEMORY, MEMORY_EDGE, w=1.85)
    decision(XC, 7, "Queue ≥ 40, P > 0.90 or\nprecedent disagrees?")
    process(XR, 7, "Request supervisor\napproval (simulated)", HUMAN, HUMAN_EDGE, w=1.85)
    decision(XR, 8, "Approved?", HUMAN, HUMAN_EDGE)
    process(XR, 9, "Request queue-rule\nfallback", HUMAN, HUMAN_EDGE, w=1.85)
    process(XC, 10, "Send request to signal plant")
    process(XC, 11, "Plant applies interlocks: API failure,\nmin/max green, clearance", w=2.6)
    process(XC, 12, "Read acknowledged phase and\nqueue outcome (feedback)", w=2.6)
    process(XC, 13, "Append event to short-term memory\n(deque, last 12 events)", MEMORY, MEMORY_EDGE, w=2.6)
    database(XC, 14, "Commit event to audit\nknowledge base (SQLite)")
    decision(XC, 15, "More steps?")
    terminator(XC, 16, "End of run")

    half = H / 2
    path([(XC, y(0) - half), (XC, y(1) + half)])
    path([(XC, y(1) - half), (XC, y(2) + DH / 2)])
    path([(XC - DW / 2, y(2)), (XL, y(2)), (XL, y(3) + half)], "Yes", at=0, offset=(-0.32, 0.08))
    path([(XC, y(2) - DH / 2), (XC, y(3) + half)], "No", offset=(0.07, -0.1))
    path([(XL, y(3) - half), (XL, y(4) + half)])
    path([(XC, y(3) - half), (XC, y(4) + half)])
    path([(XC, y(4) - half), (XC, y(5) + half)])
    path([(XR + 0.15 - 1.55 / 2, y(5)), (XC + 2.3 / 2, y(5))])
    path([(XC, y(5) - half), (XC, y(6) + DH / 2)])
    # Memory check: repeated faults route to the queue-rule safe mode.
    path([(XC + DW / 2, y(6)), (XR - 1.85 / 2, y(6))], "Yes", offset=(0.06, 0.08))
    path([(XC, y(6) - DH / 2), (XC, y(7) + DH / 2)], "No", offset=(0.07, -0.08))
    # Review gate.
    path([(XC + DW / 2, y(7)), (XR - 1.85 / 2, y(7))], "Yes", offset=(0.06, 0.08))
    path([(XR, y(7) - half), (XR, y(8) + DH / 2)])
    path([(XC, y(7) - DH / 2), (XC, y(10) + half)], "No", offset=(0.07, -0.1))
    path([(XR - DW / 2, y(8)), (XC + 0.55, y(8)), (XC + 0.55, y(10) + half)], "Yes", offset=(-0.32, 0.08))
    path([(XR, y(8) - DH / 2), (XR, y(9) + half)], "No", offset=(0.07, -0.1))
    right = XR + 1.85 / 2 + 0.08
    ax.plot([XR + 1.85 / 2, right, right, XR], [y(6), y(6), y(10), y(10)], color=INK, linewidth=0.9)
    path([(XR, y(9) - half), (XR, y(10)), (XC + W / 2, y(10))])
    path([(XL, y(4) - half), (XL, y(10)), (XC - W / 2, y(10))])
    for r in range(10, 13):
        path([(XC, y(r) - half), (XC, y(r + 1) + half)])
    path([(XC, y(13) - half), (XC, y(14) + half + 0.09)])
    path([(XC, y(14) - half - 0.09), (XC, y(15) + DH / 2)])
    path([(XC, y(15) - DH / 2), (XC, y(16) + half)], "No", offset=(0.07, -0.1))
    loop_x = 0.02
    path([(XC - DW / 2, y(15)), (loop_x, y(15)), (loop_x, y(1)), (XC - W / 2, y(1))],
         "Yes: next 5-second step", at=1, offset=(0.08, 0.1))

    note_x = XC + 2.6 / 2 + 0.12
    ax.plot([note_x, note_x], [y(13) + H / 2, y(14) - H / 2 - 0.09], color=MEMORY_EDGE, linewidth=1)
    ax.text(note_x + 0.1, (y(13) + y(14)) / 2, "Both stores are written after\nthe outcome is known. The deque\nis read at the next step to count\nfaults; this SQLite log is for audit.\nThe experience base is built\noffline from validation runs",
            ha="left", va="center", fontsize=6.3, color=INK_2, style="italic", linespacing=1.2)
    fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved", OUT)


def architecture():
    """Figure 1.1: agent architecture (boxes and data flow)."""
    teal, fill, ink = "#1f8a8f", "#eef7f8", "#173042"
    fig = plt.figure(figsize=(10.5, 4.3))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 10.5)
    ax.set_ylim(-0.75, 3.75)
    ax.axis("off")
    bw, bh = 2.0, 0.9
    cx = (1.2, 3.7, 6.2, 8.7)
    top, mid, low = 3.0, 1.6, 0.25

    def box(x, yy, text, w=bw, colour=teal, face=fill):
        ax.add_patch(Rectangle((x - w / 2, yy - bh / 2), w, bh, facecolor=face, edgecolor=colour, linewidth=1.2))
        ax.text(x - w / 2 + 0.1, yy, text, ha="left", va="center", fontsize=9.6, color=ink, linespacing=1.25)

    def arrow(points):
        xs, ys = zip(*points)
        ax.plot(xs[:-1], ys[:-1], color=teal, linewidth=1.2)
        ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>", mutation_scale=11,
                                     color=teal, linewidth=1.2, shrinkA=0, shrinkB=0))

    box(cx[0], top, "Observed queues\nand missing-data flag")
    box(cx[1], top, "Discretised state\nand Q-learning policy")
    box(cx[2], top, "Supervisor layer:\nsafe mode, human review\nor queue-rule fallback")
    box(cx[3], top, "Local interlocks\nand simulated plant")
    box(cx[1], mid, "Bayesian congestion\nmodel (assumed rates)")
    box(cx[2], mid, "Short-term memory\n(last 12 events):\nrecent fault count", colour=MEMORY_EDGE, face=MEMORY)
    box(cx[2], low, "SQLite audit\nknowledge base", colour=MEMORY_EDGE, face=MEMORY)
    box(cx[3], low, "Acknowledged phase\nand observed outcome")
    box(cx[3], mid, "Experience base\n(indexed SQLite):\nprecedent", colour=MEMORY_EDGE, face=MEMORY)
    gap = 0.25
    for i in range(3):
        arrow([(cx[i] + bw / 2, top), (cx[i + 1] - bw / 2, top)])
    # Observations feed the Bayesian model.
    arrow([(cx[0], top - bh / 2), (cx[0], mid), (cx[1] - bw / 2, mid)])
    # Bayesian probability and memory fault count feed the supervisor layer.
    arrow([(cx[1] + bw / 2, mid + 0.2), (cx[1] + bw / 2 + gap, mid + 0.2), (cx[1] + bw / 2 + gap, top - 0.25),
           (cx[2] - bw / 2, top - 0.25)])
    arrow([(cx[2], mid + bh / 2), (cx[2], top - bh / 2)])
    # Plant outcome is fed back and stored.
    edge = cx[3] + bw / 2
    arrow([(edge, top - 0.2), (edge + 0.25, top - 0.2), (edge + 0.25, low), (edge, low)])
    # Precedent retrieved from the experience base feeds the supervisor layer.
    arrow([(cx[3] - 0.5, mid + bh / 2), (cx[3] - 0.5, mid + 0.7), (cx[2] + bw / 2 + gap, mid + 0.7),
           (cx[2] + bw / 2 + gap, top - 0.25), (cx[2] + bw / 2, top - 0.25)])
    arrow([(cx[3] - bw / 2, low), (cx[2] + bw / 2, low)])
    arrow([(cx[3] - bw / 2, low + 0.3), (cx[2] + bw / 2 + gap, low + 0.3), (cx[2] + bw / 2 + gap, mid),
           (cx[2] + bw / 2, mid)])
    ax.text(0.2, -0.45, "Short-term memory and the experience base are read every step. The experience base is built offline "
            "from validation runs; the audit log is not used for retraining.", fontsize=9, color="#52514e")
    out = OUT.parent / "architecture.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved", out)


def sequence():
    """Report figure: UML-style sequence for one flagged step (step 5 of the seed-40001 demo)."""
    names = ["Sensor", "Supervisor\nlayer", "Q-policy", "Experience\nbase", "Operator", "Signal\ncontroller", "Audit log\n(SQLite)"]
    xs = [0.7 + i * 1.55 for i in range(len(names))]
    msgs = [  # (from, to, text, reply)
        (0, 1, "1  queues [28, 11]", False),
        (1, 1, "2  Bayes update P = 0.981; recent faults 2 (< 3)", False),
        (1, 2, "3  state (queue bins, phase E-W, green age)", False),
        (2, 1, "4  propose E-W (Q = -264.3 vs -341.8)", True),
        (1, 3, "5  retrieve precedent for this situation", False),
        (3, 1, "6  N-S -0.15, E-W +1.96 cars per 20 s (26 cases each)", True),
        (1, 4, "7  review: P > 0.90, precedent disagrees + evidence", False),
        (4, 1, "8  approval withheld (simulated)", True),
        (1, 6, "9  log decision and reason", False),
        (1, 5, "10  request N-S (queue-rule fallback)", False),
        (5, 1, "11  command rejected; E-W still displayed", True),
        (1, 6, "12  commit event and outcome", False),
        (1, 1, "13  add event to short-term memory: faults now 3 (safe mode next step)", False),
    ]
    top, gap = 7.0, 0.47
    height = top - len(msgs) * gap - 0.3
    fig = plt.figure(figsize=(10.4, 7.6))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, xs[-1] + 0.75)
    ax.set_ylim(height - 0.2, top + 0.75)
    ax.axis("off")
    for x, name in zip(xs, names):
        face, edge = ((MEMORY, MEMORY_EDGE) if name.startswith(("Experience", "Audit")) else
                      (HUMAN, HUMAN_EDGE) if name == "Operator" else (STEP, STEP_EDGE))
        ax.add_patch(Rectangle((x - 0.62, top + 0.05), 1.24, 0.55, facecolor=face, edgecolor=edge, linewidth=1))
        ax.text(x, top + 0.325, name, ha="center", va="center", fontsize=8.4, color=INK, linespacing=1.1)
        ax.plot([x, x], [top + 0.05, height], color="#9a9a9a", linewidth=0.8, linestyle=(0, (4, 3)))
    act = Rectangle((xs[1] - 0.07, height + 0.05), 0.14, top - height - 0.15, facecolor="#dfe9f6", edgecolor=STEP_EDGE, linewidth=0.8)
    ax.add_patch(act)
    for k, (a_, b_, text, reply) in enumerate(msgs):
        yy = top - 0.35 - k * gap
        if a_ == b_:
            x0 = xs[a_] + 0.07
            ax.plot([x0, x0 + 0.35, x0 + 0.35], [yy + 0.08, yy + 0.08, yy - 0.12], color=INK, linewidth=0.9)
            ax.add_patch(FancyArrowPatch((x0 + 0.35, yy - 0.12), (x0, yy - 0.12), arrowstyle="-|>", mutation_scale=8,
                                         color=INK, linewidth=0.9, shrinkA=0, shrinkB=0))
            ax.text(x0 + 0.45, yy - 0.02, text, ha="left", va="center", fontsize=7.6, color=INK)
            continue
        x0 = xs[a_] + (0.07 if b_ > a_ else -0.07)
        x1 = xs[b_] - (0.07 if b_ > a_ else -0.07) if b_ == 1 or a_ == 1 else xs[b_]
        if b_ != 1:
            x1 = xs[b_]
        ax.add_patch(FancyArrowPatch((x0, yy), (x1, yy), arrowstyle="-|>", mutation_scale=9, color=INK,
                                     linewidth=0.9, linestyle="--" if reply else "-", shrinkA=0, shrinkB=0))
        left = min(x0, x1)
        ax.text(left + 0.08, yy + 0.09, text, ha="left", va="bottom", fontsize=7.6, color=INK)
    ax.text(0.1, height - 0.05, "Solid arrows: requests. Dashed arrows: replies. Values are from step 5 of the report demonstration (seed 40001, approval withheld).",
            fontsize=7.6, color=INK_2, va="top")
    out = OUT.parent / "sequence_flagged_step.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved", out)


def operator_mockup():
    """Report figure: design mock-up of the supervisor review screen (not implemented)."""
    fig = plt.figure(figsize=(9.6, 5.6))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 9.6)
    ax.set_ylim(0, 5.6)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.15, 0.15), 9.3, 5.3, boxstyle="round,pad=0,rounding_size=0.12",
                                facecolor="#f7f8fa", edgecolor="#5b6670", linewidth=1.2))
    ax.add_patch(Rectangle((0.15, 4.95), 9.3, 0.5, facecolor="#2b3640", edgecolor="none"))
    ax.text(0.4, 5.2, "Supervisor review   |   Junction 1   |   step 5 (t = 25 s)", color="white", fontsize=10.5,
            va="center", fontweight="bold")
    ax.text(9.25, 5.2, "DESIGN MOCK-UP: not implemented", color="#f2b134", fontsize=8.5, va="center", ha="right",
            fontweight="bold")

    def card(x, y, w, h, title, lines, edge="#c9d1d9", face="white"):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=face, edgecolor=edge, linewidth=1))
        ax.text(x + 0.15, y + h - 0.22, title, fontsize=8.6, fontweight="bold", color="#2b3640", va="center")
        for k, line in enumerate(lines):
            ax.text(x + 0.15, y + h - 0.55 - k * 0.27, line, fontsize=8.1, color=INK, va="center")

    card(0.4, 3.0, 2.9, 1.75, "Observations", ["N-S queue 28, E-W queue 11", "Sensor noise: about 4 cars (SD)",
                                                "Recent faults: 2 of last 12 events", "Safe mode at 3 faults"])
    card(0.4, 1.05, 2.9, 1.75, "Congestion belief", ["P(congestion) = 0.981", "Assumed detector rates:",
                                                     "sensitivity 0.85, false positive 0.10", "Not calibrated on real data"])
    card(3.5, 3.0, 2.9, 1.75, "Learned proposal", ["Proposed: E-W green", "Estimated return -264.3",
                                                   "Alternative: N-S green", "Estimated return -341.8"], edge=STEP_EDGE)
    card(3.5, 1.05, 2.9, 1.75, "Precedent (26 similar cases each)", ["N-S green: -0.15 cars after 20 s",
                                                                     "E-W green: +1.96 cars after 20 s",
                                                                     "Favours N-S: disagrees with", "the learned proposal"], edge=MEMORY_EDGE)
    card(6.6, 3.0, 2.6, 1.75, "Why review is required", ["P(congestion) > 0.90", "Precedent disagrees",
                                                         "Estimated return is not", "a safety guarantee"], edge=HUMAN_EDGE, face="#fff7f3")
    buttons = [("Approve E-W", "#2f8f62"), ("Override: N-S green", "#2a78d6"), ("Reject: use queue rule", "#d84a3a")]
    for k, (text, colour) in enumerate(buttons):
        ax.add_patch(FancyBboxPatch((6.6, 2.3 - k * 0.42), 2.6, 0.32, boxstyle="round,pad=0,rounding_size=0.06",
                                    facecolor=colour, edgecolor="none"))
        ax.text(7.9, 2.46 - k * 0.42, text, color="white", fontsize=8.4, ha="center", va="center", fontweight="bold")
    ax.add_patch(Rectangle((0.4, 0.38), 8.8, 0.45, facecolor="white", edgecolor="#c9d1d9", linewidth=1))
    ax.text(0.55, 0.605, "Reason (required for every decision): ______________________________     "
            "No automatic approval on timeout. Decision, reason and triggers are logged for later audit.",
            fontsize=7.9, color=INK_2, va="center")
    out = OUT.parent / "operator_mockup.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved", out)



def _canvas(w, h):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def _card(ax, x, yy, w, h, text, face, edge, size=8.2, bold=False):
    ax.add_patch(Rectangle((x, yy), w, h, facecolor=face, edgecolor=edge, linewidth=1))
    ax.text(x + 0.1, yy + h / 2, text, ha="left", va="center", fontsize=size, color=INK, linespacing=1.2,
            fontweight="bold" if bold else "normal")


def _save(fig, name):
    out = OUT.parent / name
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved", out)


def uncertainty_map():
    """Report figure: each source of uncertainty, its effect, the design response and what remains."""
    rows = [
        ("Noisy sensor\n(SD 4 cars)", "Queue order can reverse;\nwrong table row chosen",
         "Coarse bins; Bayes filter weighs\nevidence; precedent check", "Noise itself is not\ndetected as a fault"),
        ("Missing reading\n(20% in sensor test)", "No state, so no\nlearned action", "Fixed timing; Bayes predict-only;\ncounted as a fault (safe mode at 3)",
         "Fixed timing is poor\nunder changing load"),
        ("Incomplete environment\ndata", "No turns, pedestrians or\nneighbouring junctions", "Not handled: stated as a\nscope limit",
         "Action may be unsuitable\nwhile the estimate is right"),
        ("Human intervention", "Approval uncertain;\nfrequent requests", "Three review triggers; withheld\napproval gives the queue rule; logged",
         "Automation bias and\nroutine approval"),
        ("Failed command\n(10% in sensor test)", "Displayed phase differs\nfrom the request", "Acknowledged phase logged; local\nmin/max green still enforced; a fault",
         "Agent loses control\nwhile commands fail"),
    ]
    heads = ["Uncertainty", "Effect on the agent", "Response in this design", "Remaining risk"]
    xs, ws = [0.1, 2.35, 4.6, 7.55], [2.1, 2.1, 2.8, 2.1]
    styles = [("#f1f0ec", "#9b9a95"), (STEP, STEP_EDGE), (MEMORY, MEMORY_EDGE), (HUMAN, HUMAN_EDGE)]
    fig, ax = _canvas(9.75, 4.6)
    for x, w, h in zip(xs, ws, heads):
        ax.text(x + 0.1, 4.35, h, fontsize=9, fontweight="bold", color=INK, va="center")
    for r, row in enumerate(rows):
        yy = 3.45 - r * 0.8
        for c, (x, w, text) in enumerate(zip(xs, ws, row)):
            _card(ax, x, yy, w, 0.66, text, *styles[c], bold=(c == 0))
            if c < 3:
                ax.add_patch(FancyArrowPatch((x + w, yy + 0.33), (xs[c + 1], yy + 0.33), arrowstyle="-|>",
                                             mutation_scale=8, color=INK_2, linewidth=0.8, shrinkA=0, shrinkB=0))
    _save(fig, "uncertainty_map.png")


def signal_state_machine():
    """Appendix figure: local interlocks of the simulated signal controller, as in Environment.step()."""
    fig, ax = _canvas(9.0, 3.6)
    def state(x, yy, text, face=STEP, edge=STEP_EDGE, w=2.3, h=0.85):
        ax.add_patch(FancyBboxPatch((x - w / 2, yy - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.15",
                                    facecolor=face, edgecolor=edge, linewidth=1.1))
        ax.text(x, yy, text, ha="center", va="center", fontsize=8.2, color=INK, linespacing=1.2)
    def edge(a, b, text, rad=0.0, off=(0, 0.12)):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=10, color=INK, linewidth=0.9,
                                     connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))
        ax.text((a[0] + b[0]) / 2 + off[0], (a[1] + b[1]) / 2 + off[1], text, ha="center", va="center",
                fontsize=7.3, color=INK_2)
    state(1.4, 2.6, "Green, age 0-1\n(minimum green:\nrequests ignored)")
    state(4.5, 2.6, "Green, age 2-5\n(switch allowed\non request)")
    state(7.6, 2.6, "Green, age 6\n(maximum green:\nswitch forced)", HUMAN, HUMAN_EDGE)
    state(4.5, 0.75, "Clearance step\n(all-red, no discharge;\nother phase becomes green)", MEMORY, MEMORY_EDGE, w=3.0)
    edge((2.55, 2.6), (3.35, 2.6), "serve, age + 1")
    edge((5.65, 2.6), (6.45, 2.6), "serve, age + 1")
    edge((4.5, 2.17), (4.5, 1.18), "request switch\n(command accepted)", off=(0.75, 0))
    edge((7.6, 2.17), (5.6, 1.0), "switch even if\ncommand failed", rad=-0.2, off=(0.85, -0.05))
    edge((3.0, 0.75), (1.4, 2.17), "new phase,\nage 0", rad=-0.25, off=(0.55, -0.35))
    ax.text(4.5, 3.4, "Command failed (10% in the sensor-fault test): request rejected, current phase kept, age + 1",
            ha="center", fontsize=7.6, color=INK_2)
    _save(fig, "signal_state_machine.png")


def sqlite_schema():
    """Appendix figure: tables in the two SQLite files written and read by the demo."""
    fig, ax = _canvas(9.0, 2.45)
    def table(x, top, title, cols, note):
        w, rh = 2.75, 0.24
        ax.add_patch(Rectangle((x, top - rh), w, rh, facecolor=MEMORY_EDGE, edgecolor=MEMORY_EDGE))
        ax.text(x + 0.08, top - rh / 2, title, color="white", fontsize=8.4, fontweight="bold", va="center")
        for i, c in enumerate(cols):
            yy = top - rh * (i + 2)
            ax.add_patch(Rectangle((x, yy), w, rh, facecolor="white", edgecolor="#c9d1d9", linewidth=0.8))
            ax.text(x + 0.08, yy + rh / 2, c, fontsize=7.6, color=INK, va="center", family="monospace")
        ax.text(x, top - rh * (len(cols) + 1) - 0.18, note, fontsize=7.2, color=INK_2, va="top", linespacing=1.2)
    table(0.1, 2.35, "events (agent_memory.sqlite)", ["run_id   TEXT", "step     INTEGER", "payload  TEXT (JSON row)"],
          "Written after every demo step:\nfull audit record of the decision.")
    table(3.1, 2.35, "operator_decisions (agent_memory)", ["run_id    TEXT", "step      INTEGER", "decision  TEXT",
          "reason    TEXT", "triggers  TEXT"], "Written for every review:\napproved or withheld, and why.")
    table(6.1, 2.35, "experience (experience.sqlite)", ["ns, ew    INTEGER (queue bins)", "phase, age INTEGER",
          "action    INTEGER", "queue_change_20s INTEGER", "scenario, seed"],
          "17,123 validation events, read every\nstep; INDEX situation (ns, ew, phase, age).")
    _save(fig, "sqlite_schema.png")

if __name__ == "__main__":
    main()
    architecture()
    sequence()
    operator_mockup()
    uncertainty_map()
    signal_state_machine()
    sqlite_schema()
