import base64
import csv
import glob
import io
import math
import os
import sys
import webbrowser
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IDLE_SPEED = 40    # pixels per second; slower than this counts as low motion
SMOOTH = 5         # frames of smoothing to remove camera jitter
ALLOWANCE = 0.15   # personal, fatigue and delay allowance for standard time
RATING = 1.00      # performance rating (1.00 = normal pace)
SHIFT_HOURS = 8


def has_cycles(path):
    with open(path) as f:
        return sum(1 for _ in f) > 1


def smooth(a):
    k = np.ones(SMOOTH) / SMOOTH
    return np.column_stack([np.convolve(a[:, i], k, mode="valid") for i in range(2)])


def load(stamp):
    """Read one session and return its per-cycle metrics and summary numbers."""
    with open(f"cycles_{stamp}.csv") as f:
        cycles = list(csv.DictReader(f))
    with open(f"session_{stamp}.csv") as f:
        rows = list(csv.DictReader(f))
    if not cycles:
        sys.exit(f"Session {stamp} has no completed cycles.")
    t = np.array([float(r["t_sec"]) for r in rows])
    lw = np.array([[float(r["lw_x"]), float(r["lw_y"])] for r in rows])
    rw = np.array([[float(r["rw_x"]), float(r["rw_y"])] for r in rows])
    cyc = np.array([int(r["cycle"]) for r in rows])

    work, dist_l, dist_r, idle, peak, direct = [], [], [], [], [], []
    for c in cycles:
        work.append(float(c["duration_sec"]))
        m = cyc == int(c["cycle"])
        if m.sum() < SMOOTH + 2:
            continue
        edge = SMOOTH // 2
        tt, l, r = t[m][edge:-edge], smooth(lw[m]), smooth(rw[m])
        dl = np.linalg.norm(np.diff(l, axis=0), axis=1)
        dr = np.linalg.norm(np.diff(r, axis=0), axis=1)
        dt = np.diff(tt)
        dt[dt == 0] = 1e-6
        dist_l.append(dl.sum()); dist_r.append(dr.sum())
        idle.append(dt[(dl / dt < IDLE_SPEED) & (dr / dt < IDLE_SPEED)].sum())
        peak.append(max((dl / dt).max(), (dr / dt).max()))
        main, path = (l, dl.sum()) if dl.sum() >= dr.sum() else (r, dr.sum())
        if path > 20:
            direct.append(min(1.0, np.linalg.norm(main[-1] - main[0]) / path))

    starts = np.array([float(c["t_start"]) for c in cycles])
    ends = np.array([float(c["t_end"]) for c in cycles])
    work = np.array(work)
    full = np.diff(starts)
    ret = starts[1:] - ends[:-1]
    mean = lambda x: float(np.mean(x)) if len(x) else float("nan")
    sd = float(work.std(ddof=1)) if len(work) > 1 else 0.0
    base = mean(full) if len(full) else mean(work)
    std_time = base * RATING * (1 + ALLOWANCE)
    return {
        "stamp": stamp, "n": len(work), "work": work, "lw": lw[cyc > 0], "rw": rw[cyc > 0],
        "m": {
            "work": mean(work), "full": mean(full), "ret": mean(ret), "sd": sd,
            "cv": 100 * sd / mean(work), "std_time": std_time, "out": 3600 / std_time,
            "dist_l": mean(dist_l), "dist_r": mean(dist_r),
            "dist": mean(dist_l) + mean(dist_r), "idle": mean(idle),
            "peak": mean(peak), "direct": mean(direct),
        },
    }


# ---------- Pick the two sessions ----------
if len(sys.argv) == 3:
    stamp_a, stamp_b = sys.argv[1], sys.argv[2]
else:
    good = [f[len("cycles_"):-len(".csv")] for f in sorted(glob.glob("cycles_*.csv")) if has_cycles(f)]
    if len(good) < 2:
        sys.exit("Need two recorded sessions with completed cycles to compare.")
    stamp_a, stamp_b = good[-2], good[-1]
A, B = load(stamp_a), load(stamp_b)

# ---------- Is the cycle time difference real? (permutation test) ----------
rng = np.random.default_rng(0)
pooled = np.concatenate([A["work"], B["work"]])
observed = B["work"].mean() - A["work"].mean()
count = 0
TRIALS = 10000
for _ in range(TRIALS):
    rng.shuffle(pooled)
    if abs(pooled[A["n"]:].mean() - pooled[:A["n"]].mean()) >= abs(observed):
        count += 1
p_value = (count + 1) / (TRIALS + 1)
significant = p_value < 0.05

# ---------- Metric table ----------
# (label, key, digits, which direction is better: -1 lower, +1 higher, 0 neither)
METRICS = [
    ("Work time, START to END (s)", "work", 2, -1),
    ("Full cycle, START to START (s)", "full", 2, -1),
    ("Return time, END to START (s)", "ret", 2, -1),
    ("Cycle time std dev (s)", "sd", 2, -1),
    ("Consistency, CV (%)", "cv", 1, -1),
    ("Standard time (s)", "std_time", 2, -1),
    ("Output at standard (units/hr)", "out", 0, +1),
    ("Left hand travel per cycle (px)", "dist_l", 0, -1),
    ("Right hand travel per cycle (px)", "dist_r", 0, -1),
    ("Total hand travel per cycle (px)", "dist", 0, -1),
    ("Low-motion time per cycle (s)", "idle", 2, -1),
    ("Peak hand speed (px/s)", "peak", 0, 0),
    ("Path directness (1 = straight)", "direct", 2, +1),
]


def pct(key):
    a, b = A["m"][key], B["m"][key]
    if math.isnan(a) or math.isnan(b) or a == 0:
        return float("nan")
    return 100 * (b - a) / a


table = []
for label, key, digits, better in METRICS:
    a, b, change = A["m"][key], B["m"][key], pct(key)
    if math.isnan(change):
        verdict = "-"
    elif better == 0 or abs(change) < 0.5:
        verdict = "same" if abs(change) < 0.5 else "-"
    else:
        verdict = "better" if change * better > 0 else "worse"
    table.append((label, a, b, change, verdict, digits))

time_change = pct("work")
out_change = pct("out")
extra_units = SHIFT_HOURS * (B["m"]["out"] - A["m"]["out"])
word = "faster" if time_change < 0 else "slower"
headline = f"Method B is {abs(time_change):.1f}% {word} than Method A"
confidence = (f"The cycle time difference is statistically significant (p = {p_value:.3f})."
              if significant else
              f"The cycle time difference is NOT statistically significant (p = {p_value:.3f}); "
              f"it could be normal variation. Record more cycles to be sure.")

# ---------- Charts ----------
CA, CB = "#9aa3ad", "#2f6fb0"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def to_img(fig):
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


fig, ax = plt.subplots(figsize=(6, 3.4))
bp = ax.boxplot([A["work"], B["work"]], widths=0.5, patch_artist=True, showfliers=False)
for patch, c in zip(bp["boxes"], [CA, CB]):
    patch.set_facecolor(c); patch.set_alpha(0.5)
for i, (s, c) in enumerate([(A, CA), (B, CB)], start=1):
    ax.scatter(np.full(s["n"], i) + rng.uniform(-0.08, 0.08, s["n"]), s["work"], color=c, s=18, zorder=3)
ax.set_xticks([1, 2]); ax.set_xticklabels(["Method A", "Method B"])
ax.set_ylabel("Work time (s)")
img_box = to_img(fig)

fig, ax = plt.subplots(figsize=(6, 3.4))
ax.plot(range(1, A["n"] + 1), A["work"], marker="o", color=CA, label="Method A")
ax.plot(range(1, B["n"] + 1), B["work"], marker="o", color=CB, label="Method B")
ax.set_xlabel("Cycle"); ax.set_ylabel("Work time (s)"); ax.legend(fontsize=8)
img_runs = to_img(fig)

fig, ax = plt.subplots(figsize=(6, 3.4))
for i, s in enumerate([A, B]):
    m = s["m"]
    idle = 0 if math.isnan(m["idle"]) else m["idle"]
    ret = 0 if math.isnan(m["ret"]) else m["ret"]
    ax.bar(i, m["work"] - idle, color="#2f6fb0", label="Moving" if i == 0 else None)
    ax.bar(i, idle, bottom=m["work"] - idle, color="#e08a2e", label="Low motion" if i == 0 else None)
    ax.bar(i, ret, bottom=m["work"], color="#c9ced4", label="Return to START" if i == 0 else None)
ax.set_xticks([0, 1]); ax.set_xticklabels(["Method A", "Method B"])
ax.set_ylabel("Average seconds per cycle"); ax.legend(fontsize=8)
img_break = to_img(fig)

fig, axes = plt.subplots(1, 2, figsize=(6, 3.4), sharex=True, sharey=True)
for ax, s, name in zip(axes, [A, B], ["Method A", "Method B"]):
    ax.plot(s["lw"][:, 0], s["lw"][:, 1], color="#e08a2e", linewidth=0.7, label="Left")
    ax.plot(s["rw"][:, 0], s["rw"][:, 1], color="#2f6fb0", linewidth=0.7, label="Right")
    ax.set_title(name, fontsize=10); ax.set_aspect("equal", adjustable="box")
axes[0].invert_yaxis(); axes[0].legend(fontsize=7)
img_path = to_img(fig)

# ---------- HTML report ----------
def fmt(v, digits):
    return "-" if math.isnan(v) else f"{v:.{digits}f}"


def signed(v, unit="%"):
    return "-" if math.isnan(v) else f"{v:+.1f}{unit}"


color = {"better": "#1e8e3e", "worse": "#c0392b", "same": "#5b6775", "-": "#5b6775"}
kpis = [
    ("Cycle time", signed(time_change), time_change < 0),
    ("Output per hour", signed(out_change), out_change > 0),
    ("Hand travel", signed(pct("dist")), pct("dist") < 0),
    ("Variability (CV)", signed(pct("cv")), pct("cv") < 0),
    (f"Units per {SHIFT_HOURS}-hr shift", f"{extra_units:+.0f}", extra_units > 0),
]
body_rows = "".join(
    f"<tr><td>{label}</td><td>{fmt(a, d)}</td><td>{fmt(b, d)}</td><td>{signed(ch)}</td>"
    f"<td style='color:{color[v]};font-weight:600'>{v}</td></tr>"
    for label, a, b, ch, v, d in table)

html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Method Comparison</title><style>
body{{font-family:-apple-system,Helvetica,Arial,sans-serif;background:#f3f5f8;color:#1c2530;margin:0;padding:32px}}
.wrap{{max-width:1100px;margin:auto}} h1{{margin:0 0 4px;font-size:26px}} .sub{{color:#5b6775;margin-bottom:20px}}
.hero{{background:#fff;border-left:6px solid {'#1e8e3e' if time_change < 0 else '#c0392b'};border-radius:10px;padding:18px 20px;margin-bottom:16px;box-shadow:0 1px 3px rgba(0,0,0,.08)}}
.hero .big{{font-size:22px;font-weight:700;margin-bottom:4px}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin-bottom:16px}}
.kpi,.card{{background:#fff;border-radius:10px;padding:16px;box-shadow:0 1px 3px rgba(0,0,0,.08)}}
.kpi .v{{font-size:24px;font-weight:700}} .kpi .l{{font-size:12px;color:#5b6775;text-transform:uppercase;letter-spacing:.04em}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(440px,1fr));gap:12px;margin-bottom:12px}}
.card h2{{font-size:15px;margin:0 0 8px}} .card img{{width:100%}} .card{{margin-bottom:12px}}
table{{border-collapse:collapse;width:100%;font-size:14px}} th,td{{padding:7px 10px;text-align:right;border-bottom:1px solid #e6eaef}}
th:first-child,td:first-child{{text-align:left}} th{{color:#5b6775;font-weight:600}}
</style></head><body><div class="wrap">
<h1>Method Comparison</h1>
<div class="sub">Method A: session {A['stamp']} ({A['n']} cycles) &middot; Method B: session {B['stamp']} ({B['n']} cycles)</div>
<div class="hero"><div class="big">{headline}</div>{confidence}</div>
<div class="kpis">{''.join(f'<div class="kpi"><div class="v" style="color:{color["better"] if good else color["worse"]}">{v}</div><div class="l">{l}</div></div>' for l, v, good in kpis)}</div>
<div class="card"><h2>All metrics, Method A vs Method B</h2>
<table><tr><th>Metric</th><th>Method A</th><th>Method B</th><th>Change</th><th>Result</th></tr>{body_rows}</table></div>
<div class="grid">
<div class="card"><h2>Cycle time distribution</h2><img src="data:image/png;base64,{img_box}"></div>
<div class="card"><h2>Cycle time by cycle</h2><img src="data:image/png;base64,{img_runs}"></div>
<div class="card"><h2>Where the time goes</h2><img src="data:image/png;base64,{img_break}"></div>
<div class="card"><h2>Hand paths (spaghetti diagrams)</h2><img src="data:image/png;base64,{img_path}"></div>
</div></div></body></html>"""

out = f"compare_{A['stamp']}_vs_{B['stamp']}.html"
with open(out, "w") as f:
    f.write(html)

print(f"\nMethod A = session {A['stamp']} ({A['n']} cycles)")
print(f"Method B = session {B['stamp']} ({B['n']} cycles)\n")
print(f"{'Metric':<36}{'A':>9}{'B':>9}{'Change':>9}  Result")
for label, a, b, ch, v, d in table:
    print(f"{label:<36}{fmt(a, d):>9}{fmt(b, d):>9}{signed(ch):>9}  {v}")
print(f"\n{headline}")
print(confidence)
print(f"\nReport saved: {out}")
webbrowser.open("file://" + os.path.abspath(out))