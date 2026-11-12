import base64
import csv
import glob
import io
import json
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

# ---------- Load the session ----------
if len(sys.argv) > 1:
    stamp = sys.argv[1]
else:
    files = sorted(glob.glob("cycles_*.csv"))
    if not files:
        sys.exit("No cycles_*.csv files found. Run track.py first.")
    stamp = files[-1][len("cycles_"):-len(".csv")]

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


def smooth(a):
    k = np.ones(SMOOTH) / SMOOTH
    return np.column_stack([np.convolve(a[:, i], k, mode="valid") for i in range(2)])


# ---------- Per-cycle metrics ----------
per = []
for c in cycles:
    n = int(c["cycle"])
    d = {"n": n, "t_start": float(c["t_start"]), "t_end": float(c["t_end"]),
         "work": float(c["duration_sec"]), "dist_l": 0.0, "dist_r": 0.0,
         "idle": 0.0, "peak": 0.0, "direct": float("nan"), "ret": float("nan")}
    m = cyc == n
    if m.sum() >= SMOOTH + 2:
        edge = SMOOTH // 2
        tt, l, r = t[m][edge:-edge], smooth(lw[m]), smooth(rw[m])
        dl = np.linalg.norm(np.diff(l, axis=0), axis=1)
        dr = np.linalg.norm(np.diff(r, axis=0), axis=1)
        dt = np.diff(tt)
        dt[dt == 0] = 1e-6
        sl, sr = dl / dt, dr / dt
        d["dist_l"], d["dist_r"] = dl.sum(), dr.sum()
        d["idle"] = dt[(sl < IDLE_SPEED) & (sr < IDLE_SPEED)].sum()
        d["peak"] = max(sl.max(), sr.max())
        main, path = (l, dl.sum()) if dl.sum() >= dr.sum() else (r, dr.sum())
        if path > 20:
            d["direct"] = min(1.0, np.linalg.norm(main[-1] - main[0]) / path)
    per.append(d)
for a, b in zip(per, per[1:]):
    a["ret"] = b["t_start"] - a["t_end"]


def col(key):
    x = np.array([p[key] for p in per], dtype=float)
    return x[~np.isnan(x)]


def stats(x):
    if len(x) == 0:
        return None
    sd = x.std(ddof=1) if len(x) > 1 else 0.0
    return x.mean(), sd, x.min(), x.max()


# ---------- Summary numbers ----------
work = col("work")
n = len(work)
mean_w, sd_w, min_w, max_w = stats(work)
cv = 100 * sd_w / mean_w
ret = col("ret")
full = np.array([b["t_start"] - a["t_start"] for a, b in zip(per, per[1:])])
base = full.mean() if len(full) else mean_w          # time per unit, incl. return
normal_time = base * RATING
standard_time = normal_time * (1 + ALLOWANCE)
slope = np.polyfit(np.arange(1, n + 1), work, 1)[0] if n >= 3 else 0.0
half = n // 2
first_half, second_half = (work[:half].mean(), work[half:].mean()) if n >= 4 else (mean_w, mean_w)
outliers = [p["n"] for p in per if sd_w > 0 and abs(p["work"] - mean_w) > 2 * sd_w]
needed = math.ceil((1.96 * sd_w / (0.05 * mean_w)) ** 2) if sd_w > 0 else 1
tot_l, tot_r = col("dist_l").sum(), col("dist_r").sum()
left_pct = 100 * tot_l / (tot_l + tot_r) if tot_l + tot_r > 0 else 50
idle_pct = 100 * col("idle").sum() / work.sum()
direct = col("direct")

# ---------- Charts ----------
BLUE, ORANGE, GRAY, RED = "#2f6fb0", "#e08a2e", "#9aa3ad", "#c0392b"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def to_img(fig):
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


x = np.arange(1, n + 1)

fig, ax = plt.subplots(figsize=(6, 3.4))
ax.bar(x, work, color=[RED if i in outliers else BLUE for i in x])
ax.axhline(mean_w, color="black", linestyle="--", linewidth=1, label=f"mean {mean_w:.2f}s")
ax.axhspan(mean_w - sd_w, mean_w + sd_w, color=GRAY, alpha=0.2, label="+/- 1 std dev")
if n >= 3:
    ax.plot(x, np.polyval(np.polyfit(x, work, 1), x), color=ORANGE, label="trend")
ax.set_xlabel("Cycle"); ax.set_ylabel("Work time (s)"); ax.set_xticks(x)
ax.legend(fontsize=8)
img_cycles = to_img(fig)

fig, ax = plt.subplots(figsize=(6, 3.4))
moving = work - col("idle") if len(col("idle")) == n else work
ax.bar(x, moving, color=BLUE, label="Moving")
ax.bar(x, work - moving, bottom=moving, color=ORANGE, label="Low motion")
if len(ret):
    ax.bar(x[:len(ret)], ret, bottom=work[:len(ret)], color=GRAY, label="Return to START")
ax.set_xlabel("Cycle"); ax.set_ylabel("Seconds"); ax.set_xticks(x)
ax.legend(fontsize=8)
img_break = to_img(fig)

fig, ax = plt.subplots(figsize=(6, 3.4))
active = cyc > 0
if active.sum() > SMOOTH:
    ax.plot(lw[active, 0], lw[active, 1], color=ORANGE, linewidth=0.8, alpha=0.8, label="Left wrist")
    ax.plot(rw[active, 0], rw[active, 1], color=BLUE, linewidth=0.8, alpha=0.8, label="Right wrist")
if os.path.exists("zones.json"):
    with open("zones.json") as f:
        for z, name, c in zip(json.load(f), ["START", "END"], ["green", RED]):
            ax.add_patch(plt.Rectangle((z[0], z[1]), z[2] - z[0], z[3] - z[1], fill=False, color=c, linewidth=1.5))
            ax.text(z[0], z[1] - 6, name, color=c, fontsize=8)
ax.invert_yaxis(); ax.set_aspect("equal", adjustable="datalim")
ax.set_xlabel("x (px)"); ax.set_ylabel("y (px)")
ax.legend(fontsize=8)
img_path = to_img(fig)

fig, ax = plt.subplots(figsize=(6, 3.4))
ax.bar(x - 0.2, col("dist_l"), width=0.4, color=ORANGE, label="Left hand")
ax.bar(x + 0.2, col("dist_r"), width=0.4, color=BLUE, label="Right hand")
ax.set_xlabel("Cycle"); ax.set_ylabel("Travel (px)"); ax.set_xticks(x)
ax.legend(fontsize=8)
img_dist = to_img(fig)

# ---------- HTML report ----------
def fmt(v, digits=2):
    return "-" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{digits}f}"


kpis = [
    ("Cycles observed", str(n)),
    ("Mean work time", f"{mean_w:.2f} s"),
    ("Consistency (CV)", f"{cv:.1f}%"),
    ("Standard time", f"{standard_time:.2f} s"),
    ("Output at standard", f"{3600 / standard_time:.0f} / hr"),
    ("Low-motion share", f"{idle_pct:.0f}%"),
]
stat_rows = [
    ("Work time, START to END (s)", stats(work)),
    ("Full cycle, START to START (s)", stats(full)),
    ("Return time, END to START (s)", stats(ret)),
    ("Left hand travel (px)", stats(col("dist_l"))),
    ("Right hand travel (px)", stats(col("dist_r"))),
    ("Low-motion time (s)", stats(col("idle"))),
    ("Peak hand speed (px/s)", stats(col("peak"))),
    ("Path directness (1 = straight line)", stats(direct)),
]
findings = [
    f"Cycle time trend: {slope:+.3f} s per cycle "
    f"(first half {first_half:.2f} s, second half {second_half:.2f} s).",
    f"Hand use: left {left_pct:.0f}% / right {100 - left_pct:.0f}% of total travel.",
    f"Outlier cycles (more than 2 std dev from mean): {', '.join(map(str, outliers)) or 'none'}.",
    f"Observations needed for +/-5% accuracy at 95% confidence: {needed} "
    f"({'enough data' if n >= needed else f'record at least {needed - n} more'}).",
    f"Standard time uses a {RATING:.0%} rating and {ALLOWANCE:.0%} allowance on the "
    f"{'full cycle' if len(full) else 'work'} time of {base:.2f} s.",
]

html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Motion Study Report {stamp}</title><style>
body{{font-family:-apple-system,Helvetica,Arial,sans-serif;background:#f3f5f8;color:#1c2530;margin:0;padding:32px}}
.wrap{{max-width:1100px;margin:auto}} h1{{margin:0 0 4px;font-size:26px}} .sub{{color:#5b6775;margin-bottom:24px}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin-bottom:20px}}
.kpi,.card{{background:#fff;border-radius:10px;padding:16px;box-shadow:0 1px 3px rgba(0,0,0,.08)}}
.kpi .v{{font-size:24px;font-weight:700;color:#2f6fb0}} .kpi .l{{font-size:12px;color:#5b6775;text-transform:uppercase;letter-spacing:.04em}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(440px,1fr));gap:12px;margin-bottom:12px}}
.card h2{{font-size:15px;margin:0 0 8px}} .card img{{width:100%}} .card{{margin-bottom:12px}}
table{{border-collapse:collapse;width:100%;font-size:14px}} th,td{{padding:7px 10px;text-align:right;border-bottom:1px solid #e6eaef}}
th:first-child,td:first-child{{text-align:left}} th{{color:#5b6775;font-weight:600}} li{{margin-bottom:6px}}
</style></head><body><div class="wrap">
<h1>Motion Study Report</h1><div class="sub">Session {stamp} &middot; {n} cycles &middot; {t[-1]:.0f} s recorded</div>
<div class="kpis">{''.join(f'<div class="kpi"><div class="v">{v}</div><div class="l">{l}</div></div>' for l, v in kpis)}</div>
<div class="grid">
<div class="card"><h2>Cycle time by cycle</h2><img src="data:image/png;base64,{img_cycles}"></div>
<div class="card"><h2>Where the time goes</h2><img src="data:image/png;base64,{img_break}"></div>
<div class="card"><h2>Hand paths during cycles (spaghetti diagram)</h2><img src="data:image/png;base64,{img_path}"></div>
<div class="card"><h2>Hand travel per cycle</h2><img src="data:image/png;base64,{img_dist}"></div>
</div>
<div class="card"><h2>Findings</h2><ul>{''.join(f'<li>{x}</li>' for x in findings)}</ul></div>
<div class="card"><h2>Summary statistics</h2><table><tr><th>Metric</th><th>Mean</th><th>Std dev</th><th>Min</th><th>Max</th></tr>
{''.join(f'<tr><td>{name}</td>' + ''.join(f'<td>{fmt(v)}</td>' for v in s) + '</tr>' for name, s in stat_rows if s)}</table></div>
<div class="card"><h2>Cycle detail</h2><table><tr><th>Cycle</th><th>Work (s)</th><th>Return (s)</th><th>Low motion (s)</th>
<th>Left travel (px)</th><th>Right travel (px)</th><th>Peak speed (px/s)</th><th>Directness</th></tr>
{''.join(f"<tr><td>{p['n']}</td><td>{fmt(p['work'])}</td><td>{fmt(p['ret'])}</td><td>{fmt(p['idle'])}</td><td>{fmt(p['dist_l'], 0)}</td><td>{fmt(p['dist_r'], 0)}</td><td>{fmt(p['peak'], 0)}</td><td>{fmt(p['direct'])}</td></tr>" for p in per)}</table></div>
</div></body></html>"""

out = f"report_{stamp}.html"
with open(out, "w") as f:
    f.write(html)

print(f"\nSession {stamp}: {n} cycles")
print(f"Mean work time {mean_w:.2f}s, std dev {sd_w:.2f}s, consistency {cv:.1f}%")
print(f"Standard time {standard_time:.2f}s -> {3600 / standard_time:.0f} units per hour")
for line in findings[:4]:
    print("- " + line)
print(f"\nReport saved: {out}")
webbrowser.open("file://" + os.path.abspath(out))