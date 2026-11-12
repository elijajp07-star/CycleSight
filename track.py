import csv
import json
import os
import sys
import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
    min_tracking_confidence=0.3,
)
landmarker = vision.HandLandmarker.create_from_options(options)
connections = vision.HandLandmarksConnections.HAND_CONNECTIONS

PALM = [0, 5, 9, 13, 17]   # wrist and knuckles; their average is the palm center
MAX_H, MAX_W = 900, 1400   # shrink big frames (like phone video) to fit the screen

# Video file given on the command line, or the live webcam if none
video = os.path.expanduser(sys.argv[1]) if len(sys.argv) > 1 else None
if video:
    name = os.path.splitext(os.path.basename(video))[0]
    ZONE_FILE = f"zones_{name}.json"   # each video keeps its own zones
else:
    ZONE_FILE = "zones.json"
ZONE_NAMES = ["START", "END"]
ZONE_COLORS = [(0, 200, 0), (0, 0, 255)]
HAND_COLORS = [(0, 140, 255), (255, 100, 0)]   # left = orange, right = blue
HAND_NAMES = ["L", "R"]
HOLD_FRAMES = 3  # hand must stay in a zone this many frames to count

zones = []
if os.path.exists(ZONE_FILE):
    with open(ZONE_FILE) as f:
        zones = json.load(f)

drag = {"p0": None, "p1": None}


def on_mouse(event, x, y, flags, param):
    if len(zones) >= 2:
        return
    if event == cv2.EVENT_LBUTTONDOWN:
        drag["p0"], drag["p1"] = (x, y), (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and drag["p0"]:
        drag["p1"] = (x, y)
    elif event == cv2.EVENT_LBUTTONUP and drag["p0"]:
        x0, y0 = drag["p0"]
        zone = [min(x0, x), min(y0, y), max(x0, x), max(y0, y)]
        drag["p0"] = None
        if zone[2] - zone[0] > 10 and zone[3] - zone[1] > 10:
            zones.append(zone)
            if len(zones) == 2:
                with open(ZONE_FILE, "w") as f:
                    json.dump(zones, f)


def inside(pt, z):
    return z[0] <= pt[0] <= z[2] and z[1] <= pt[1] <= z[3]


def dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def assign(found, last, width):
    """Decide which detected hand is slot 0 (left of image) and slot 1 (right).
    Hands are matched to wherever each one was last seen, so a hand keeps its
    label even when the other one disappears for a while."""
    seen = [None, None]
    if len(found) >= 2:
        a, b = found[0], found[1]
        if last[0] and last[1]:
            straight = dist(a, last[0]) + dist(b, last[1])
            swapped = dist(a, last[1]) + dist(b, last[0])
            seen = [a, b] if straight <= swapped else [b, a]
        elif last[0]:
            seen = [a, b] if dist(a, last[0]) <= dist(b, last[0]) else [b, a]
        elif last[1]:
            seen = [b, a] if dist(a, last[1]) <= dist(b, last[1]) else [a, b]
        else:
            seen = [a, b] if a[0] <= b[0] else [b, a]
    elif len(found) == 1:
        a = found[0]
        if last[0] and last[1]:
            slot = 0 if dist(a, last[0]) <= dist(a, last[1]) else 1
        elif last[0]:
            slot = 0 if dist(a, last[0]) < 0.25 * width else 1
        elif last[1]:
            slot = 1 if dist(a, last[1]) < 0.25 * width else 0
        else:
            slot = 0 if a[0] < width / 2 else 1
        seen[slot] = a
    return seen


def fill(values):
    """Fill frames where a hand was not detected with its nearest known position."""
    out, prev = [], None
    for v in values:
        prev = v if v is not None else prev
        out.append(prev)
    first = next((v for v in out if v is not None), (0, 0))
    return [v if v is not None else first for v in out]


def draw_overlay(frame, msg, footer):
    h = frame.shape[0]
    for i, z in enumerate(zones):
        cv2.rectangle(frame, (z[0], z[1]), (z[2], z[3]), ZONE_COLORS[i], 2)
        cv2.putText(frame, ZONE_NAMES[i], (z[0] + 5, z[1] + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, ZONE_COLORS[i], 2)
    if drag["p0"]:
        cv2.rectangle(frame, drag["p0"], drag["p1"], (255, 255, 255), 1)
    for text, y in ((msg, 30), (footer, h - 15)):
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4)
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)


cv2.namedWindow("Hands")
cv2.setMouseCallback("Hands", on_mouse)

cap = cv2.VideoCapture(video if video else 0)
if not cap.isOpened():
    sys.exit(f"Could not open {video if video else 'the webcam'}.")
fps = cap.get(cv2.CAP_PROP_FPS) or 30
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if video else 0
start = time.time()
stamp = time.strftime("%Y%m%d_%H%M%S")
frame_no = 0


def read_frame():
    """Return (ok, frame, time in seconds). Video time comes from the video itself,
    so results are the same no matter how fast the computer processes it."""
    global frame_no
    ok, frame = cap.read()
    if not ok:
        return False, None, 0.0
    if video:
        ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        t = ms / 1000 if ms > 0 else frame_no / fps
    else:
        t = time.time() - start
    frame_no += 1
    h, w = frame.shape[:2]
    scale = min(1.0, MAX_H / h, MAX_W / w)
    if scale < 1.0:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
    return True, frame, t


# For a video: pause on the first frame so the zones can be drawn before it plays
pending = None
if video:
    pending = read_frame()
    if not pending[0]:
        sys.exit(f"Could not read any frames from {video}.")
    while True:
        view = pending[1].copy()
        if len(zones) < 2:
            msg = f"Drag a box for the {ZONE_NAMES[len(zones)]} zone"
        else:
            msg = "SPACE = start analysis"
        draw_overlay(view, msg, "R = redraw zones   Q = quit")
        cv2.imshow("Hands", view)
        key = cv2.waitKey(20) & 0xFF
        if key == ord(" ") and len(zones) == 2:
            break
        if key == ord("r"):
            zones.clear()
        if key == ord("q"):
            sys.exit("Cancelled.")

cyc_log = open(f"cycles_{stamp}.csv", "w", newline="")
cyc_writer = csv.writer(cyc_log)
cyc_writer.writerow(["cycle", "t_start", "t_end", "duration_sec"])

frames = []            # (t, left position or None, right position or None, cycle)
last = [None, None]    # last known position of each hand
state = "START"
hold = 0
cycle_start = None
durations = []

last_ms = -1
MIN_CYCLE = 0.5    # seconds; a faster START -> END is ignored as a false trigger
EXIT_FRAMES = 5    # a zone must be empty this many frames before it can trigger again
empty = [0, 0]     # frames in a row each zone has been empty
ready = [False, False]

while True:
    ok, frame, t = pending if pending else read_frame()
    pending = None
    if not ok:
        break
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    last_ms = max(last_ms + 1, int(t * 1000))
    result = landmarker.detect_for_video(image, last_ms)

    found = []
    for hand in result.hand_landmarks:
        pts = [(int(p.x * w), int(p.y * h)) for p in hand]
        for c in connections:
            cv2.line(frame, pts[c.start], pts[c.end], (220, 220, 220), 1)
        cx = sum(pts[i][0] for i in PALM) // len(PALM)
        cy = sum(pts[i][1] for i in PALM) // len(PALM)
        found.append((cx, cy))

    seen = assign(found, last, w)
    for i in (0, 1):
        if seen[i]:
            last[i] = seen[i]
            cv2.circle(frame, seen[i], 12, HAND_COLORS[i], 3)
            cv2.putText(frame, HAND_NAMES[i], (seen[i][0] + 16, seen[i][1] + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, HAND_COLORS[i], 2)

    if len(zones) == 2:
        # A zone counts as occupied if any hand's last known position is in it
        occupied = [any(p is not None and inside(p, z) for p in last) for z in zones]
        for i in (0, 1):
            empty[i] = 0 if occupied[i] else empty[i] + 1
            if empty[i] >= EXIT_FRAMES:
                ready[i] = True   # zone was cleared, so it may trigger again
        idx = 0 if state == "START" else 1
        hold = hold + 1 if occupied[idx] else 0
        if hold >= HOLD_FRAMES and ready[idx]:
            if state == "START":
                hold = 0
                ready[0] = False
                cycle_start = t
                state = "END"
            elif t - cycle_start >= MIN_CYCLE:
                hold = 0
                ready[1] = False
                durations.append(t - cycle_start)
                cyc_writer.writerow([len(durations), round(cycle_start, 3),
                                     round(t, 3), round(t - cycle_start, 3)])
                state = "START"

    in_cycle = len(durations) + 1 if state == "END" else 0
    frames.append((round(t, 3), seen[0], seen[1], in_cycle))

    if len(zones) < 2:
        msg = f"Drag a box for the {ZONE_NAMES[len(zones)]} zone"
    else:
        last_time = f"{durations[-1]:.2f}s" if durations else "-"
        msg = f"Cycles: {len(durations)}  Last: {last_time}  Waiting for: {state}"
    if video:
        footer = f"Hands in view: {len(found)}   Frame {frame_no}/{total}   Q = stop"
    else:
        footer = f"Hands in view: {len(found)}   Q = quit   R = redraw zones"
    draw_overlay(frame, msg, footer)

    cv2.imshow("Hands", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("r") and not video:
        zones.clear()
        state, hold = "START", 0

cyc_log.close()
cap.release()
cv2.destroyAllWindows()

# Save the hand positions, filling short gaps where a hand was not detected
left = fill([f[1] for f in frames])
right = fill([f[2] for f in frames])
with open(f"session_{stamp}.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["t_sec", "lw_x", "lw_y", "rw_x", "rw_y", "cycle", "l_seen", "r_seen"])
    for (t, l, r, c), lp, rp in zip(frames, left, right):
        writer.writerow([t, lp[0], lp[1], rp[0], rp[1], c, int(l is not None), int(r is not None)])

if frames:
    for i, name in enumerate(["Left", "Right"]):
        pct = 100 * sum(1 for f in frames if f[i + 1] is not None) / len(frames)
        print(f"{name} hand tracked in {pct:.0f}% of frames")
if durations:
    avg = sum(durations) / len(durations)
    print(f"{len(durations)} cycles, average {avg:.2f}s, "
          f"fastest {min(durations):.2f}s, slowest {max(durations):.2f}s")
else:
    print("No complete cycles recorded.")
print(f"Session: {stamp}" + (f"  (from {os.path.basename(video)})" if video else ""))