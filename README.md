# CycleSight

A computer-vision time and motion study tool. CycleSight tracks a worker's hands
in video, automatically detects each work cycle, and reports cycle time, hand
travel, and standard time. Record two methods of doing the same task and it
quantifies the improvement between them.

Built with Python, OpenCV, and MediaPipe.

## What it does

- Tracks both hands in a live webcam feed or a recorded video (phone video works)
- Detects work cycles automatically using START and END zones drawn on screen
- Generates a report for a session: cycle times, consistency, standard time,
  output per hour, hand travel, low-motion time, path directness, hand balance
- Compares two methods side by side with percent change for every metric and a
  statistical significance test on cycle time
- Produces charts: cycle time by cycle, time breakdown, hand-path (spaghetti)
  diagrams

## How it works

1. `track.py` runs MediaPipe hand tracking on each frame and logs the palm
   position of each hand. A cycle is counted when a hand enters the START zone
   and then the END zone.
2. `analyze.py` turns one recorded session into an HTML report.
3. `compare.py` compares two sessions (baseline method vs improved method).

## Setup

    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    curl -L -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task

## Usage

Record from the webcam:

    python track.py

Or analyze a video file:

    python track.py path/to/video.mov

Draw the START and END zones, perform the task, then:

    python analyze.py
    python compare.py

## Industrial engineering concepts used

- Time study: observed time, performance rating, allowances, standard time
- Required number of observations for a target accuracy and confidence level
- Motion economy: hand travel distance, two-handed work, path directness
- Method comparison with a permutation test for statistical significance


## Roadmap

- Calibrate pixels to real-world distance
- Detect grasp and release to break cycles into motion elements
- Ergonomic posture scoring
