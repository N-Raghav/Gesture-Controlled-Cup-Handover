# Viam Hack — Gesture-Controlled Cup Handover

A robot arm that picks up a cup, hands it to a person, and lets go: all
triggered by hand gestures. Built on top of the Viam
[Vision-Guided Pick-and-Place workshop](pick-and-place/README.md), on
a uFactory xArm6 with a wrist-mounted Intel RealSense depth camera.

## Demo

[![Watch the demo](https://img.youtube.com/vi/Q8n1Ia-qGjA/maxresdefault.jpg)](https://youtube.com/shorts/Q8n1Ia-qGjA)

*(click the thumbnail to watch on YouTube)*

## What it does

1. **Wait for a pick gesture** — the arm sits at a home pose watching the
   camera until it sees a right-hand thumbs up.
2. **Detect the cup** — moves to a fixed cup-view pose, segments the cup from
   the scene, and reconstructs its point cloud in the world frame.
3. **Estimate geometry** — fits a circle to the cup's rim (falling back to a
   silhouette estimate if the rim isn't fully visible) to find its axis,
   radius, and height.
4. **Grasp** — plans a top-down approach, descends in a straight line to just
   below the rim, and closes the gripper.
5. **Hand over** — lifts the cup, moves to an end pose, and waits for a
   release gesture before opening the gripper. If nobody takes the cup in
   time, it's placed back where it was picked up.

All of this lives in [`pick-and-place/scripts/cup-grasp.py`](pick-and-place/scripts/cup-grasp.py).

## Repo layout

```
.
├── README.md                        # this file
└── pick-and-place/                  # workshop scaffold this project builds on
    ├── config/                      # reference machine config + safety-wall obstacles
    ├── scripts/
    │   ├── cup-grasp.py             # this project: gesture-triggered cup pick + handover
    │   ├── reference-solution.py    # workshop reference: block pick-and-sort
    │   ├── starter-script.py        # workshop starting point with TODOs
    │   └── module-reference.py      # notes on the arm-position-saver / obstacle modules
    └── setup/
        └── frame-calibration-worksheet.md
```

See [`pick-and-place/README.md`](pick-and-place/README.md) for the full
workshop background, hardware list, and machine setup steps.

## Running it

```sh
cd pick-and-place/scripts
uv sync
export VIAM_MACHINE_ADDRESS=<your machine address>
export VIAM_API_KEY=<your api key>
export VIAM_API_KEY_ID=<your api key id>

uv run python cup-grasp.py                        # dry run — prints the plan, doesn't move
uv run python cup-grasp.py --execute               # approach, grasp, lift
uv run python cup-grasp.py --execute --handover     # ...then hand the cup to a person
```

Useful flags:

| Flag | What it does |
|---|---|
| `--execute` | Actually move the arm (default is a dry run) |
| `--handover` | After lifting, go to the end pose and wait for a release gesture |
| `--no-release` | Rehearse a handover without ever opening the gripper |
| `--stop-after approach\|grasp\|lift` | Stop early, useful for tuning |
| `--print-joints` | Print current joint angles (for setting `START_JOINTS_DEG`) |
| `--watch-gestures` | Print gesture-detector output for 60s without moving the arm |
| `--dump <csv>` | Save the detected cup's world-frame point cloud for inspection |

**Before running `--execute` near people or equipment**, check the printed
grasp plan against the scene and tune the constants at the top of
`cup-grasp.py` (grasp depth, standoff, gripper length offset, gesture
thresholds) for your hardware.

## Hardware

| Component | Role |
|---|---|
| uFactory xArm6 | 6-DOF arm |
| uFactory finger gripper | End effector |
| Intel RealSense D435 | RGB + depth camera, wrist-mounted |
| System76 Meerkat | Robot computer running `viam-server` |

## Credits

Built at a Viam hackathon on top of Viam's Vision-Guided Pick-and-Place
workshop.
