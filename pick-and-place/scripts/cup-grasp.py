"""
Viam Pick-and-Place — Cup Grasp Generator
=========================================

Generates a top-down grasp pose for a cup: the gripper points straight down,
and the fingers descend on either side of the cup, just below the rim, and
close across it.

    1. Detect the cup and read its point cloud from the segmentation service.
    2. Move the points into the world frame and estimate the cup's axis, radius
       and height.
    3. Build three gripper poses in the world frame: approach (standoff), grasp
       and lift. The wrist keeps the yaw it has at the cup-view pose.

By default this only prints the plan. Pass --execute to move the arm.

With --execute the arm first moves to the start pose and waits for the "pick"
gesture (pick_gesture_seen, a right-hand thumbs up), then goes to the cup-view pose
and runs the grasp.

With --handover, after lifting the cup the arm goes to the end pose and waits
for the "release" gesture (release_gesture_seen, either hand in view for now), then
opens the gripper. If nobody shows it in time, the cup is put back where it
was and the arm finishes at the end pose.

    uv run python cup-grasp.py                        # dry run
    uv run python cup-grasp.py --execute              # approach, grasp and lift
    uv run python cup-grasp.py --execute --handover   # ...then hand the cup over
    uv run python cup-grasp.py --execute --handover --no-release   # rehearse: never opens
    uv run python cup-grasp.py --execute --stop-after approach   # just the approach

Connection details come from the environment (Connect tab -> Python SDK):
VIAM_MACHINE_ADDRESS, VIAM_API_KEY, VIAM_API_KEY_ID.

NOTE: Like the reference solution, this must be validated against YOUR hardware.
Check the printed poses against the 3D scene before using --execute, and tune
the constants below. Handled mugs are treated as a plain cylinder, so the
fingers may land on the handle.
"""

import argparse
import asyncio
import math
import os
import struct
import time

from viam.robot.client import RobotClient
from viam.components.arm import Arm
from viam.components.gripper import Gripper
from viam.components.switch import Switch
from viam.services.motion import MotionClient
from viam.services.vision import VisionClient
from viam.proto.common import PoseInFrame, Pose
from viam.proto.component.arm import JointPositions
from viam.proto.service.motion import Constraints, LinearConstraint

# --- Tuning constants ---------------------------------------------------------
TOP_GRASP_DEPTH_MM = 40  # top-down grasp: how far below the rim the finger pads sit
TABLE_CLEARANCE_MM = 15  # ...and this far above the table
RIM_BAND_MM = 16  # thickness of the slice below the rim used to find the cup's centre and width
MAX_CUP_DEPTH_MM = 100  # anything this far behind the cup's nearest surface (a wall, the table) is not the cup
MIN_RIM_ARC_DEG = 180  # a circle fit needs the rim visible around at least this much of its circumference
STANDOFF_MM = 100  # how far above the cup the approach pose sits
LIFT_MM = 80  # how far to raise the cup after grabbing
GRIPPER_LENGTH_MM = (
    -60
)  # offset from the gripper's claw-geometry TCP to the real fingertip contact point
GRIPPER_MAX_OPEN_MM = 85  # widest opening of the finger gripper
FINGER_CLEARANCE_MM = 3  # spare room per side when the gripper is open
FINGER_ROLL_DEG = 0  # 0 if the fingers close along the gripper's y axis, 90 if along x
KEEP_WRIST_YAW = True  # top-down: keep the fingers closing the way they do at the cup-view pose, instead of turning the wrist to the tangent
Z_OFFSET_MM = 0  # shift added to every estimated cup height, to correct a calibration error
LINE_TOLERANCE_MM = 5  # how far the straight moves may stray from the line
ORIENTATION_TOLERANCE_DEGS = 5  # ...and how far the gripper may tilt on them
SETTLE_S = 0.3  # finger gripper settle time after grab
CAMERA_SETTLE_S = 0.2  # let the camera catch up after the arm stops
MIN_CUP_HEIGHT_MM = 40  # reject flat detections such as the table
MIN_POINTS = 50
MAX_POINTS = 4000  # a cup cloud has ~25k points; keep every n-th so the estimate stays fast

# --- Handover tuning ----------------------------------------------------------
HAND_MIN_CONFIDENCE = 0.3  # the detector is less sure of a hand close to the lens
GESTURE_MIN_CONFIDENCE = 0.5  # the gesture detector's own threshold is 0.5 too
HAND_LOST_GRACE_S = 0.7  # a hand this close to the camera flickers; tolerate gaps this long
HAND_DWELL_S = 0.3  # a hand must stay in view this long before the gripper opens
RIGHT_HAND_LABEL = "right"  # substring of the hand detector's class name for a right hand
LEFT_HAND_LABEL = "left"  # ...and for a left hand
RELEASE_TIMEOUT_S = 30
NO_HAND_TIMEOUT_S = 3  # with no hand in view this long at the end pose, the cup is put back
RELEASE_SPEED = 300  # gripper speed while letting go (1-5000; lower is slower)
NORMAL_SPEED = 1500  # ...and restored afterwards (the finger gripper's firmware default)
RELEASE_OPEN_TIMEOUT_S = 15  # stop waiting for the slow opening after this long

# --- Start pose ---------------------------------------------------------------
# With --execute the arm first goes to the start switch and waits for the "pick"
# gesture, then moves to the cup-view pose and carries on. After the grasp it
# goes to the end switch to wait for the "release" gesture. The start and end
# poses are saved on the machine (arm-position-saver switches); change them there.
START_SWITCH = "homepose"
END_SWITCH = "goalpose"  # where the arm takes the cup and releases it
# Where the arm looks at the cup from before grasping. It is the pose of the
# arm's own end (like the switches), not of the gripper. World frame, mm / degrees.
CUP_VIEW_POSE = Pose(
    x=187.21903348433494,
    y=36.11541758394837,
    z=386.1985804878573,
    o_x=0.07951343407776505,
    o_y=0.0964422637483874,
    o_z=-0.99215749937409,
    theta=50.40256760042287,
)
# Joint angles (degrees, joints 1-6) for the start pose, from --print-joints.
# Set these to reach the start pose by a plain joint move, which never picks a
# different wrist solution. None: use the start switch instead.
START_JOINTS_DEG: list[float] | None = None

# --- Connection details -------------------------------------------------------
MACHINE_ADDRESS = os.environ.get("VIAM_MACHINE_ADDRESS", "mybeautifularm")
API_KEY = os.environ.get("VIAM_API_KEY", "mybeautifulkey")
API_KEY_ID = os.environ.get("VIAM_API_KEY_ID", "mybeautifulkeyid")

# --- Resource names (must match the CONFIGURE tab exactly) --------------------
ARM_NAME = "arm"
GRIPPER_NAME = "gripper"
CAMERA_NAME = "cam"
VISION_NAME = "segmentation-cup"  # cup detections lifted to 3D
HAND_DETECTOR_NAME = "hand-detect"  # 2D hand boxes
GESTURE_DETECTOR_NAME = "gesture-detector"  # hand gestures
MOTION_NAME = "builtin"
BASE_XY = (0.0, 0.0)  # arm base in the world frame


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * pct / 100)]


_last_lap = time.monotonic()


def lap(what: str) -> None:
    """Print how long the step that just finished took."""
    global _last_lap
    now = time.monotonic()
    print(f"  [{now - _last_lap:.1f}s] {what}")
    _last_lap = now


def parse_pcd(data: bytes) -> list[tuple[float, float, float]]:
    """Read x/y/z from a PCD blob (ascii or uncompressed binary), keeping at most about MAX_POINTS."""
    header_end = data.index(b"DATA")
    line_end = data.index(b"\n", header_end)
    header = {
        line.split()[0]: line.split()[1:]
        for line in data[:header_end].decode().splitlines()
        if line.strip()
    }
    mode = data[header_end:line_end].split()[1].decode()
    body = data[line_end + 1 :]

    fields = header["FIELDS"]
    sizes = [int(s) for s in header["SIZE"]]
    types = header["TYPE"]
    count = int(header["POINTS"][0])
    ix, iy, iz = (fields.index(a) for a in "xyz")

    if mode == "ascii":
        rows = [line.split() for line in body.decode().splitlines() if line.strip()]
        return [(float(r[ix]), float(r[iy]), float(r[iz])) for r in rows[:: max(1, len(rows) // MAX_POINTS)]]
    if mode != "binary":
        raise ValueError(f"unsupported PCD encoding: {mode}")

    fmt = "<" + "".join(
        {"F": {4: "f", 8: "d"}, "U": {1: "B", 2: "H", 4: "I"}, "I": {1: "b", 2: "h", 4: "i"}}[t][s]
        for t, s in zip(types, sizes)
    )
    step = struct.calcsize(fmt)
    points = []
    for i in range(0, count, max(1, count // MAX_POINTS)):
        row = struct.unpack_from(fmt, body, i * step)
        points.append((row[ix], row[iy], row[iz]))
    return points


def rot_zyz(lon: float, lat: float, theta: float) -> list[list[float]]:
    """Rz(lon) @ Ry(lat) @ Rz(theta): the rotation an orientation vector encodes."""

    def rz(a):
        return [[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]]

    def ry(a):
        return [[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]]

    def mul(a, b):
        return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]

    return mul(mul(rz(lon), ry(lat)), rz(theta))


def tool_axes(pose: Pose) -> dict[str, tuple[float, ...]]:
    """The gripper's x/y/z axes in the world frame, for sanity-checking a pose."""
    lat = math.acos(max(-1.0, min(1.0, pose.o_z)))
    lon = math.atan2(pose.o_y, pose.o_x) if abs(pose.o_z) < 1 - 1e-8 else 0.0
    rot = rot_zyz(lon, lat, math.radians(pose.theta))
    return {
        name: tuple(round(rot[row][col], 3) for row in range(3))
        for col, name in enumerate("xyz")
    }


async def cloud_in_world(machine: RobotClient, point_cloud: bytes, center_mm: Pose):
    """Return the object's points and the camera origin, both in the world frame.

    The frame system does the transform: the camera's origin and axes are
    transformed into world, so no orientation math is done here.
    """
    points = parse_pcd(point_cloud)
    if len(points) < MIN_POINTS:
        raise ValueError(f"only {len(points)} points in the cup's cloud")

    # PCD files hold meters while poses are in mm; match the cloud to the pose.
    cx, cy, cz = (sum(p[i] for p in points) / len(points) for i in range(3))
    cloud_norm = math.sqrt(cx * cx + cy * cy + cz * cz)
    center_norm = math.sqrt(center_mm.x**2 + center_mm.y**2 + center_mm.z**2)
    if cloud_norm > 0 and center_norm > 20 * cloud_norm:
        points = [(x * 1000, y * 1000, z * 1000) for x, y, z in points]

    async def to_world(x, y, z):
        result = await machine.transform_pose(
            PoseInFrame(reference_frame=CAMERA_NAME, pose=Pose(x=x, y=y, z=z)), "world"
        )
        return (result.pose.x, result.pose.y, result.pose.z)

    origin = await to_world(0, 0, 0)
    basis = []
    for axis in ((1000, 0, 0), (0, 1000, 0), (0, 0, 1000)):
        moved = await to_world(*axis)
        basis.append(tuple((moved[i] - origin[i]) / 1000 for i in range(3)))

    world = [
        tuple(origin[i] + sum(basis[a][i] * p[a] for a in range(3)) for i in range(3))
        for p in points
    ]
    return world, origin


def fit_circle(xy: list[tuple[float, float]]) -> tuple[float, float, float] | None:
    """Least-squares circle through the points (x, y, radius), ignoring outliers, or None if it is not a rim.

    The points are the cup's rim seen from above. Stray points (a handle, the
    table) are dropped by refitting on the 80% that lie nearest the circle.
    """

    def kasa(pts):
        n = len(pts)
        sx = sum(x for x, _ in pts)
        sy = sum(y for _, y in pts)
        sxx = sum(x * x for x, _ in pts)
        syy = sum(y * y for _, y in pts)
        sxy = sum(x * y for x, y in pts)
        r2 = [x * x + y * y for x, y in pts]
        m = [
            [sxx, sxy, sx, sum(x * r for (x, _), r in zip(pts, r2))],
            [sxy, syy, sy, sum(y * r for (_, y), r in zip(pts, r2))],
            [sx, sy, n, sum(r2)],
        ]
        for i in range(3):  # Gauss-Jordan
            pivot = max(range(i, 3), key=lambda r: abs(m[r][i]))
            m[i], m[pivot] = m[pivot], m[i]
            if abs(m[i][i]) < 1e-9:
                return None
            for r in range(3):
                if r != i:
                    f = m[r][i] / m[i][i]
                    m[r] = [m[r][k] - f * m[i][k] for k in range(4)]
        a, b, c = (m[i][3] / m[i][i] for i in range(3))
        cx, cy = a / 2, b / 2
        return cx, cy, math.sqrt(max(c + cx * cx + cy * cy, 0.0))

    if len(xy) < MIN_POINTS:
        return None
    fit = kasa(xy)
    for _ in range(4):
        if fit is None:
            return None
        cx, cy, r = fit
        dist = [abs(math.hypot(x - cx, y - cy) - r) for x, y in xy]
        limit = max(3.0, percentile(dist, 80))
        fit = kasa([p for p, d in zip(xy, dist) if d <= limit])
    if fit is None:
        return None

    cx, cy, r = fit
    angles = sorted(math.atan2(y - cy, x - cx) for x, y in xy)
    gaps = [b - a for a, b in zip(angles, angles[1:])] + [angles[0] + 2 * math.pi - angles[-1]]
    arc = 360 - math.degrees(max(gaps))
    if arc < MIN_RIM_ARC_DEG or not 10 < r < GRIPPER_MAX_OPEN_MM:
        return None
    return cx, cy, r


def drop_background(points, cam_xyz):
    """Keep the points within MAX_CUP_DEPTH_MM of the nearest ones, seen from above.

    The segmentation can merge the wall or the table behind the cup into its
    cluster, which stretches the cup's apparent size.
    """
    ranges = [math.hypot(p[0] - cam_xyz[0], p[1] - cam_xyz[1]) for p in points]
    limit = percentile(ranges, 3) + MAX_CUP_DEPTH_MM
    near = [p for p, r in zip(points, ranges) if r <= limit]
    return near if len(near) >= MIN_POINTS else points


def estimate_cup(points, cam_xyz) -> dict:
    """Estimate the cup's vertical axis, radius and vertical extent (world, mm).

    The rim, seen from above, is fitted with a circle for the axis and radius.
    If too little of it is visible for that, fall back on the silhouette: it
    is the full diameter across the view direction, and the axis sits one
    radius behind the nearest surface.
    """
    points = drop_background(points, cam_xyz)
    zs = [p[2] for p in points]
    base_z, top_z = percentile(zs, 1), percentile(zs, 99)
    height = top_z - base_z
    if height < MIN_CUP_HEIGHT_MM:
        raise ValueError(
            f"cup looks only {height:.0f} mm tall ({len(points)} points, z {base_z:.0f} to "
            f"{top_z:.0f} mm). The camera probably sees only part of the cup; move the arm "
            "to a pose that views the whole cup and try again."
        )

    grasp_z = max(top_z - TOP_GRASP_DEPTH_MM, base_z + TABLE_CLEARANCE_MM)

    band = [p for p in points if p[2] >= top_z - RIM_BAND_MM]
    if len(band) < MIN_POINTS:
        band = points

    circle = fit_circle([(p[0], p[1]) for p in band])
    if circle:
        print(f"Rim circle fit: centre ({circle[0]:.0f}, {circle[1]:.0f}), diameter {2 * circle[2]:.0f} mm")
        return {
            "x": circle[0],
            "y": circle[1],
            "radius": circle[2],
            "base_z": base_z + Z_OFFSET_MM,
            "top_z": top_z + Z_OFFSET_MM,
            "grasp_z": grasp_z + Z_OFFSET_MM,
        }
    print("Rim not visible all around; estimating the axis from the silhouette")

    # Horizontal view direction v from the camera to the cup, and its perpendicular p.
    mx = sum(p[0] for p in points) / len(points) - cam_xyz[0]
    my = sum(p[1] for p in points) / len(points) - cam_xyz[1]
    norm = math.hypot(mx, my)
    # Directly overhead the direction is arbitrary; a full rim ring works with any.
    vx, vy = (mx / norm, my / norm) if norm > 1 else (1.0, 0.0)
    px, py = -vy, vx

    s = [(p[0] - cam_xyz[0]) * vx + (p[1] - cam_xyz[1]) * vy for p in band]
    t = [(p[0] - cam_xyz[0]) * px + (p[1] - cam_xyz[1]) * py for p in band]
    t_lo, t_hi = percentile(t, 1), percentile(t, 99)
    radius = (t_hi - t_lo) / 2
    s_axis = percentile(s, 3) + radius
    t_axis = (t_lo + t_hi) / 2

    return {
        "x": cam_xyz[0] + vx * s_axis + px * t_axis,
        "y": cam_xyz[1] + vy * s_axis + py * t_axis,
        "radius": radius,
        "base_z": base_z + Z_OFFSET_MM,
        "top_z": top_z + Z_OFFSET_MM,
        "grasp_z": grasp_z + Z_OFFSET_MM,
    }


def plan_grasp(cup: dict, yaw_deg: float | None = None) -> dict[str, Pose]:
    """Build the approach, grasp and lift poses for the gripper frame (world, mm).

    yaw_deg, if given, is the heading of the gripper's y axis (see gripper_yaw_deg);
    the fingers then keep closing along it instead of turning to the tangent.
    """
    diameter = 2 * cup["radius"]
    if diameter + 2 * FINGER_CLEARANCE_MM > GRIPPER_MAX_OPEN_MM:
        raise ValueError(
            f"cup is ~{diameter:.0f} mm wide but the gripper only opens "
            f"{GRIPPER_MAX_OPEN_MM} mm (with {FINGER_CLEARANCE_MM} mm clearance per side). "
            "If the cup is really narrower, the detection probably includes the gripper "
            "or the table: start from a pose where the camera sees the cup alone."
        )

    # Radial direction from the arm base to the cup.
    rx, ry = cup["x"] - BASE_XY[0], cup["y"] - BASE_XY[1]
    reach = math.hypot(rx, ry)
    dx, dy = rx / reach, ry / reach
    stop_short = abs(GRIPPER_LENGTH_MM)

    # Gripper z points straight down. Close the fingers along the tangent
    # (perpendicular to the reach). Straight down, the tool's y axis is
    # (sin th, cos th) and its x axis is (-cos th, sin th) in the world xy
    # plane, so solve th to put the closing axis on the tangent.
    theta = math.degrees(math.atan2(-dy, dx)) + FINGER_ROLL_DEG
    # theta and theta + 180 grasp identically; keep the one nearest 0 so the
    # wrist stays near its neutral yaw (for a cup ahead of the arm, about -7 deg).
    theta = (theta + 90) % 180 - 90
    if yaw_deg is not None:
        # A cup is round, so any finger direction works. Straight down, the
        # tool's y axis points at heading 90 - theta; reuse the current heading.
        theta = (90 - yaw_deg + 180) % 360 - 180

    def pose(above_mm: float) -> Pose:
        # The frame sits stop_short above the fingertip contact point.
        return Pose(
            x=cup["x"],
            y=cup["y"],
            z=cup["grasp_z"] + stop_short + above_mm,
            o_x=0,
            o_y=0,
            o_z=-1,
            theta=theta,
        )

    return {
        "approach": pose(STANDOFF_MM),
        "grasp": pose(0),
        "lift": pose(LIFT_MM),
    }


async def gripper_yaw_deg(machine: RobotClient) -> float:
    """The heading of the gripper's y axis (the way its fingers close) in the world.

    Whatever way the gripper points, this axis stays horizontal, so it says which
    way to turn the wrist for a top-down grasp. The x axis would not: it points
    straight up or down when the gripper is horizontal.
    """
    pose = (
        await machine.transform_pose(PoseInFrame(reference_frame=GRIPPER_NAME, pose=Pose()), "world")
    ).pose
    y_axis = tool_axes(pose)["y"]
    return math.degrees(math.atan2(y_axis[1], y_axis[0]))


async def detect_cup(machine: RobotClient, vision: VisionClient):
    """Return (points_world, camera_origin_world) for the object that looks like the cup.

    The segmentation service can return a large flat cluster (the table) next
    to the real cup, so skip anything too flat to be a cup.
    """
    objects = await vision.get_object_point_clouds(CAMERA_NAME)
    lap(f"vision service returned {len(objects)} object(s)")
    if not objects:
        raise RuntimeError("No objects detected")

    def label(o) -> str:
        return o.geometries.geometries[0].label or ""

    cups = [o for o in objects if "cup" in label(o).lower()]
    seen = []
    for obj in sorted(cups or objects, key=lambda o: len(o.point_cloud), reverse=True):
        points, cam_origin = await cloud_in_world(
            machine, obj.point_cloud, obj.geometries.geometries[0].center
        )
        zs = [p[2] for p in points]
        height = percentile(zs, 99) - percentile(zs, 1)
        lap(f"moved {label(obj)!r} cloud into the world frame")
        if height >= MIN_CUP_HEIGHT_MM:
            print(f"Detected: {label(obj)} ({len(points)} points, {height:.0f} mm tall)")
            return points, cam_origin
        seen.append(f"{label(obj)!r}: {len(points)} points, {height:.0f} mm tall")

    raise RuntimeError(
        f"None of the {len(seen)} detections is at least {MIN_CUP_HEIGHT_MM} mm tall: "
        + "; ".join(seen)
        + ". Move the arm so the camera sees the whole cup."
    )


async def move_gripper(
    motion: MotionClient, pose: Pose, straight: bool = False, component: str = GRIPPER_NAME
) -> None:
    """Plan and run a move of the gripper (or another component) in the world frame with the builtin motion service."""
    # The final descent and the lift must run straight so the fingers don't
    # sweep through the cup; other moves can take any collision-free path.
    constraints = (
        Constraints(
            linear_constraint=[
                LinearConstraint(
                    line_tolerance_mm=LINE_TOLERANCE_MM,
                    orientation_tolerance_degs=ORIENTATION_TOLERANCE_DEGS,
                )
            ]
        )
        if straight
        else None
    )
    ok = await motion.move(
        component_name=component,
        destination=PoseInFrame(reference_frame="world", pose=pose),
        constraints=constraints,
    )
    lap(f"moved to x={pose.x:.0f} y={pose.y:.0f} z={pose.z:.0f} ({'straight' if straight else 'planned'})")
    if not ok:
        raise RuntimeError(f"motion service could not reach {pose}")


def is_hand(detection, label: str) -> bool:
    return label in (detection.class_name or "").lower()


async def hands_in_view(machine: RobotClient) -> list:
    detector = VisionClient.from_robot(machine, HAND_DETECTOR_NAME)
    return [
        d
        for d in await detector.get_detections_from_camera(CAMERA_NAME)
        if d.confidence > HAND_MIN_CONFIDENCE
    ]


async def gesture_labels(machine: RobotClient) -> list[str]:
    """The lowercase labels the gesture detector reports for the camera's view right now."""
    detector = VisionClient.from_robot(machine, GESTURE_DETECTOR_NAME)
    return [
        c.class_name.lower()
        for c in await detector.get_classifications_from_camera(CAMERA_NAME, 5)
        if c.confidence > GESTURE_MIN_CONFIDENCE
    ]


async def pick_gesture_seen(machine: RobotClient, hands: list, gestures: list[str]) -> bool:
    """Return True on a poll where the user is showing the "pick the cup" gesture: a right-hand thumbs up.

    `hands` are the confident hand detections and `gestures` the gesture labels
    from the camera this poll. If a gesture label names its hand ("right", "left")
    that decides which hand made it. If not, the hand detector decides: a right
    hand in view and no left hand.
    """
    thumbs_up = [g for g in gestures if "thumb" in g and "up" in g]
    if not thumbs_up:
        return False
    named = [g for g in thumbs_up if RIGHT_HAND_LABEL in g or LEFT_HAND_LABEL in g]
    if named:
        return any(RIGHT_HAND_LABEL in g for g in named)
    return any(is_hand(d, RIGHT_HAND_LABEL) for d in hands) and not any(
        is_hand(d, LEFT_HAND_LABEL) for d in hands
    )


async def release_gesture_seen(machine: RobotClient, hands: list) -> bool:
    """Return True on a poll where the user is showing the "release" gesture.

    `hands` are the confident hand detections from the camera this poll.
    Placeholder until the real gesture check exists: a hand in view, either one.
    """
    return any(is_hand(d, LEFT_HAND_LABEL) or is_hand(d, RIGHT_HAND_LABEL) for d in hands)


async def wait_for_pick_gesture(machine: RobotClient) -> None:
    print("Waiting for a right-hand thumbs up...")
    while True:
        hands, gestures = await asyncio.gather(hands_in_view(machine), gesture_labels(machine))
        if hands or gestures:
            print(f"  hands in view: {[d.class_name for d in hands]}  gestures: {gestures}")
        if await pick_gesture_seen(machine, hands, gestures):
            return
        await asyncio.sleep(0.2)


async def go_to_start(machine: RobotClient) -> None:
    """Move the arm to the start pose: by joint angles if they are set, else with the start switch."""
    if START_JOINTS_DEG:
        await Arm.from_robot(machine, ARM_NAME).move_to_joint_positions(
            JointPositions(values=START_JOINTS_DEG)
        )
    else:
        await Switch.from_robot(machine, START_SWITCH).set_position(2)


def object_label(obj) -> str:
    return (obj.geometries.geometries[0].label or "").lower()


def box(d) -> tuple[float, float, float, float]:
    return (d.x_min, d.y_min, d.x_max, d.y_max)


async def hand_over(machine, gripper, release: bool) -> bool:
    """Take the held cup to the end pose; open when the release gesture is shown. False if nobody asked for it."""
    print("Moving to the end pose")
    await Switch.from_robot(machine, END_SWITCH).set_position(2)
    lap("moved to the end pose")
    await asyncio.sleep(CAMERA_SETTLE_S)

    since, last_seen, deadline = None, 0.0, time.monotonic() + RELEASE_TIMEOUT_S
    last_hand = time.monotonic()
    print("Waiting for the release gesture...")
    while time.monotonic() < deadline:
        hands = await hands_in_view(machine)
        gesture = await release_gesture_seen(machine, hands)
        now = time.monotonic()
        if hands:
            last_hand = now
        elif now - last_hand >= NO_HAND_TIMEOUT_S:
            print(f"No hand in view for {NO_HAND_TIMEOUT_S} s")
            return False
        if gesture:
            since, last_seen = since or now, now
        elif since and now - last_seen > HAND_LOST_GRACE_S:
            since = None
        held_s = now - since if since else 0.0
        print(f"{len(hands)} hand(s) in view, release gesture {gesture}  {held_s:.1f}/{HAND_DWELL_S:.1f}s")

        if held_s >= HAND_DWELL_S:
            if not release:
                print("Release gesture seen (--no-release, so the gripper stays closed)")
                since = None
            else:
                print("Release gesture seen: releasing")
                await release_cup(gripper)
                return True
        await asyncio.sleep(0.2)
    print("No release gesture")
    return False


async def release_cup(gripper: Gripper) -> None:
    """Open the gripper slowly so the cup is lowered out of the fingers, not dropped."""
    try:
        await gripper.do_command({"set_gripper_speed": RELEASE_SPEED})
    except Exception as e:  # never keep hold of the cup because the speed could not be set
        print(f"Could not slow the gripper ({e}); opening at its current speed")
    await gripper.open()
    deadline = time.monotonic() + RELEASE_OPEN_TIMEOUT_S
    while await gripper.is_moving() and time.monotonic() < deadline:
        await asyncio.sleep(0.1)
    await asyncio.sleep(SETTLE_S)
    lap("gripper opened")
    try:
        await gripper.do_command({"set_gripper_speed": NORMAL_SPEED})
    except Exception as e:
        print(f"Could not restore the gripper speed ({e})")


async def put_back(motion: MotionClient, gripper: Gripper, plan: dict[str, Pose]) -> None:
    """Return the cup to where it was picked up and let go."""
    await move_gripper(motion, plan["approach"])
    await move_gripper(motion, plan["grasp"], straight=True)
    await release_cup(gripper)
    await move_gripper(motion, plan["approach"], straight=True)


async def execute(machine: RobotClient, plan: dict[str, Pose], args) -> None:
    """Drive the plan with the builtin motion service; every move is planned."""
    gripper = Gripper.from_robot(machine, GRIPPER_NAME)
    motion = MotionClient.from_robot(machine, MOTION_NAME)

    # Open the gripper before the arm moves. Running it alongside the motion
    # service call dropped the connection and expired the session.
    await gripper.open()
    # A straight move stays in the arm's current wrist configuration. A free
    # plan may reach the same pose with the wrist turned half a revolution.
    try:
        await move_gripper(motion, plan["approach"], straight=True)
    except Exception as e:
        print(f"Could not approach in a straight line ({e}); planning a free path")
        await move_gripper(motion, plan["approach"])
    if args.stop_after == "approach":
        return
    await move_gripper(motion, plan["grasp"], straight=True)
    if args.stop_after == "grasp":
        return
    grabbed = await gripper.grab()
    await asyncio.sleep(SETTLE_S)
    lap("gripper closed")
    if not grabbed:
        print("Gripper closed on nothing; check the grasp height and cup width")
        return
    await move_gripper(motion, plan["lift"], straight=True)

    if args.handover and not await hand_over(machine, gripper, release=not args.no_release):
        print("No release gesture; putting the cup back")
        await put_back(motion, gripper, plan)
        print("Moving to the start pose")
        await go_to_start(machine)


async def watch_gestures(machine: RobotClient, seconds: float = 60) -> None:
    """Print what the gesture detector reports, to see the labels it uses. Does not move the arm."""
    detector = VisionClient.from_robot(machine, GESTURE_DETECTOR_NAME)
    print(f"Show gestures to the camera for {seconds:.0f} s...")
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        classifications = await detector.get_classifications_from_camera(CAMERA_NAME, 5)
        print(f"classifications {[(c.class_name, round(c.confidence, 2)) for c in classifications]}")
        await asyncio.sleep(0.5)


async def plan_from_camera(machine: RobotClient, args) -> dict[str, Pose]:
    """Look at the cup from the cup-view pose and plan the grasp for it."""
    # Observe from the cup-view pose so the wrist-mounted camera sees the whole cup.
    print("Moving to the cup-view pose")
    await move_gripper(MotionClient.from_robot(machine, MOTION_NAME), CUP_VIEW_POSE, component=ARM_NAME)
    lap("moved to the cup-view pose")
    await asyncio.sleep(CAMERA_SETTLE_S)
    vision = VisionClient.from_robot(machine, VISION_NAME)

    points, cam_origin = await detect_cup(machine, vision)
    if args.dump:
        with open(args.dump, "w") as f:
            f.write(f"# camera origin (world, mm): {cam_origin[0]:.1f} {cam_origin[1]:.1f} {cam_origin[2]:.1f}\n")
            f.write("x,y,z\n")
            f.writelines(f"{x:.1f},{y:.1f},{z:.1f}\n" for x, y, z in points)
        print(f"Saved {len(points)} points to {args.dump}")
    cup = estimate_cup(points, cam_origin)
    yaw = await gripper_yaw_deg(machine) if KEEP_WRIST_YAW else None
    plan = plan_grasp(cup, yaw)

    print(
        f"Cup axis ({cup['x']:.0f}, {cup['y']:.0f}) mm, "
        f"rim diameter {2 * cup['radius']:.0f} mm, "
        f"z {cup['base_z']:.0f} to {cup['top_z']:.0f} mm"
    )
    return plan


async def main(args) -> None:
    if not (MACHINE_ADDRESS and API_KEY and API_KEY_ID):
        raise SystemExit(
            "Set VIAM_MACHINE_ADDRESS, VIAM_API_KEY and VIAM_API_KEY_ID first."
        )

    async with await RobotClient.at_address(
        MACHINE_ADDRESS,
        # The SDK's background connection check gives up after 1 s. A busy machine
        # (planning a move) can miss that, and the reconnect attempt then crashes
        # the process, so turn the check off. A real disconnect still errors on the next call.
        options=RobotClient.Options.with_api_key(
            api_key=API_KEY,
            api_key_id=API_KEY_ID,
            check_connection_interval=0,
            attempt_reconnect_interval=0,
        ),
    ) as machine:
        if args.print_joints:
            joints = await Arm.from_robot(machine, ARM_NAME).get_joint_positions()
            print(f"START_JOINTS_DEG = {[round(v, 2) for v in joints.values]}")
            return

        if args.watch_gestures:
            await watch_gestures(machine)
            return

        if args.execute:
            print("Moving to the start pose")
            await go_to_start(machine)
            await wait_for_pick_gesture(machine)

        lap("start pose and pick gesture")
        plan = await plan_from_camera(machine, args)

        for name, pose in plan.items():
            print(
                f"{name:>8}: x={pose.x:.1f} y={pose.y:.1f} z={pose.z:.1f} "
                f"ov=({pose.o_x:.3f}, {pose.o_y:.3f}, {pose.o_z:.3f}) th={pose.theta:.0f}"
            )
        axes = tool_axes(plan["grasp"])
        print(f"Gripper axes in world at grasp: {axes}")

        if args.execute:
            await execute(machine, plan, args)
        else:
            print("Dry run; pass --execute to move the arm")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument(
        "--watch-gestures",
        action="store_true",
        help="print what the gesture detector reports for 60 s and exit (does not move)",
    )
    parser.add_argument("--execute", action="store_true", help="move the arm")
    parser.add_argument("--dump", metavar="CSV", help="save the cup's world-frame points here")
    parser.add_argument(
        "--stop-after",
        choices=("approach", "grasp", "lift"),
        default="lift",
        help="with --execute, stop after this step (the gripper stays open through 'grasp')",
    )
    parser.add_argument("--handover", action="store_true", help="after the lift, go to the end pose and release on the release gesture")
    parser.add_argument("--no-release", action="store_true", help="with --handover, never open the gripper")
    parser.add_argument(
        "--print-joints",
        action="store_true",
        help="print the arm's current joint angles as START_JOINTS_DEG and exit (does not move)",
    )
    asyncio.run(main(parser.parse_args()))
