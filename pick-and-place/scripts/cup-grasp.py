"""
Viam Pick-and-Place — Cup Grasp Generator
=========================================

Generates a grasp pose for a cup, from the side or from above:

    top   (default) The gripper points straight down. The fingers descend on
          either side of the cup, just below the rim, and close across it.
    side  The gripper comes in horizontally, the way a hand does, and wraps the
          body of the cup around its middle.

    1. Detect the cup and read its point cloud from the segmentation service.
    2. Move the points into the world frame and estimate the cup's axis, radius
       and height.
    3. Build three gripper poses in the world frame: approach (standoff), grasp
       and lift. The fingers close along a horizontal line tangent to the arm's
       reach, so the wrist doesn't have to swing sideways.

By default this only prints the plan. Pass --execute to move the arm.

    uv run python cup-grasp.py                        # dry run, top-down grasp
    uv run python cup-grasp.py --mode side            # dry run, side grasp
    uv run python cup-grasp.py --execute              # approach, grasp and lift
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

from viam.errors import ResourceNotFoundError
from viam.robot.client import RobotClient
from viam.components.gripper import Gripper
from viam.components.switch import Switch
from viam.services.motion import MotionClient
from viam.services.vision import VisionClient
from viam.proto.common import PoseInFrame, Pose
from viam.proto.service.motion import Constraints, LinearConstraint

# --- Tuning constants ---------------------------------------------------------
GRASP_HEIGHT_FRACTION = 0.55  # side grasp: where along the cup's height to hold it (0 base, 1 rim)
TOP_GRASP_DEPTH_MM = 40  # top-down grasp: how far below the rim the finger pads sit
RIM_CLEARANCE_MM = 15  # side grasp: keep the finger pads at least this far below the rim
TABLE_CLEARANCE_MM = 15  # ...and this far above the table
BAND_MM = 10  # half-height of the slice used to measure the cup's width
STANDOFF_MM = 100  # how far back from (or above) the cup the approach pose sits
LIFT_MM = 80  # how far to raise the cup after grabbing
GRIPPER_LENGTH_MM = (
    -60
)  # offset from the gripper's claw-geometry TCP to the real fingertip contact point
GRIPPER_MAX_OPEN_MM = 85  # widest opening of the finger gripper
FINGER_CLEARANCE_MM = 3  # spare room per side when the gripper is open
FINGER_ROLL_DEG = 0  # 0 if the fingers close along the gripper's y axis, 90 if along x
Z_OFFSET_MM = 0  # shift added to every estimated cup height, to correct a calibration error
LINE_TOLERANCE_MM = 5  # how far the straight moves may stray from the line
ORIENTATION_TOLERANCE_DEGS = 5  # ...and how far the gripper may tilt on them
SETTLE_S = 0.3  # finger gripper settle time after grab
MIN_CUP_HEIGHT_MM = 40  # reject flat detections such as the table
MIN_POINTS = 50

# --- Connection details -------------------------------------------------------
MACHINE_ADDRESS = os.environ.get("VIAM_MACHINE_ADDRESS", "armfarm8-main.310sld03v2.viam.cloud")
API_KEY = os.environ.get("VIAM_API_KEY", "5bh4v7iq5ngnw1as33asikvsak380wef")
API_KEY_ID = os.environ.get("VIAM_API_KEY_ID", "4c8355a9-92c2-4f69-87f1-c579606e23c2")

# --- Resource names (must match the CONFIGURE tab exactly) --------------------
GRIPPER_NAME = "gripper"
CAMERA_NAME = "cam"
VISION_NAME = "segmentation"
HOME_POSE = "home"
BASE_XY = (0.0, 0.0)  # arm base in the world frame


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * pct / 100)]


def parse_pcd(data: bytes) -> list[tuple[float, float, float]]:
    """Read x/y/z from a PCD blob (ascii or uncompressed binary)."""
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
        return [(float(r[ix]), float(r[iy]), float(r[iz])) for r in rows]
    if mode != "binary":
        raise ValueError(f"unsupported PCD encoding: {mode}")

    fmt = "<" + "".join(
        {"F": {4: "f", 8: "d"}, "U": {1: "B", 2: "H", 4: "I"}, "I": {1: "b", 2: "h", 4: "i"}}[t][s]
        for t, s in zip(types, sizes)
    )
    step = struct.calcsize(fmt)
    points = []
    for i in range(count):
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


def estimate_cup(points, cam_xyz, top_down: bool) -> dict:
    """Estimate the cup's vertical axis, radius and vertical extent (world, mm).

    The camera sees only part of the cup, but the silhouette across the view
    direction is the full diameter. The axis sits one radius behind the nearest
    surface. A side grasp measures the cup at the grasp height; a top-down
    grasp measures the rim, which is the widest part the fingers must clear.
    """
    zs = [p[2] for p in points]
    base_z, top_z = percentile(zs, 1), percentile(zs, 99)
    height = top_z - base_z
    if height < MIN_CUP_HEIGHT_MM:
        raise ValueError(
            f"cup looks only {height:.0f} mm tall ({len(points)} points, z {base_z:.0f} to "
            f"{top_z:.0f} mm). The camera probably sees only part of the cup; move the arm "
            "to a pose that views the whole cup and try again."
        )

    if top_down:
        grasp_z = top_z - TOP_GRASP_DEPTH_MM
        measure_z = top_z - BAND_MM
    else:
        grasp_z = min(base_z + GRASP_HEIGHT_FRACTION * height, top_z - RIM_CLEARANCE_MM)
        measure_z = grasp_z
    grasp_z = max(grasp_z, base_z + TABLE_CLEARANCE_MM)

    band = [p for p in points if abs(p[2] - measure_z) <= BAND_MM]
    if len(band) < MIN_POINTS:
        band = points

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


def plan_grasp(cup: dict, top_down: bool) -> dict[str, Pose]:
    """Build the approach, grasp and lift poses for the gripper frame (world, mm)."""
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

    if top_down:
        # Gripper z points straight down. Close the fingers along the tangent
        # (perpendicular to the reach). Straight down, the tool's y axis is
        # (sin th, cos th) and its x axis is (-cos th, sin th) in the world xy
        # plane, so solve th to put the closing axis on the tangent.
        theta = math.degrees(math.atan2(-dy, dx)) + FINGER_ROLL_DEG
        # theta and theta + 180 grasp identically; keep the one nearest 0 so the
        # wrist stays near its neutral yaw (for a cup ahead of the arm, about -7 deg).
        theta = (theta + 90) % 180 - 90

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

    # Side grasp: approach radially so the fingers straddle the cup tangentially
    # and the reach is a straight line out from the shoulder.
    def pose(back_mm: float, z: float) -> Pose:
        # back_mm is how far behind the fingertip contact point the frame sits.
        return Pose(
            x=cup["x"] - dx * back_mm,
            y=cup["y"] - dy * back_mm,
            z=z,
            o_x=dx,
            o_y=dy,
            o_z=0,
            theta=FINGER_ROLL_DEG,
        )

    return {
        "approach": pose(stop_short + STANDOFF_MM, cup["grasp_z"]),
        "grasp": pose(stop_short, cup["grasp_z"]),
        "lift": pose(stop_short, cup["grasp_z"] + LIFT_MM),
    }


async def detect_cup(machine: RobotClient, vision: VisionClient):
    """Return (points_world, camera_origin_world) for the object that looks like the cup.

    The segmentation service can return a large flat cluster (the table) next
    to the real cup, so skip anything too flat to be a cup.
    """
    objects = await vision.get_object_point_clouds(CAMERA_NAME)
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
        if height >= MIN_CUP_HEIGHT_MM:
            print(f"Detected: {label(obj)} ({len(points)} points, {height:.0f} mm tall)")
            return points, cam_origin
        seen.append(f"{label(obj)!r}: {len(points)} points, {height:.0f} mm tall")

    raise RuntimeError(
        f"None of the {len(seen)} detections is at least {MIN_CUP_HEIGHT_MM} mm tall: "
        + "; ".join(seen)
        + ". Move the arm so the camera sees the whole cup."
    )


async def execute(machine: RobotClient, plan: dict[str, Pose], stop_after: str) -> None:
    """Drive the plan with the builtin motion service; every move is planned."""
    gripper = Gripper.from_robot(machine, GRIPPER_NAME)
    motion = MotionClient.from_robot(machine, "builtin")

    async def move_to(pose: Pose, straight: bool = False) -> None:
        # The final descent and the lift must run straight so the fingers
        # don't sweep through the cup; the approach can take any collision-free path.
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
            component_name=GRIPPER_NAME,
            destination=PoseInFrame(reference_frame="world", pose=pose),
            constraints=constraints,
        )
        if not ok:
            raise RuntimeError(f"motion service could not reach {pose}")

    await gripper.open()
    await move_to(plan["approach"])
    if stop_after == "approach":
        return
    await move_to(plan["grasp"], straight=True)
    if stop_after == "grasp":
        return
    grabbed = await gripper.grab()
    await asyncio.sleep(SETTLE_S)
    if not grabbed:
        print("Gripper closed on nothing; check the grasp height and cup width")
    await move_to(plan["lift"], straight=True)


async def main(run: bool, top_down: bool, stop_after: str, dump: str | None) -> None:
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
        # Observe from home so the wrist-mounted camera sees the cup. Not every
        # machine has a saved home pose; without one, leave the arm where it is.
        try:
            await Switch.from_robot(machine, HOME_POSE).set_position(2)
        except ResourceNotFoundError:
            print(f"No '{HOME_POSE}' switch; using the arm where it is. The cup must be in view.")
        vision = VisionClient.from_robot(machine, VISION_NAME)

        points, cam_origin = await detect_cup(machine, vision)
        if dump:
            with open(dump, "w") as f:
                f.write(f"# camera origin (world, mm): {cam_origin[0]:.1f} {cam_origin[1]:.1f} {cam_origin[2]:.1f}\n")
                f.write("x,y,z\n")
                f.writelines(f"{x:.1f},{y:.1f},{z:.1f}\n" for x, y, z in points)
            print(f"Saved {len(points)} points to {dump}")
        cup = estimate_cup(points, cam_origin, top_down)
        plan = plan_grasp(cup, top_down)

        print(
            f"Cup axis ({cup['x']:.0f}, {cup['y']:.0f}) mm, "
            f"{'rim ' if top_down else ''}diameter {2 * cup['radius']:.0f} mm, "
            f"z {cup['base_z']:.0f} to {cup['top_z']:.0f} mm"
        )
        for name, pose in plan.items():
            print(
                f"{name:>8}: x={pose.x:.1f} y={pose.y:.1f} z={pose.z:.1f} "
                f"ov=({pose.o_x:.3f}, {pose.o_y:.3f}, {pose.o_z:.3f}) th={pose.theta:.0f}"
            )
        axes = tool_axes(plan["grasp"])
        print(f"Gripper axes in world at grasp: {axes}")

        if run:
            await execute(machine, plan, stop_after)
        else:
            print("Dry run; pass --execute to move the arm")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--mode", choices=("top", "side"), default="top")
    parser.add_argument("--execute", action="store_true", help="move the arm")
    parser.add_argument("--dump", metavar="CSV", help="save the cup's world-frame points here")
    parser.add_argument(
        "--stop-after",
        choices=("approach", "grasp", "lift"),
        default="lift",
        help="with --execute, stop after this step (the gripper stays open through 'grasp')",
    )
    args = parser.parse_args()
    asyncio.run(main(args.execute, args.mode == "top", args.stop_after, args.dump))
