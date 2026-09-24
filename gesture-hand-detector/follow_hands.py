import asyncio
import math
import statistics
from collections import deque

from real_connect import real_connect
from viam.components.arm import Arm
from viam.proto.common import Pose, PoseInFrame
from viam.proto.service.motion import Constraints, LinearConstraint
from viam.services.motion import MotionClient
from viam.services.vision import VisionClient


CAMERA_NAME = "cam"
SEGMENTER_NAME = "hand-segm"

# Replace these with the exact Viam resource names.
ARM_NAME = "arm"
FOLLOW_COMPONENT = "gripper"

WORLD_FRAME = "world"
TARGET_LABEL = "right_hand"

# Start in observation-only mode.
ENABLE_MOTION = False

# These are demonstration values, not guaranteed safety values.
STANDOFF_MM = 300.0
MAX_STEP_MM = 25.0
DEADBAND_MM = 30.0
SAMPLES_REQUIRED = 5
SAMPLE_PERIOD_SECONDS = 0.1

# Replace this with physically verified limits before enabling motion.
# Example:
# {
#     "x": (200, 700),
#     "y": (-300, 300),
#     "z": (300, 900),
# }
SAFE_BOUNDS_MM = None


def distance(a, b) -> float:
    return math.sqrt(
        (a[0] - b[0]) ** 2
        + (a[1] - b[1]) ** 2
        + (a[2] - b[2]) ** 2
    )


def median_point(points):
    return (
        statistics.median(point[0] for point in points),
        statistics.median(point[1] for point in points),
        statistics.median(point[2] for point in points),
    )


def inside_safe_workspace(point) -> bool:
    if SAFE_BOUNDS_MM is None:
        return False

    x, y, z = point

    return (
        SAFE_BOUNDS_MM["x"][0] <= x <= SAFE_BOUNDS_MM["x"][1]
        and SAFE_BOUNDS_MM["y"][0] <= y <= SAFE_BOUNDS_MM["y"][1]
        and SAFE_BOUNDS_MM["z"][0] <= z <= SAFE_BOUNDS_MM["z"][1]
    )


def calculate_follow_goal(hand, current):
    """Remain on the current side of the hand at STANDOFF_MM."""
    hand_x, hand_y, hand_z = hand
    current_x, current_y, current_z = current

    from_hand = (
        current_x - hand_x,
        current_y - hand_y,
        current_z - hand_z,
    )
    current_distance = math.sqrt(
        from_hand[0] ** 2
        + from_hand[1] ** 2
        + from_hand[2] ** 2
    )

    if current_distance < 1.0:
        raise RuntimeError(
            "The end effector is too close to the detected hand"
        )

    scale = STANDOFF_MM / current_distance

    return (
        hand_x + from_hand[0] * scale,
        hand_y + from_hand[1] * scale,
        hand_z + from_hand[2] * scale,
    )


def limit_step(current, goal):
    """Limit every commanded movement to MAX_STEP_MM."""
    remaining = distance(current, goal)

    if remaining <= DEADBAND_MM:
        return None

    ratio = min(1.0, MAX_STEP_MM / remaining)

    return (
        current[0] + (goal[0] - current[0]) * ratio,
        current[1] + (goal[1] - current[1]) * ratio,
        current[2] + (goal[2] - current[2]) * ratio,
    )


async def get_hand_in_world(machine, segmenter):
    objects = await segmenter.get_object_point_clouds(
        CAMERA_NAME,
        timeout=15,
    )

    for point_cloud_object in objects:
        source_frame = (
            point_cloud_object.geometries.reference_frame
            or CAMERA_NAME
        )

        for geometry in point_cloud_object.geometries.geometries:
            if geometry.label != TARGET_LABEL:
                continue

            camera_pose = PoseInFrame(
                reference_frame=source_frame,
                pose=geometry.center,
            )

            return await machine.transform_pose(
                camera_pose,
                WORLD_FRAME,
            )

    return None


async def main() -> None:
    if ENABLE_MOTION and SAFE_BOUNDS_MM is None:
        raise RuntimeError(
            "Set physically verified SAFE_BOUNDS_MM before enabling motion"
        )

    history = deque(maxlen=SAMPLES_REQUIRED)

    async with await real_connect() as machine:
        segmenter = VisionClient.from_robot(
            machine,
            SEGMENTER_NAME,
        )
        motion = MotionClient.from_robot(machine, "builtin")
        arm = Arm.from_robot(machine, ARM_NAME)

        constraints = Constraints(
            linear_constraint=[
                LinearConstraint(
                    line_tolerance_mm=10,
                    orientation_tolerance_degs=10,
                )
            ]
        )

        try:
            while True:
                hand_pose = await get_hand_in_world(
                    machine,
                    segmenter,
                )

                if hand_pose is None:
                    history.clear()
                    print(f"Waiting for {TARGET_LABEL}")
                    await asyncio.sleep(SAMPLE_PERIOD_SECONDS)
                    continue

                hand_point = (
                    hand_pose.pose.x,
                    hand_pose.pose.y,
                    hand_pose.pose.z,
                )
                history.append(hand_point)

                if len(history) < SAMPLES_REQUIRED:
                    print(
                        f"Stabilizing target "
                        f"{len(history)}/{SAMPLES_REQUIRED}"
                    )
                    continue

                filtered_hand = median_point(history)

                current_pose = await motion.get_pose(
                    FOLLOW_COMPONENT,
                    WORLD_FRAME,
                    timeout=10,
                )
                current_point = (
                    current_pose.pose.x,
                    current_pose.pose.y,
                    current_pose.pose.z,
                )

                follow_goal = calculate_follow_goal(
                    filtered_hand,
                    current_point,
                )
                next_point = limit_step(
                    current_point,
                    follow_goal,
                )

                if next_point is None:
                    print("Holding position inside deadband")
                    await asyncio.sleep(SAMPLE_PERIOD_SECONDS)
                    continue

                print(
                    f"hand={filtered_hand}, "
                    f"next end-effector target={next_point}"
                )

                if not ENABLE_MOTION:
                    print("DRY RUN: motion disabled")
                    await asyncio.sleep(SAMPLE_PERIOD_SECONDS)
                    continue

                if not inside_safe_workspace(next_point):
                    print("Rejected target outside safe workspace")
                    history.clear()
                    await asyncio.sleep(SAMPLE_PERIOD_SECONDS)
                    continue

                # Preserve the end effector's current orientation. Do not use
                # the point-cloud geometry orientation as palm orientation.
                destination = PoseInFrame(
                    reference_frame=WORLD_FRAME,
                    pose=Pose(
                        x=next_point[0],
                        y=next_point[1],
                        z=next_point[2],
                        o_x=current_pose.pose.o_x,
                        o_y=current_pose.pose.o_y,
                        o_z=current_pose.pose.o_z,
                        theta=current_pose.pose.theta,
                    ),
                )

                moved = await motion.move(
                    component_name=FOLLOW_COMPONENT,
                    destination=destination,
                    constraints=constraints,
                    extra={"collision_buffer_mm": 50},
                    timeout=30,
                )

                if not moved:
                    print("Motion planner did not execute the move")

                # Reacquire the hand after each small movement.
                history.clear()

        finally:
            if ENABLE_MOTION:
                await arm.stop()


if __name__ == "__main__":
    asyncio.run(main())