import asyncio

from real_connect import real_connect
from viam.components.camera import Camera
from viam.proto.common import PoseInFrame
from viam.services.vision import VisionClient
import time


CAMERA_NAME = "cam"
SEGMENTER_NAME = "hand-segm"
DESTINATION_FRAME = "world"


async def main() -> None:
    async with await real_connect() as machine:
        camera = Camera.from_robot(machine, CAMERA_NAME)
        camera_properties = await camera.get_properties(
            timeout=10,
        )

        if not camera_properties.supports_pcd:
            raise RuntimeError(
                f"Camera {CAMERA_NAME!r} does not support point clouds"
            )

        segmenter = VisionClient.from_robot(
            machine,
            SEGMENTER_NAME,
        )

        properties = await segmenter.get_properties(timeout=10)

        if not properties.object_point_clouds_supported:
            raise RuntimeError(
                f"{SEGMENTER_NAME!r} is not a 3D segmenter"
            )

        try: 
            while True:

                objects = await segmenter.get_object_point_clouds(
                    CAMERA_NAME,
                    timeout=15,
                )

                print(f"Found {len(objects)} hand segment(s)")

                for index, point_cloud_object in enumerate(objects):
                    geometries = (
                        point_cloud_object.geometries.geometries
                    )

                    if not geometries:
                        print(f"Object {index} has no geometry")
                        continue

                    geometry = geometries[0]

                    source_frame = (
                        point_cloud_object.geometries.reference_frame
                        or CAMERA_NAME
                    )

                    camera_center = PoseInFrame(
                        reference_frame=source_frame,
                        pose=geometry.center,
                    )

                    world_center = await machine.transform_pose(
                        camera_center,
                        DESTINATION_FRAME,
                    )

                    camera_pose = camera_center.pose
                    world_pose = world_center.pose
                    label = geometry.label or f"hand_{index}"

                    print(
                        f"{label}: "
                        f"camera=({camera_pose.x:.1f}, "
                        f"{camera_pose.y:.1f}, "
                        f"{camera_pose.z:.1f}) mm, "
                        f"world=({world_pose.x:.1f}, "
                        f"{world_pose.y:.1f}, "
                        f"{world_pose.z:.1f}) mm"
                    )

                await asyncio.sleep(0.5)
        except KeyboardInterrupt:
            print('Stop flow')

if __name__ == "__main__":
    asyncio.run(main())