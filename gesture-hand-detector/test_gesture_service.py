import asyncio
import sys
from pathlib import Path

from google.protobuf.json_format import ParseDict
from google.protobuf.struct_pb2 import Struct
from viam.components.camera import Camera
from viam.proto.app.robot import ComponentConfig

from real_connect import real_connect


MODULE_SOURCE = (
    Path(__file__).resolve().parent
    / "modules"
    / "gesture-detect"
    / "src"
)
sys.path.insert(0, str(MODULE_SOURCE))

from gesture_service import GestureDetectorService  # noqa: E402


CAMERA_NAME = "cam"
POLL_INTERVAL = 0.25


async def main():
    async with await real_connect() as machine:
        camera = Camera.from_robot(
            machine,
            CAMERA_NAME,
        )

        attributes = ParseDict(
            {
                "camera_name": CAMERA_NAME,
                "source_name": "color",
                "score_threshold": 0.5,
                "num_hands": 2,
            },
            Struct(),
        )

        config = ComponentConfig(
            name="gesture-detector",
            attributes=attributes,
        )

        dependencies = {
            Camera.get_resource_name(CAMERA_NAME): camera,
        }

        service = GestureDetectorService.new(
            config,
            dependencies,
        )

        try:
            properties = await service.get_properties()
            print(
                "Service ready:",
                f"classifications_supported="
                f"{properties.classifications_supported}",
            )
            print("Press Ctrl+C to stop")

            while True:
                classifications = (
                    await service.get_classifications_from_camera(
                        CAMERA_NAME,
                        count=2,
                    )
                )

                if classifications:
                    print(
                        [
                            (
                                result.class_name,
                                round(result.confidence, 2),
                            )
                            for result in classifications
                        ]
                    )
                else:
                    print("No recognized gesture")

                await asyncio.sleep(POLL_INTERVAL)

        finally:
            await service.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped")