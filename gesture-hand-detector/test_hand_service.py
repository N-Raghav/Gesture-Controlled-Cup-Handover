import asyncio

from detect_hands_service import MediaPipeHandDetector, MODEL_PATH
from real_connect import real_connect
from viam.components.camera import Camera
import time


POLL_TIME = 0.25


async def main():
    async with await real_connect() as machine:
        camera = Camera.from_robot(machine, "cam")
        

        detector = MediaPipeHandDetector(
            MODEL_PATH,
            padding=20,
        )

        try:
            while True:
                images, _ = await camera.get_images(
                            filter_source_names=["color"]
                        )
                
                if not images:
                    raise RuntimeError("Camera returned no color image")
        
                detections = await detector.detect(images[0])

                print(f"Detected {len(detections)} hand(s)")
                for detection in detections:
                    print(
                        detection.class_name,
                        f"confidence={detection.confidence:.2f}",
                        f"box=({detection.x_min}, {detection.y_min})"
                        f"-({detection.x_max}, {detection.y_max})",
                    )

                time.sleep(POLL_TIME)
        finally:
            await detector.close()


asyncio.run(main())