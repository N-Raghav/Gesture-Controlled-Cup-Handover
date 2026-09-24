import asyncio

import cv2
import numpy as np
from viam.components.camera import Camera

from real_connect import real_connect


async def main():
    async with await real_connect() as machine:
        cam = Camera.from_robot(machine, "cam")

        print("Streaming camera 'cam'. Press q or Esc to quit.")

        try:
            while True:
                images, _ = await cam.get_images()

                if not images:
                    print("Camera returned no images")
                    await asyncio.sleep(0.1)
                    continue

                encoded_image = np.frombuffer(
                    images[0].data,
                    dtype=np.uint8,
                )
                frame = cv2.imdecode(
                    encoded_image,
                    cv2.IMREAD_COLOR,
                )

                if frame is None:
                    print("Could not decode camera frame")
                    await asyncio.sleep(0.1)
                    continue

                cv2.imshow("Viam camera", frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
        finally:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    asyncio.run(main())