import asyncio
import time
from io import BytesIO

import mediapipe as mp
import numpy as np
from PIL import Image
from viam.components.camera import Camera
from pathlib import Path
from real_connect import real_connect
import cv2

CAMERA_NAME = "cam"
MODEL_PATH = str(
    Path(__file__).resolve().parent
    / "models"
    / "hand_landmarker.task"
)


def draw_hand_boxes(frame, result, padding=20):
    height, width = frame.shape[:2]

    for hand_index, landmarks in enumerate(result.hand_landmarks):
        xs = [int(point.x * width) for point in landmarks]
        ys = [int(point.y * height) for point in landmarks]

        x1 = max(0, min(xs) - padding)
        y1 = max(0, min(ys) - padding)
        x2 = min(width - 1, max(xs) + padding)
        y2 = min(height - 1, max(ys) + padding)

        label = f"Hand {hand_index + 1}"

        if result.handedness[hand_index]:
            handedness = result.handedness[hand_index][0]
            label = f"{handedness.category_name}: {handedness.score:.2f}"

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            frame,
            label,
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    return frame


async def main():
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=MODEL_PATH
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.5,
    )

    async with await real_connect() as machine:
        print(machine.resource_names)
        camera = Camera.from_robot(machine, CAMERA_NAME)

        with mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:
            try: 
                while True:
                    images, _ = await camera.get_images()

                    pil_image = Image.open(
                        BytesIO(images[0].data)
                    ).convert("RGB")

                    rgb_frame = np.ascontiguousarray(np.asarray(pil_image))
                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=rgb_frame,
                    )

                    timestamp_ms = time.time_ns()

                    # Avoid blocking the asyncio event loop during inference.
                    result = await asyncio.to_thread(
                        detector.detect_for_video,
                        mp_image,
                        timestamp_ms,
                    )

                    hand_frame = rgb_frame.copy()
                    hand_frame = draw_hand_boxes(hand_frame, result)

                    hand_detected = bool(result.hand_landmarks)
                    cv2.imshow('hands', hand_frame)
                    print("Hand detected:", hand_detected)

                    if hand_detected:
                        for hand_number, landmarks in enumerate(
                            result.hand_landmarks
                        ):
                            wrist = landmarks[0]
                            print(
                                f"Hand {hand_number}: "
                                f"wrist=({wrist.x:.2f}, {wrist.y:.2f})"
                            )

                    await asyncio.sleep(0.03)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
            except KeyboardInterrupt:
                print('Stop Stream')


if __name__ == "__main__":
    asyncio.run(main())