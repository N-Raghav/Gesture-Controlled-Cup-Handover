import asyncio
import os
import time
from io import BytesIO
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
from viam.components.camera import Camera

from real_connect import real_connect


CAMERA_NAME = "cam"
MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "gesture_recognizer.task"
)
GESTURE_SCORE_THRESHOLD = 0.5


def top_category(groups, hand_index):
    if hand_index >= len(groups):
        return None

    if not groups[hand_index]:
        return None

    return groups[hand_index][0]


def gesture_summary(result):
    summary = []

    for hand_index in range(len(result.hand_landmarks)):
        handedness = top_category(
            result.handedness,
            hand_index,
        )
        gesture = top_category(
            result.gestures,
            hand_index,
        )

        hand_name = (
            handedness.category_name
            if handedness and handedness.category_name
            else f"Hand {hand_index + 1}"
        )
        gesture_name = (
            gesture.category_name
            if gesture and gesture.category_name
            else "Unrecognized"
        )
        confidence = (
            float(gesture.score or 0.0)
            if gesture
            else 0.0
        )

        summary.append(
            (hand_name, gesture_name, confidence)
        )

    return summary


def draw_gestures(frame, result, padding=20):
    height, width = frame.shape[:2]
    summary = gesture_summary(result)

    for hand_index, landmarks in enumerate(
        result.hand_landmarks
    ):
        xs = [
            int(point.x * width)
            for point in landmarks
        ]
        ys = [
            int(point.y * height)
            for point in landmarks
        ]

        x1 = max(0, min(xs) - padding)
        y1 = max(0, min(ys) - padding)
        x2 = min(width - 1, max(xs) + padding)
        y2 = min(height - 1, max(ys) + padding)

        hand_name, gesture_name, confidence = (
            summary[hand_index]
        )
        label = (
            f"{hand_name}: {gesture_name} "
            f"{confidence:.2f}"
        )

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
            0.65,
            (0, 255, 0),
            2,
        )

    return frame, summary


async def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Gesture model not found: {MODEL_PATH}"
        )

    classifier_options = (
        mp.tasks.components.processors.ClassifierOptions(
            score_threshold=GESTURE_SCORE_THRESHOLD,
        )
    )

    options = mp.tasks.vision.GestureRecognizerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(MODEL_PATH),
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.5,
        canned_gesture_classifier_options=(
            classifier_options
        ),
    )

    async with await real_connect() as machine:
        camera = Camera.from_robot(
            machine,
            CAMERA_NAME,
        )
        previous_labels = None

        with (
            mp.tasks.vision.GestureRecognizer
            .create_from_options(options)
        ) as recognizer:
            try:
                while True:
                    images, _ = await camera.get_images(
                        filter_source_names=["color"]
                    )

                    if not images:
                        print("Camera returned no color image")
                        await asyncio.sleep(0.1)
                        continue

                    with Image.open(
                        BytesIO(images[0].data)
                    ) as image:
                        rgb_frame = np.ascontiguousarray(
                            np.asarray(
                                image.convert("RGB")
                            )
                        )

                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=rgb_frame,
                    )

                    timestamp_ms = (
                        time.monotonic_ns() // 1_000_000
                    )

                    result = await asyncio.to_thread(
                        recognizer.recognize_for_video,
                        mp_image,
                        timestamp_ms,
                    )

                    display_frame = cv2.cvtColor(
                        rgb_frame,
                        cv2.COLOR_RGB2BGR,
                    )
                    display_frame, summary = (
                        draw_gestures(
                            display_frame,
                            result,
                        )
                    )

                    current_labels = tuple(
                        (hand_name, gesture_name)
                        for (
                            hand_name,
                            gesture_name,
                            _,
                        ) in summary
                    )

                    if current_labels != previous_labels:
                        if summary:
                            for (
                                hand_name,
                                gesture_name,
                                confidence,
                            ) in summary:
                                print(
                                    f"{hand_name}: "
                                    f"{gesture_name} "
                                    f"({confidence:.2f})"
                                )
                        else:
                            print("No hand detected")

                        previous_labels = current_labels

                    cv2.imshow(
                        "hand gestures",
                        display_frame,
                    )

                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break

                    await asyncio.sleep(0.03)
            finally:
                cv2.destroyAllWindows()


if __name__ == "__main__":
    asyncio.run(main())