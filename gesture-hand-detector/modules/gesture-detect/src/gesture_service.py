import asyncio
import math
import os
import threading
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import mediapipe as mp
import numpy as np
from viam.components.camera import Camera
from viam.errors import MethodNotImplementedError, ValidationError
from viam.media.utils.pil import viam_to_pil_image
from viam.media.video import ViamImage
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import PointCloudObject, ResourceName
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.services.vision import (
    CaptureAllResult,
    Classification,
    Detection,
    Vision,
)
from viam.utils import ValueTypes


MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "gesture_recognizer.task"
)


def decode_rgb(image: ViamImage) -> np.ndarray:
    if not image.data:
        raise ValueError("Image data is empty")

    with viam_to_pil_image(image) as source:
        source.load()
        return np.array(
            source.convert("RGB"),
            dtype=np.uint8,
            copy=True,
            order="C",
        )


def top_category(groups, hand_index):
    if hand_index >= len(groups):
        return None

    if not groups[hand_index]:
        return None

    return groups[hand_index][0]


def result_to_classifications(
    result,
) -> list[Classification]:
    classifications = []

    for hand_index in range(
        len(result.hand_landmarks)
    ):
        gesture = top_category(
            result.gestures,
            hand_index,
        )

        if gesture is None:
            continue

        gesture_name = (
            gesture.category_name or ""
        ).strip()

        # MediaPipe returns "None" when a hand is
        # present but no canned gesture matches.
        if (
            not gesture_name
            or gesture_name.casefold() == "none"
        ):
            continue

        handedness = top_category(
            result.handedness,
            hand_index,
        )
        hand_name = f"hand_{hand_index + 1}"

        if handedness is not None:
            handedness_name = (
                handedness.category_name or ""
            ).strip().casefold()

            if handedness_name in {"left", "right"}:
                hand_name = f"{handedness_name}_hand"

        score = float(gesture.score or 0.0)

        if not math.isfinite(score):
            score = 0.0

        classifications.append(
            Classification(
                class_name=(
                    f"{hand_name}:{gesture_name}"
                ),
                confidence=min(
                    1.0,
                    max(0.0, score),
                ),
            )
        )

    classifications.sort(
        key=lambda item: item.confidence,
        reverse=True,
    )

    return classifications


class MediaPipeGestureClassifier:
    def __init__(
        self,
        model_path: Path,
        score_threshold: float,
        num_hands: int,
    ) -> None:
        classifier_options = (
            mp.tasks.components.processors
            .ClassifierOptions(
                score_threshold=score_threshold,
            )
        )

        options = (
            mp.tasks.vision.GestureRecognizerOptions(
                base_options=mp.tasks.BaseOptions(
                    model_asset_path=str(model_path),
                ),
                # Each Viam RPC supplies an independent image.
                running_mode=(
                    mp.tasks.vision.RunningMode.IMAGE
                ),
                num_hands=num_hands,
                min_hand_detection_confidence=0.6,
                min_hand_presence_confidence=0.6,
                min_tracking_confidence=0.5,
                canned_gesture_classifier_options=(
                    classifier_options
                ),
            )
        )

        self._recognizer = (
            mp.tasks.vision.GestureRecognizer
            .create_from_options(options)
        )
        self._native_lock = threading.Lock()
        self._closed = False

    async def classify(
        self,
        image: ViamImage,
    ) -> list[Classification]:
        # Copy the RPC data before entering the worker thread.
        data = bytes(image.data)
        mime_type = image.mime_type

        return await asyncio.to_thread(
            self._classify_sync,
            data,
            mime_type,
        )

    def _classify_sync(
        self,
        data: bytes,
        mime_type: str,
    ) -> list[Classification]:
        rgb_frame = decode_rgb(
            ViamImage(data, mime_type)
        )
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame,
        )

        with self._native_lock:
            if self._closed:
                raise RuntimeError(
                    "Gesture recognizer is closed"
                )

            result = self._recognizer.recognize(
                mp_image
            )

        return result_to_classifications(result)

    async def close(self) -> None:
        await asyncio.to_thread(self._close_sync)

    def _close_sync(self) -> None:
        with self._native_lock:
            if not self._closed:
                self._recognizer.close()
                self._closed = True


class GestureDetectorService(Vision, EasyResource):
    MODEL = "hackathons:gesture-detect:mediapipe"

    @classmethod
    def validate_config(
        cls,
        config: ComponentConfig,
    ) -> Tuple[Sequence[str], Sequence[str]]:
        fields = config.attributes.fields

        if (
            "camera_name" not in fields
            or not fields["camera_name"].string_value
        ):
            raise ValidationError(
                "Attribute 'camera_name' must be "
                "a non-empty string"
            )

        if "score_threshold" in fields:
            threshold = (
                fields["score_threshold"].number_value
            )

            if not 0.0 <= threshold <= 1.0:
                raise ValidationError(
                    "Attribute 'score_threshold' must "
                    "be between 0 and 1"
                )

        if "num_hands" in fields:
            num_hands = fields[
                "num_hands"
            ].number_value

            if (
                num_hands < 1
                or not float(num_hands).is_integer()
            ):
                raise ValidationError(
                    "Attribute 'num_hands' must be "
                    "a positive integer"
                )

        return [
            fields["camera_name"].string_value
        ], []

    @classmethod
    def new(
        cls,
        config: ComponentConfig,
        dependencies: Mapping[
            ResourceName,
            ResourceBase,
        ],
    ) -> "GestureDetectorService":
        service = cls(config.name)
        fields = config.attributes.fields

        service._camera_name = (
            fields["camera_name"].string_value
        )
        service._source_name = (
            fields["source_name"].string_value
            if (
                "source_name" in fields
                and fields[
                    "source_name"
                ].string_value
            )
            else "color"
        )

        score_threshold = (
            float(
                fields[
                    "score_threshold"
                ].number_value
            )
            if "score_threshold" in fields
            else 0.5
        )
        num_hands = (
            int(fields["num_hands"].number_value)
            if "num_hands" in fields
            else 2
        )

        model_path = MODEL_PATH

        if (
            "model_path" in fields
            and fields["model_path"].string_value
        ):
            model_path = Path(
                fields["model_path"].string_value
            )

            if not model_path.is_absolute():
                model_path = (
                    Path(__file__).resolve().parent
                    / model_path
                )

        if not model_path.is_file():
            raise ValidationError(
                "Gesture Recognizer model does not "
                f"exist: {model_path}"
            )

        camera_resource_name = (
            Camera.get_resource_name(
                service._camera_name
            )
        )

        try:
            service._camera = dependencies[
                camera_resource_name
            ]
        except KeyError as error:
            raise ValidationError(
                f"Camera dependency "
                f"{service._camera_name!r} "
                "was not provided"
            ) from error

        service._classifier = (
            MediaPipeGestureClassifier(
                model_path,
                score_threshold,
                num_hands,
            )
        )

        return service

    async def _get_color_image(
        self,
        camera_name: str,
    ) -> ViamImage:
        requested_camera = (
            camera_name or self._camera_name
        )

        if requested_camera != self._camera_name:
            raise ValueError(
                "This classifier is configured for "
                f"{self._camera_name!r}, not "
                f"{requested_camera!r}"
            )

        images, _ = await self._camera.get_images(
            filter_source_names=[
                self._source_name
            ]
        )

        if not images:
            raise RuntimeError(
                f"Camera {self._camera_name!r} "
                f"returned no "
                f"{self._source_name!r} image"
            )

        return images[0]

    async def get_classifications(
        self,
        image: ViamImage,
        count: int,
        *,
        extra: Optional[
            Mapping[str, ValueTypes]
        ] = None,
        timeout: Optional[float] = None,
    ) -> list[Classification]:
        classifications = (
            await self._classifier.classify(image)
        )

        return classifications[:max(0, count)]

    async def get_classifications_from_camera(
        self,
        camera_name: str,
        count: int,
        *,
        extra: Optional[
            Mapping[str, ValueTypes]
        ] = None,
        timeout: Optional[float] = None,
    ) -> list[Classification]:
        image = await self._get_color_image(
            camera_name
        )

        return await self.get_classifications(
            image,
            count,
        )

    async def capture_all_from_camera(
        self,
        camera_name: str,
        return_image: bool = False,
        return_classifications: bool = False,
        return_detections: bool = False,
        return_object_point_clouds: bool = False,
        *,
        extra: Optional[
            Mapping[str, ValueTypes]
        ] = None,
        timeout: Optional[float] = None,
    ) -> CaptureAllResult:
        if return_detections:
            raise MethodNotImplementedError(
                "get_detections"
            )

        if return_object_point_clouds:
            raise MethodNotImplementedError(
                "get_object_point_clouds"
            )

        image = await self._get_color_image(
            camera_name
        )
        classifications = (
            await self._classifier.classify(image)
            if return_classifications
            else None
        )

        return CaptureAllResult(
            image=image if return_image else None,
            classifications=classifications,
        )

    async def get_detections_from_camera(
        self,
        camera_name: str,
        *,
        extra: Optional[
            Mapping[str, ValueTypes]
        ] = None,
        timeout: Optional[float] = None,
    ) -> list[Detection]:
        raise MethodNotImplementedError(
            "get_detections_from_camera"
        )

    async def get_detections(
        self,
        image: ViamImage,
        *,
        extra: Optional[
            Mapping[str, ValueTypes]
        ] = None,
        timeout: Optional[float] = None,
    ) -> list[Detection]:
        raise MethodNotImplementedError(
            "get_detections"
        )

    async def get_object_point_clouds(
        self,
        camera_name: str,
        *,
        extra: Optional[
            Mapping[str, ValueTypes]
        ] = None,
        timeout: Optional[float] = None,
    ) -> list[PointCloudObject]:
        raise MethodNotImplementedError(
            "get_object_point_clouds"
        )

    async def get_properties(
        self,
        *,
        extra: Optional[
            Mapping[str, ValueTypes]
        ] = None,
        timeout: Optional[float] = None,
    ) -> Vision.Properties:
        return Vision.Properties(
            classifications_supported=True,
            detections_supported=False,
            object_point_clouds_supported=False,
            default_camera=self._camera_name,
        )

    async def close(self) -> None:
        await self._classifier.close()