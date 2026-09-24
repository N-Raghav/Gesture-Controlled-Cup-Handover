import asyncio
import math
import os
import threading
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

# MediaPipe imports matplotlib even though this service does not draw images.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import mediapipe as mp
import numpy as np
from viam.components.camera import Camera
from viam.errors import MethodNotImplementedError, ValidationError
from viam.media.utils.pil import viam_to_pil_image
from viam.media.video import ViamImage
from viam.module.module import Module
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
    / "hand_landmarker.task"
)


def decode_rgb(image: ViamImage) -> np.ndarray:
    """Decode a Viam image without changing its raster orientation."""
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


def landmark_bounds(
    landmarks: Any,
    width: int,
    height: int,
    padding: int,
) -> Optional[Tuple[int, int, int, int]]:
    """Convert normalized MediaPipe landmarks to a pixel bounding box."""
    xs = [
        float(point.x)
        for point in landmarks
        if point.x is not None
        and math.isfinite(float(point.x))
    ]
    ys = [
        float(point.y)
        for point in landmarks
        if point.y is not None
        and math.isfinite(float(point.y))
    ]

    if not xs or not ys:
        return None

    # x_max and y_max are exclusive image bounds.
    x_min = max(
        0,
        min(width, math.floor(min(xs) * width) - padding),
    )
    y_min = max(
        0,
        min(height, math.floor(min(ys) * height) - padding),
    )
    x_max = max(
        0,
        min(
            width,
            math.floor(max(xs) * width) + 1 + padding,
        ),
    )
    y_max = max(
        0,
        min(
            height,
            math.floor(max(ys) * height) + 1 + padding,
        ),
    )

    if x_min >= x_max or y_min >= y_max:
        return None

    return x_min, y_min, x_max, y_max


def hand_identity(
    result: Any,
    hand_index: int,
) -> Tuple[str, float]:
    """Return a hand label and confidence."""
    label = "hand"
    confidence = 1.0

    handedness = getattr(result, "handedness", ()) or ()
    categories = (
        handedness[hand_index]
        if hand_index < len(handedness)
        else ()
    )

    if not categories:
        return label, confidence

    category = categories[0]
    name = (
        getattr(category, "category_name", None) or ""
    ).strip().casefold()

    if name in {"left", "right"}:
        label = f"{name}_hand"

    score = getattr(category, "score", None)
    if score is not None and math.isfinite(float(score)):
        confidence = min(1.0, max(0.0, float(score)))

    return label, confidence


def result_to_detections(
    result: Any,
    width: int,
    height: int,
    padding: int,
) -> list[Detection]:
    """Convert HandLandmarker output to Viam detections."""
    detections = []

    for hand_index, landmarks in enumerate(
        getattr(result, "hand_landmarks", ()) or ()
    ):
        bounds = landmark_bounds(
            landmarks,
            width,
            height,
            padding,
        )

        if bounds is None:
            continue

        label, confidence = hand_identity(
            result,
            hand_index,
        )

        detections.append(
            Detection(
                x_min=bounds[0],
                y_min=bounds[1],
                x_max=bounds[2],
                y_max=bounds[3],
                confidence=confidence,
                class_name=label,
            )
        )

    return detections


class MediaPipeHandDetector:
    """Thread-safe wrapper around MediaPipe HandLandmarker."""

    def __init__(
        self,
        model_path: Path,
        padding: int,
    ) -> None:
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(model_path)
            ),
            # Each Vision RPC contains an independent image.
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.5,
        )

        self._landmarker = (
            mp.tasks.vision.HandLandmarker.create_from_options(
                options
            )
        )
        self._padding = padding
        self._native_lock = threading.Lock()
        self._closed = False

    async def detect(
        self,
        image: ViamImage,
    ) -> list[Detection]:
        # Copy the RPC image before moving inference to a worker.
        data = bytes(image.data)
        mime_type = image.mime_type

        return await asyncio.to_thread(
            self._detect_sync,
            data,
            mime_type,
        )

    def _detect_sync(
        self,
        data,
        mime_type,
    ) -> list[Detection]:
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
                    "Hand detector is closed"
                )

            result = self._landmarker.detect(mp_image)

        height, width = rgb_frame.shape[:2]

        return result_to_detections(
            result,
            width,
            height,
            self._padding,
        )

    async def close(self) -> None:
        await asyncio.to_thread(self._close_sync)

    def _close_sync(self) -> None:
        with self._native_lock:
            if not self._closed:
                self._landmarker.close()
                self._closed = True


class HandDetectorService(Vision, EasyResource):
    """Viam Vision detector backed by HandLandmarker."""

    # Change "local" to your Viam public namespace before
    # publishing this module to the registry.
    MODEL = "hackathons:hand-detect:mediapipe"

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
                "Attribute 'camera_name' must be a "
                "non-empty string"
            )

        if "padding_px" in fields:
            padding = fields["padding_px"].number_value

            if (
                padding < 0
                or not float(padding).is_integer()
            ):
                raise ValidationError(
                    "Attribute 'padding_px' must be "
                    "a non-negative integer"
                )

        # This makes the camera an implicit dependency.
        return [fields["camera_name"].string_value], []

    @classmethod
    def new(
        cls,
        config: ComponentConfig,
        dependencies: Mapping[
            ResourceName,
            ResourceBase,
        ],
    ) -> "HandDetectorService":
        service = cls(config.name)
        fields = config.attributes.fields

        service._camera_name = (
            fields["camera_name"].string_value
        )
        service._source_name = (
            fields["source_name"].string_value
            if "source_name" in fields
            else "color"
        )

        padding = (
            int(fields["padding_px"].number_value)
            if "padding_px" in fields
            else 20
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
                f"HandLandmarker model does not exist: "
                f"{model_path}"
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

        service._detector = MediaPipeHandDetector(
            model_path,
            padding,
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
                f"This detector is configured for "
                f"{self._camera_name!r}, not "
                f"{requested_camera!r}"
            )

        images, _ = await self._camera.get_images(
            filter_source_names=[self._source_name]
        )

        if not images:
            raise RuntimeError(
                f"Camera {self._camera_name!r} "
                f"returned no {self._source_name!r} image"
            )

        return images[0]

    async def get_detections(
        self,
        image: ViamImage,
        *,
        extra: Optional[
            Mapping[str, ValueTypes]
        ] = None,
        timeout: Optional[float] = None,
    ) -> list[Detection]:
        # detections-to-segments supplies the color image
        # corresponding to its depth capture. Do not fetch a
        # second camera image here.
        return await self._detector.detect(image)

    async def get_detections_from_camera(
        self,
        camera_name: str,
        *,
        extra: Optional[
            Mapping[str, ValueTypes]
        ] = None,
        timeout: Optional[float] = None,
    ) -> list[Detection]:
        image = await self._get_color_image(
            camera_name
        )
        return await self.get_detections(image)

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
        if return_classifications:
            raise MethodNotImplementedError(
                "get_classifications"
            )

        if return_object_point_clouds:
            raise MethodNotImplementedError(
                "get_object_point_clouds"
            )

        image = await self._get_color_image(
            camera_name
        )

        detections = (
            await self.get_detections(image)
            if return_detections
            else None
        )

        return CaptureAllResult(
            image=image if return_image else None,
            detections=detections,
        )

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
        raise MethodNotImplementedError(
            "get_classifications_from_camera"
        )

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
        raise MethodNotImplementedError(
            "get_classifications"
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
            classifications_supported=False,
            detections_supported=True,
            object_point_clouds_supported=False,
            default_camera=self._camera_name,
        )

    async def close(self) -> None:
        await self._detector.close()


if __name__ == "__main__":
    asyncio.run(Module.run_from_registry())