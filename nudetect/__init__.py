import numpy as np
import onnxruntime
from PIL import Image

from .detector_utils import preprocess_image

MODEL_CHECKPOINT_URL = "https://github.com/notAI-tech/NudeNet/releases/download/v0/detector_v2_default_checkpoint.onnx"
CLASSES = [  # order is important
    "EXPOSED_ANUS",
    "EXPOSED_ARMPITS",
    "COVERED_BELLY",
    "EXPOSED_BELLY",
    "COVERED_BUTTOCKS",
    "EXPOSED_BUTTOCKS",
    "FACE_F",
    "FACE_M",
    "COVERED_FEET",
    "EXPOSED_FEET",
    "COVERED_BREAST_F",
    "EXPOSED_BREAST_F",
    "COVERED_GENITALIA_F",
    "EXPOSED_GENITALIA_F",
    "EXPOSED_BREAST_M",
    "EXPOSED_GENITALIA_M"
]


class Detector:

    def __init__(self, model_checkpoint_path):
        """
        model = Detector()
        """

        self.detection_model = onnxruntime.InferenceSession(model_checkpoint_path)

    def detect(self, image: Image, mode="default", min_prob=None):
        if mode == "fast":
            image_rgb, scale = preprocess_image(image, min_side=480, max_side=800)
            if not min_prob:
                min_prob = 0.5
        else:
            image_rgb, scale = preprocess_image(image)
            if not min_prob:
                min_prob = 0.6

        outputs = self.detection_model.run(
            [s_i.name for s_i in self.detection_model.get_outputs()],
            {self.detection_model.get_inputs()[0].name: np.expand_dims(image_rgb, axis=0)},
        )

        labels = [op for op in outputs if op.dtype == "int32"][0]
        scores = [op for op in outputs if isinstance(op[0][0], np.float32)][0]
        boxes = [op for op in outputs if isinstance(op[0][0], np.ndarray)][0]

        boxes /= scale
        processed_boxes = []
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            if score < min_prob:
                continue
            box = box.astype(int).tolist()
            label = CLASSES[label]
            processed_boxes.append(
                {"box": [int(c) for c in box], "score": float(score), "label": label}
            )

        return processed_boxes
