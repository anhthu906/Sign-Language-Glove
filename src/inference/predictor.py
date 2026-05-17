"""Stage 5: load the trained Keras model + label classes and run prediction."""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

EXPECTED_INPUT_SHAPE = (20, 9)


@dataclass
class Prediction:
    label: str
    confidence: float
    top3: list[tuple[str, float]]


class Predictor:
    def __init__(self, model_path: str, labels_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"model not found at {model_path}. Run notebooks/Pipeline.ipynb first."
            )
        if not os.path.exists(labels_path):
            raise FileNotFoundError(
                f"label classes not found at {labels_path}. Run notebooks/Pipeline.ipynb first."
            )

        from tensorflow.keras.models import load_model

        self.model = load_model(model_path)
        self.classes: np.ndarray = np.load(labels_path, allow_pickle=True)

        got_shape = tuple(self.model.input_shape[1:])
        if got_shape != EXPECTED_INPUT_SHAPE:
            raise ValueError(
                f"model input shape mismatch: got {got_shape}, expected {EXPECTED_INPUT_SHAPE}. "
                f"The model at {model_path} was not trained by the current Pipeline.ipynb."
            )

        n_out = int(self.model.output_shape[-1])
        if n_out != len(self.classes):
            raise ValueError(
                f"model output classes ({n_out}) != label_classes.npy length ({len(self.classes)}). "
                f"The model and label encoder are out of sync."
            )

    def predict(self, x: np.ndarray) -> Prediction:
        if x.shape != (1, *EXPECTED_INPUT_SHAPE):
            raise ValueError(f"expected input shape (1, 20, 9), got {x.shape}")

        probs = self.model.predict(x, verbose=0)[0]  # (n_classes,)
        order = np.argsort(probs)[::-1]
        top3 = [(str(self.classes[i]), float(probs[i])) for i in order[:3]]
        best_idx = int(order[0])
        return Prediction(
            label=str(self.classes[best_idx]),
            confidence=float(probs[best_idx]),
            top3=top3,
        )


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.normpath(os.path.join(here, "..", ".."))
    model_path = os.path.join(root, "results", "models", "model_best.h5")
    labels_path = os.path.join(root, "data", "processed", "label_classes.npy")

    p = Predictor(model_path, labels_path)
    print(f"classes ({len(p.classes)}): {list(p.classes)}")
    print(f"model input shape: {p.model.input_shape}")

    x = np.zeros((1, 20, 9), dtype=np.float32)
    pred = p.predict(x)
    print(f"\nzero-input prediction:")
    print(f"  label={pred.label}  conf={pred.confidence:.4f}")
    print(f"  top3 = {pred.top3}")
