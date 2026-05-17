"""Stage 5: load the trained Keras model + label classes and run prediction.

Mirrors `LABEL_REMAP` in notebooks/Pipeline.ipynb (cell 3) so the API returns
clean Vietnamese display labels instead of the raw filename-stem keys saved in
label_classes.npy. Keep this dict in sync with the notebook.

The current saved label_classes.npy includes a few keys (`test2`, `khong?2`)
that don't appear in the notebook's LABEL_REMAP — they are kept here as a
defensive superset so the predictor doesn't crash on stale artifacts.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

EXPECTED_INPUT_SHAPE = (20, 9)

# raw key (from label_classes.npy) -> clean display label
LABEL_REMAP: dict[str, str] = {
    # From notebooks/Pipeline.ipynb LABEL_REMAP
    "baonhieu2": "bao nhiêu",
    "C":         "C",
    "khong_2":   "không",
    "O2":        "O",
    "pink4":     "pink",
    "tôi3":      "tôi",
    "xinchao0":  "xin chào",
    # Extra keys present in the currently-saved label_classes.npy
    "khong?2":   "không",
    "test2":     "test",
}


def remap_label(raw: str) -> str:
    """Map a raw class string to its clean display form. Unknown keys pass through."""
    return LABEL_REMAP.get(raw, raw)


@dataclass
class Prediction:
    label: str                       # display (remapped) label
    raw_label: str                   # original key from label_classes.npy
    confidence: float
    top3: list[tuple[str, float]]    # display labels


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

        missing = [c for c in self.classes if str(c) not in LABEL_REMAP]
        if missing:
            print(
                f"[predictor] warning: no LABEL_REMAP entry for {missing}; "
                f"raw keys will be returned as-is."
            )

    def predict(self, x: np.ndarray) -> Prediction:
        if x.shape != (1, *EXPECTED_INPUT_SHAPE):
            raise ValueError(f"expected input shape (1, 20, 9), got {x.shape}")

        probs = self.model.predict(x, verbose=0)[0]
        order = np.argsort(probs)[::-1]
        top3 = [(remap_label(str(self.classes[i])), float(probs[i])) for i in order[:3]]
        best_idx = int(order[0])
        raw = str(self.classes[best_idx])
        return Prediction(
            label=remap_label(raw),
            raw_label=raw,
            confidence=float(probs[best_idx]),
            top3=top3,
        )


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.normpath(os.path.join(here, "..", ".."))
    model_path = os.path.join(root, "results", "models", "model_best.h5")
    labels_path = os.path.join(root, "data", "processed", "label_classes.npy")

    p = Predictor(model_path, labels_path)
    print(f"raw classes ({len(p.classes)}): {list(p.classes)}")
    print(f"display      : {[remap_label(str(c)) for c in p.classes]}")
    print(f"model input shape: {p.model.input_shape}")

    x = np.zeros((1, 20, 9), dtype=np.float32)
    pred = p.predict(x)
    print(f"\nzero-input prediction:")
    print(f"  display label = {pred.label!r}  raw = {pred.raw_label!r}  conf={pred.confidence:.4f}")
    print(f"  top3 = {pred.top3}")