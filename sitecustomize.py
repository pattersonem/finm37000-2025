import numpy as _np

# NumPy<2.0 does not expose np.concat, but the tests expect it.
# Provide a thin alias to np.concatenate so local runs behave like
# the grading environment, which uses NumPy>=2.0.
if not hasattr(_np, "concat"):
    _np.concat = _np.concatenate  # type: ignore[attr-defined]
