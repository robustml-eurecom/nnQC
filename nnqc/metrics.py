"""Pluggable QC metrics for :func:`nnqc.check`.

``check(..., metric=...)`` measures the agreement between the candidate mask and
the model's reconstruction. The metric is fully user-definable: pass a built-in
name, a bare callable ``fn(pred, ref) -> float`` (e.g. ``medpy.metric.binary.hd95``),
or a :class:`Metric` subclass for full control over naming, orientation and how
empty masks are handled.

Examples
--------
Default (Dice)::

    nnqc.check(img, mask, task="prostate")

A medpy distance metric (lower is better, needs non-empty masks)::

    from medpy.metric.binary import hd95
    nnqc.check(img, mask, task="prostate", metric=hd95)

A custom metric class::

    from nnqc.metrics import Metric

    class SurfaceDice(Metric):
        name = "surface_dice"
        higher_is_better = True
        def score(self, pred, ref):
            ...
            return value

    nnqc.check(img, mask, task="prostate", metric=SurfaceDice())
"""
from __future__ import annotations

from typing import Callable

import numpy as np

ArrayLike = "np.ndarray"


class Metric:
    """Base class / template for a QC metric.

    Subclass and implement :meth:`score`; ``pred`` and ``ref`` are binary
    numpy arrays of identical shape (the reconstruction and the candidate mask,
    respectively). Set the class attributes to describe the metric:

    - ``name``: short identifier, surfaced on the result.
    - ``higher_is_better``: whether a larger value means better agreement
      (True for overlap metrics like Dice, False for distances like HD95).
    - ``empty_value``: value returned when one mask is empty and the other is
      not (degenerate case); defaults to NaN so it is dropped from aggregates.
    - ``empty_both_value``: value when *both* masks are empty (perfect trivial
      agreement); defaults to NaN. Dice overrides this to 1.0.
    """

    name: str = "metric"
    higher_is_better: bool = True
    empty_value: float = float("nan")
    empty_both_value: float = float("nan")

    def score(self, pred: ArrayLike, ref: ArrayLike) -> float:  # noqa: D401
        raise NotImplementedError

    def __call__(self, pred, ref) -> float:
        pred = np.asarray(pred) > 0.5
        ref = np.asarray(ref) > 0.5
        p_any, r_any = bool(pred.any()), bool(ref.any())
        if not p_any and not r_any:
            return float(self.empty_both_value)
        if not p_any or not r_any:
            return float(self.empty_value)
        try:
            return float(self.score(pred, ref))
        except Exception:
            return float(self.empty_value)


class DiceMetric(Metric):
    """Soft-free Dice overlap. 1.0 = identical, 0.0 = disjoint."""

    name = "dice"
    higher_is_better = True
    empty_both_value = 1.0  # empty vs empty = perfect agreement

    def __init__(self, eps: float = 1e-6):
        self.eps = eps

    def score(self, pred, ref):
        inter = np.logical_and(pred, ref).sum()
        denom = pred.sum() + ref.sum()
        return (2.0 * inter + self.eps) / (denom + self.eps)


class IoUMetric(Metric):
    """Intersection-over-union (Jaccard). 1.0 = identical."""

    name = "iou"
    higher_is_better = True
    empty_both_value = 1.0

    def __init__(self, eps: float = 1e-6):
        self.eps = eps

    def score(self, pred, ref):
        inter = np.logical_and(pred, ref).sum()
        union = np.logical_or(pred, ref).sum()
        return (inter + self.eps) / (union + self.eps)


class FunctionMetric(Metric):
    """Wrap a bare ``fn(pred, ref) -> float`` as a :class:`Metric`.

    Use for third-party functions such as ``medpy.metric.binary.hd95`` or
    ``assd``. Set ``higher_is_better=False`` for distance metrics. Extra keyword
    arguments are forwarded to ``fn`` on every call (e.g. ``voxelspacing=...``).
    """

    def __init__(self, fn: Callable, name: str | None = None,
                 higher_is_better: bool = True,
                 empty_value: float = float("nan"),
                 empty_both_value: float = float("nan"), **fn_kwargs):
        self.fn = fn
        self.name = name or getattr(fn, "__name__", "custom")
        self.higher_is_better = higher_is_better
        self.empty_value = empty_value
        self.empty_both_value = empty_both_value
        self.fn_kwargs = fn_kwargs

    def score(self, pred, ref):
        return self.fn(pred, ref, **self.fn_kwargs)


# Built-in metrics addressable by name (CLI and convenience).
_BUILTINS = {
    "dice": lambda: DiceMetric(),
    "iou": lambda: IoUMetric(),
    "jaccard": lambda: IoUMetric(),
}
# Distance metrics from medpy, registered lazily (optional dependency).
_MEDPY = {
    "hd95": ("hd95", False),
    "hd": ("hd", False),
    "assd": ("assd", False),
    "asd": ("asd", False),
}


def get_metric(name: str) -> Metric:
    """Return a built-in metric by name (``dice``, ``iou``, ``hd95``, ``assd`` ...)."""
    key = name.lower()
    if key in _BUILTINS:
        return _BUILTINS[key]()
    if key in _MEDPY:
        fn_name, hib = _MEDPY[key]
        try:
            from medpy.metric import binary as medpy_binary
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                f"metric {name!r} requires medpy. Install with `pip install medpy`."
            ) from exc
        return FunctionMetric(getattr(medpy_binary, fn_name), name=fn_name, higher_is_better=hib)
    raise ValueError(f"Unknown metric {name!r}. Built-ins: {sorted(_BUILTINS) + sorted(_MEDPY)}.")


def as_metric(obj) -> Metric:
    """Coerce ``obj`` into a :class:`Metric`.

    Accepts a Metric instance, a metric name (str), a Metric subclass, or any
    bare callable ``fn(pred, ref) -> float``. ``None`` yields the default Dice.
    """
    if obj is None:
        return DiceMetric()
    if isinstance(obj, Metric):
        return obj
    if isinstance(obj, str):
        return get_metric(obj)
    if isinstance(obj, type) and issubclass(obj, Metric):
        return obj()
    if callable(obj):
        return FunctionMetric(obj)
    raise TypeError(
        f"metric must be a Metric, a name, or a callable fn(pred, ref); got {type(obj)!r}."
    )
