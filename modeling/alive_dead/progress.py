"""Small optional wrapper around tqdm used by training and evaluation CLIs."""

from __future__ import annotations

from typing import Iterable, TypeVar

T = TypeVar("T")


def progress(iterable: Iterable[T], *, enabled: bool, **kwargs) -> Iterable[T]:
    """Wrap an iterable in tqdm when terminal progress is enabled."""
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm
    except ImportError as exc:
        raise RuntimeError(
            "Progress reporting requires tqdm. Install requirements-modeling.txt."
        ) from exc
    return tqdm(iterable, **kwargs)


def progress_write(message: str, *, enabled: bool) -> None:
    """Write without corrupting active tqdm bars."""
    if not enabled:
        print(message)
        return
    try:
        from tqdm.auto import tqdm
    except ImportError as exc:
        raise RuntimeError(
            "Progress reporting requires tqdm. Install requirements-modeling.txt."
        ) from exc
    tqdm.write(message)
