from __future__ import annotations

from collections.abc import Callable

CancellationProbe = Callable[[], bool]


class CancellationRequested(RuntimeError):
    """A cooperative DepthWizard job cancellation reached a safe interruption boundary."""


def raise_if_cancelled(probe: CancellationProbe | None) -> None:
    if probe is not None and probe():
        raise CancellationRequested("DepthWizard processing was cancelled by the operator")
