"""Refusals — one exception type and one guard shape for the whole package.

Every refusal in ``optimi_lab`` is raised by :func:`refuse`, and every refusal that rejects a
member of a declared set NAMES that set. Two defects are foreclosed:

* The old package refused in four shapes — ``KeyError('unmatched flag invalid')``, ``NameError``,
  an ``AttributeError`` raised where nothing was missing, and a bare ``TypeError: 'NoneType'
  object is not subscriptable`` from a matrix that was never built — so no single clause caught
  them, and two of the four read as bugs in the package rather than answers to the caller.
* The old ``surrogate_type`` refusal said only "must contain valid surrogate model type, got X"
  while the constant that exists to spell the valid set sat one module away, unquoted.

A guard answers a question the CALLER asked, so it returns the checked value: ``points =
require_axis(points, n_var, 1, 'the array returned by ask()')`` reads the constraint where it is
enforced.

THAT LINE NAMED A FUNCTION THAT DOES NOT EXIST UNTIL 2026-09-19: it demonstrated the package's one
guard shape through a three-argument call whose name was never defined here, measured as the single
occurrence of that spelling in the tree. It is the shape this family keeps rediscovering -- a dead
spelling survives longest in the PROSE that explains the thing, because prose reads as harmless and
nobody deletes it. Here it was worse than harmless: a docstring that cannot be executed, inside the
module whose whole job is to make what this package says about itself true.

    >>> import numpy as np
    >>> require_axis(np.zeros((4, 2)), 2, 1, 'a batch').shape
    (4, 2)
    >>> require_axis(np.zeros(4), 2, 1, 'a batch')
    Traceback (most recent call last):
    optimi_lab.core.errors.Refusal: a batch must be a 2-D array with 2 column(s), got shape (4,)
"""

from enum import Enum
from typing import NoReturn

import numpy as np

__all__ = [
    'Refusal',
    'coerce_enum',
    'refuse',
    'require_axis',
    'require_finite',
    'require_positive',
]


class Refusal(ValueError):
    """An argument cannot be served, and no value would be an honest answer to it.

    A ``ValueError`` subclass on purpose: every refusal this package raises is about a value
    the caller supplied, so ``except ValueError`` catches them all, and ``except Refusal``
    separates them from the interpreter's own.
    """


def refuse(message: str) -> NoReturn:
    """Raise :class:`Refusal` carrying ``message``."""
    raise Refusal(message)


def coerce_enum(enum_type: type[Enum], value: object, what: str) -> Enum:
    """Return the member of ``enum_type`` that ``value`` names, or refuse naming every member.

    ``enum_type(value)`` refuses too, but its ``ValueError`` prints only the offending value;
    the members are spelled here because a caller that guessed wrong needs the valid set.
    """
    if isinstance(value, enum_type):
        return value
    members = [member.value for member in enum_type]
    if value not in members:
        refuse(f'{what} must be one of {members}, got {value!r}')
    return enum_type(value)


def require_positive(value: float, what: str) -> float:
    """Return ``value`` if it is strictly greater than zero, else refuse.

    Written as ``not value > 0`` rather than ``value <= 0`` so that NaN — for which every
    comparison is False — is refused instead of passing.
    """
    if not value > 0:
        refuse(f'{what} must be greater than 0, got {value!r}')
    return value


def require_finite(values: np.ndarray, what: str) -> np.ndarray:
    """Return ``values`` if every entry is finite, else refuse naming the offending index."""
    array = np.asarray(values, dtype=float)
    offending = np.argwhere(~np.isfinite(array))
    if offending.size:
        cell = tuple(int(position) for position in offending[0])
        refuse(f'{what} must be finite, got {float(array[cell])!r} at index {list(cell)}')
    return array


def require_axis(values: np.ndarray, n: int, axis: int, what: str) -> np.ndarray:
    """Return ``values`` if it is 2-D with exactly ``n`` entries along ``axis``, else refuse.

    One function covers both sides of a shape because the two guards are the same guard:
    ``axis=0`` counts points, ``axis=1`` counts columns. This is what replaces the old bare
    ``TypeError`` — a one-dimensional batch, or a matrix the caller never built (``None``), is
    named here instead of being subscripted downstream.
    """
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[axis] != n:
        side = ('row', 'column')[axis]
        refuse(f'{what} must be a 2-D array with {n} {side}(s), got shape {array.shape}')
    return array
