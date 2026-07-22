"""optimi-lab exceptions — re-exported from the shared ``lab_commons.exceptions``.

The generic ``ParameterException`` / ``QuantityException`` and the ``deprecated`` /
``not_implemented`` decorators were extracted to the standalone ``lab-commons`` package
(plan-lab-commons-standalone.md §3); optimi-lab has no project-specific exception of its own (unlike
motronics' ``FEMMException``), so this module is a thin re-export, not a fork.
"""

from lab_commons.exceptions import (
    ParameterException,
    QuantityException,
    deprecated,
    not_implemented,
)

__all__ = ['ParameterException', 'QuantityException', 'deprecated', 'not_implemented']
