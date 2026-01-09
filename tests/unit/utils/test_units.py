import numpy as np
import pytest

from optimi_lab.utils import quantities
from optimi_lab.utils.exceptions import QuantityException


def test_2DPoint():
    Q_0m = quantities.Q_(0, 'm')
    Q_1m = quantities.Q_(1, 'm')
    Q_1000mm = quantities.Q_(1000, 'mm')
    p1 = [Q_0m, Q_1m]
    p2 = [quantities.Q_0mm, Q_1000mm]
    assert quantities.is_equal_2DPoint(p1, p2)
    Q_180deg = quantities.Q_(180, 'deg')
    Q_pi_rad = quantities.Q_(np.pi, 'rad')
    p_3 = [Q_0m, Q_180deg]
    p_4 = [quantities.Q_0mm, Q_pi_rad]
    assert quantities.is_equal_2DPoint(p_3, p_4)


def test_PydanticQuantity():
    assert quantities.PydanticQuantity.validate(quantities.Q_0mm) == quantities.Q_0mm

    with pytest.raises(QuantityException, match='Expected pint.Quantity'):
        quantities.PydanticQuantity.validate(1)


if __name__ == '__main__':
    # test_2DPoint()
    test_PydanticQuantity()
