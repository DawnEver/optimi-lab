"""Unit conversion implemented using pint."""

from typing import Annotated, NewType

import numpy as np
from pint import Quantity as PintQuantityType
from pint import Unit, get_application_registry
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, GetCoreSchemaHandler
from pydantic_core import core_schema

from .exceptions import QuantityException

__all__ = [
    'CONSTANTS',
    'Q_',
    'AngleType',
    'BaseModel_with_q',
    'Cartesian2DPoint',
    'LengthType',
    'Polar2DPoint',
    'Q_0deg',
    'Q_0mm',
    'Q_0s',
    'Q_180deg',
    'Q_360deg',
    'TimeType',
    'get_quantity_type',
    'is_equal_2DPoint',
]

# Use the application registry to ensure all processes use the same registry
ureg = get_application_registry()


Q_ = ureg.Quantity
"""
Q_ converts a string/dict/tuple to a Quantity object. The numeric part will be automatically converted to float.
"""


class PydanticQuantity:
    """Enable Pydantic to support pint quantity validation."""

    @classmethod
    def __get_pydantic_core_schema__(cls, _, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        return core_schema.no_info_after_validator_function(
            cls.validate,
            handler(Q_),
            serialization=core_schema.to_string_ser_schema(),
        )

    @classmethod
    def validate(cls, v) -> PintQuantityType:
        """Args:
            v: value to validate
        Returns:
            PintQuantityType: validated value
        Raises:
            QuantityException: if the value is not a Quantity.
        """
        if not isinstance(v, PintQuantityType):
            msg = 'Expected pint.Quantity'
            raise QuantityException(msg)
        return v


def get_quantity_type(
    default_unit: str,
) -> type[PintQuantityType]:
    """Generate a physical quantity type annotation with a default unit."""
    return Annotated[
        PintQuantityType,
        BeforeValidator(Q_),
        Field(..., json_schema_extra={'unit': default_unit}),  # reference unit
        PydanticQuantity,
    ]


# Geometry
Q_1 = Q_(1.0)
LengthType = NewType('LengthType', get_quantity_type('mm'))
Q_0mm = Q_(0.0, 'mm')
Q_1mm = Q_(1.0, 'mm')
AreaType = NewType('AreaType', get_quantity_type('mm^2'))
Q_0mm2 = Q_(0.0, 'mm^2')
Q_1mm2 = Q_(1.0, 'mm^2')
Q_1m2 = Q_(1.0, 'm^2')
VolumeType = NewType('VolumeType', get_quantity_type('mm^3'))
Q_0mm3 = Q_(0.0, 'mm^3')
Q_1mm3 = Q_(1.0, 'mm^3')
AngleType = NewType('AngleType', get_quantity_type('deg'))
Q_360deg = Q_(360.0, 'deg')
Q_180deg = Q_(180.0, 'deg')
Q_90deg = Q_(90.0, 'deg')
Q_0deg = Q_(0.0, 'deg')
Q_1rad = Q_(1.0, 'rad')
Q_2pi = Q_360deg.to('rad')
Q_pi = Q_180deg.to('rad')
AngleSpeedType = NewType('AngleSpeedType', get_quantity_type('rad/s'))
Q_0rpm = Q_(0.0, 'rpm')
# Physics
TimeType = NewType('TimeType', get_quantity_type('s'))
Q_0s = Q_(0.0, 's')
MassType = NewType('MassType', get_quantity_type('kg'))
Q_0kg = Q_(0.0, 'kg')
Q_1kg = Q_(1.0, 'kg')
TemperatureType = NewType('TemperatureType', get_quantity_type('K'))
Q_0degC = Q_(0.0, 'degC')
Q_20degC = Q_(20.0, 'degC')
PerTemperatureType = NewType('PerTemperatureType', get_quantity_type('1/K'))
DensityType = NewType('DensityType', get_quantity_type('g/cm^3'))
FrequencyType = NewType('FrequencyType', get_quantity_type('Hz'))
Q_0Hz = Q_(0.0, 'Hz')
Q_1Hz = Q_(1.0, 'Hz')
Q_50Hz = Q_(50.0, 'Hz')
## Mechanics
ForceType = NewType('ForceType', get_quantity_type('N'))
Q_1N = Q_(1.0, 'N')
TorqueType = NewType('TorqueType', get_quantity_type('N*m'))
Q_0Nm = Q_(0.0, 'N*m')
Q_1Nm = Q_(1.0, 'N*m')
## Power / Power loss
PowerType = NewType('PowerType', get_quantity_type('W'))
Q_0W = Q_(0.0, 'W')
Q_1W = Q_(1.0, 'W')
PowerPerMassType = NewType('PowerPerMassType', get_quantity_type('W/kg'))

## Energy / Loss
EnergyType = NewType('EnergyType', get_quantity_type('J'))
Q_0J = Q_(0.0, 'J')
Q_1J = Q_(1.0, 'J')
## Electromagnetics
VoltageType = NewType('VoltageType', get_quantity_type('V'))
CurrentType = NewType('CurrentType', get_quantity_type('A'))
Q_1A = Q_(1.0, 'A')
Q_1V = Q_(1.0, 'V')
### Current density
CurrentDensityType = NewType('CurrentDensityType', get_quantity_type('A/mm^2'))
Q_0A_per_mm2 = Q_(0.0, 'A/mm^2')
Q_1A_per_mm2 = Q_(1.0, 'A/mm^2')
### Magnetic flux density B
FluxDensityType = NewType('FluxDensityType', get_quantity_type('T'))
Q_0T = Q_(0.0, 'T')
Q_1T = Q_(1.0, 'T')
### Magnetic field intensity H
MagFieldIntensityType = NewType('MagFieldIntensityType', get_quantity_type('A/m'))
Q_0A_per_m = Q_(0.0, 'A/m')
### Magnetomotive force
MMFType = NewType('MMFType', get_quantity_type('A*turn'))
Q_1At = Q_(1.0, 'A*turn')
### Magnetic vector potential A
MagVectorPotentialType = NewType('MagVectorPotentialType', get_quantity_type('Wb/m'))
Q_0Wb_per_m = Q_(0.0, 'Wb/m')
Q_1Wb_per_m = Q_(1.0, 'Wb/m')
### Electrical conductivity
ElecConductivityType = NewType('ElecConductivityType', get_quantity_type('MS/m'))
Q_0MS_per_m = Q_(0.0, 'MS/m')
### Resistivity
ResistivityType = NewType('ResistivityType', get_quantity_type('ohm*m'))
Q_0ohm_m = Q_(0.0, 'ohm*m')
### Permeability
PermeabilityType = NewType('PermeabilityType', get_quantity_type('H/m'))
### Flux linkage
FluxLinkageType = NewType('FluxLinkageType', get_quantity_type('Wb'))
Q_0Wb = Q_(0.0, 'Wb')
Q_1Wb = Q_(1.0, 'Wb')
Q_10Wb = Q_(10.0, 'Wb')
### Inductance
InductanceType = NewType('InductanceType', get_quantity_type('H'))
Q_0H = Q_(0.0, 'H')
### Resistance
ResistanceType = NewType('ResistanceType', get_quantity_type('ohm'))
Q_0ohm = Q_(0.0, 'ohm')
Q_1ohm = Q_(1.0, 'ohm')
# Do not use np.array because coordinate units may differ
Cartesian2DPoint = Annotated[
    list[LengthType, LengthType],
    Field(..., json_schema_extra={'type': 'list', 'items': {'minItems': 2, 'maxItems': 2}}),
]
Polar2DPoint = Annotated[
    list[LengthType, AngleType],
    Field(..., json_schema_extra={'type': 'list', 'items': {'minItems': 2, 'maxItems': 2}}),
]


Cartesian2DPoint_Array = Annotated[
    list[np.ndarray[LengthType], np.ndarray[LengthType]],
    Field(..., json_schema_extra={'type': 'list', 'items': {'minItems': 2, 'maxItems': 2}}),
]
Polar2DPoint_Array = Annotated[
    list[np.ndarray[LengthType], np.ndarray[AngleType]],
    Field(..., json_schema_extra={'type': 'list', 'items': {'minItems': 2, 'maxItems': 2}}),
]


def Q_list2array(
    Q_list: list[PintQuantityType], Q_unit: str | Unit, has_unit: bool = True
) -> PintQuantityType | np.ndarray:
    """Convert a list of Quantities to an array with float type.

    Args:
        Q_list (list[PintQuantityType]): list of pint.Quantity.
        Q_unit (str | Unit): target unit, can be a string or a Unit object.
        has_unit (bool): whether to keep units. If False, returns a pure numeric array.

    Returns:
        PintQuantityType: converted array with unit Q_unit, or a numpy array without unit.

    """
    q_array = np.array([Q_.to(Q_unit).magnitude for Q_ in Q_list], dtype=float)
    if has_unit:
        q_array *= ureg.Unit(Q_unit)

    return q_array


def array2list_2Dpoint(
    point_array: Cartesian2DPoint_Array | Polar2DPoint_Array,
) -> list[Cartesian2DPoint | Polar2DPoint]:
    """Convert coordinate arrays convenient for numpy computations back to coordinate lists."""
    return [[point_array[0][idx], point_array[1][idx]] for idx, _ in enumerate(point_array[0])]


def is_equal_2DPoint(v1: Cartesian2DPoint | Polar2DPoint, v2: Cartesian2DPoint | Polar2DPoint):
    """Check whether two 2D coordinate points are equal.
    Consider numerical tolerance; cannot use v1 == v2 directly.

    Args:
        v1: first value to check
        v2: second value to check
    Returns:
        bool: whether they are equal.

    """
    return np.allclose(
        [v1[0].to_base_units().magnitude, v1[1].to_base_units().magnitude],
        [v2[0].to_base_units().magnitude, v2[1].to_base_units().magnitude],
    )


pydantic_config_dict_with_q = ConfigDict(
    str_to_lower=True,
    strict=True,
    extra='forbid',
    arbitrary_types_allowed=True,
)
pydantic_config_dict_with_q_case_insensitive = ConfigDict(
    str_to_lower=False,
    strict=True,
    extra='forbid',
    arbitrary_types_allowed=True,
)


class BaseModel_with_q(BaseModel):
    """BaseModel based on pint, supporting quantity conversion and serialization of numpy.array to list."""

    model_config = pydantic_config_dict_with_q

    def model_dump(self, **kwargs):
        """Mode = 'python' can convert np.ndarray to list, but does not support Quantity units.
        Mode = 'json' can serialize Quantity units, but does not convert np.ndarray to list.
        """
        data = super().model_dump(**kwargs)
        # Convert numpy.array to list
        for k, v in data.items():
            if isinstance(v, np.ndarray):
                data[k] = v.tolist()
        return data


class CONSTANTS:
    """Constants
    Attributes:
        vacuum_permeability (PermeabilityType): vacuum permeability in H/m.
        miu_0 (PermeabilityType): vacuum permeability in H/m.
    """

    vacuum_permeability: PermeabilityType = Q_(4 * np.pi * 1e-7, 'H/m')
    miu_0 = vacuum_permeability
