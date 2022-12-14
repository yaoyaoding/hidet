from typing import Any, Sequence
from hidet.ir.type import DataType
from .floats import float32, float16


class VectorType(DataType):
    def __init__(self, lane_type: DataType, num_lanes: int):
        name = '{}x{}'.format(lane_type.name, num_lanes)
        short_name = '{}x{}'.format(lane_type.short_name, num_lanes)
        nbytes = lane_type.nbytes * num_lanes
        super().__init__(name, short_name, nbytes)
        self._num_lanes: int = num_lanes
        self._lane_type: DataType = lane_type

        if lane_type.is_vector():
            raise ValueError('Cannot create a vector type of vectors')

    def is_float(self) -> bool:
        return False

    def is_integer(self) -> bool:
        return False

    def is_vector(self) -> bool:
        return True

    @property
    def num_lanes(self) -> int:
        return self._num_lanes

    @property
    def lane_type(self) -> DataType:
        return self._lane_type

    def constant(self, value: Sequence[Any]):
        from hidet.ir.expr import Constant

        value = [self.lane_type.constant(v) for v in value]
        if len(value) != self.num_lanes:
            raise ValueError('Invalid vector constant, expect {} elements, got {}'.format(self.num_lanes, len(value)))
        return Constant(value, self)

    def one(self):
        return self.constant([self.lane_type.one()] * self.num_lanes)

    def zero(self):
        return self.constant([self.lane_type.zero()] * self.num_lanes)

    def min_value(self):
        return self.constant([self.lane_type.min_value()] * self.num_lanes)

    def max_value(self):
        return self.constant([self.lane_type.max_value()] * self.num_lanes)


float32x4 = VectorType(float32, 4)
float16x2 = VectorType(float16, 2)

f32x4 = float32x4
f16x2 = float16x2