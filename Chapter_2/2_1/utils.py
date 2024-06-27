import math

from globals import PRIMITIVE_TYPES


class Utils:
    @staticmethod
    def check_primitive_type(element_type):
        return element_type in PRIMITIVE_TYPES

    @staticmethod
    def check_same_type(element_type, values):
        return len(values) == len(
            list(filter(lambda x: isinstance(x, element_type), values))
        )

    @staticmethod
    def check_complete_tensor(dim, values):
        return len(values) == math.prod(dim)

    @staticmethod
    def upgrade_to_2_dims(values):
        if not type(values) == list:
            return [[values]]
        elif type(values) == list and type(values[0]) != list:
            return [[value] for value in values]
        else:
            return values
