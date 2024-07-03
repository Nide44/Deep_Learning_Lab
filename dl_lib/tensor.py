import math
from dl_lib.utils import Utils
from dl_lib.tensor_utils import cast_values_same_type

class Tensor:
    def __init__(self, values, predefined=False, dim=None, element_type=None):
        if not predefined:
            values = Utils.upgrade_to_2_dims(values)
            self.dim = self.define_dimension(values)
            self.values = self.flatten_tensor_values(self.dim, values)
            self.type = self.define_type(self.dim, values)
        else:
            self.dim = dim
            self.values = values
            self.type = element_type

        self.values = cast_values_same_type(self.values)

        is_valid_tensor = self.check_valid_tensor()
        if not is_valid_tensor:
            raise Exception("Tensor is not valid")

    def __str__(self):
        return self.stringify_tensor(self.dim, self.dim, [])

    def __add__(self, other):
        if self.dim != other.dim:
            raise Exception("Cannot add tensors as they have a different shape")

        new_values = list(map(lambda x, y: x + y, self.values, other.values))
        return type(self)(new_values, True, self.dim, self.type)

    def __eq__(self, other):
        return (
            self.dim == other.dim
            and self.values == other.values
            and self.type == other.type
        )

    def stringify_tensor(self, current_dim, global_dim, current_indices):
        if len(current_dim) == 1:
            start_index = sum(
                [
                    math.prod(global_dim[i + 1 :]) * current_index
                    for i, current_index in enumerate(current_indices)
                ]
            )
            values_string = " ".join(
                [
                    str(value)
                    for value in self.values[start_index : start_index + current_dim[0]]
                ]
            )
            return values_string if current_dim[0] == 1 else "[" + values_string + "]"
        else:
            lower_dim = tuple(list(current_dim)[1:])
            sub_values_strings = []
            for i in range(current_dim[0]):
                next_current_indices = current_indices[:]
                next_current_indices.append(i)
                sub_values_strings.append(
                    self.stringify_tensor(lower_dim, global_dim, next_current_indices)
                )
            values_string = ((len(current_dim) - 1) * "\n").join(sub_values_strings)
            return values_string if current_dim[0] == 1 else "[" + values_string + "]"

    def define_dimension(self, values):
        if isinstance(values[0], list):
            lower_dim = self.define_dimension(values[0])
            return tuple([len(values)] + list(lower_dim))
        else:
            return (len(values),)

    def flatten_tensor_values(self, dim, values):
        if len(dim) == 1:
            return values
        else:
            lower_dim = tuple(list(dim)[1:])
            return sum(
                [self.flatten_tensor_values(lower_dim, sublist) for sublist in values],
                start=[],
            )

    def define_type(self, dim, values):
        if len(dim) == 1:
            return type(values[0])
        else:
            lower_dim = tuple(list(dim)[1:])
            return self.define_type(lower_dim, values[0])

    def check_valid_tensor(self):
        return (
            Utils.check_primitive_type(self.type)
            and Utils.check_same_type(self.type, self.values)
            and Utils.check_complete_tensor(self.dim, self.values)
        )
