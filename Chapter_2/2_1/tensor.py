import math
from utils import Utils


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

    def transpose_dimension(
        self, current_dim, global_dim, switch_indices, new_values, element_indices
    ):
        if len(current_dim) == 1:
            for i in range(current_dim[0]):
                current_indices = element_indices[:]
                current_indices.append(i)
                new_indices = current_indices[:]
                new_indices[switch_indices[0]], new_indices[switch_indices[1]] = (
                    new_indices[switch_indices[1]],
                    new_indices[switch_indices[0]],
                )
                new_global_dim = list(global_dim)[:]
                new_global_dim[switch_indices[0]], new_global_dim[switch_indices[1]] = (
                    new_global_dim[switch_indices[1]],
                    new_global_dim[switch_indices[0]],
                )
                start_index_new = sum(
                    [
                        math.prod(new_global_dim[j + 1 :]) * new_index
                        for j, new_index in enumerate(new_indices[:-1])
                    ]
                )
                start_index_current = sum(
                    [
                        math.prod(global_dim[j + 1 :]) * current_index
                        for j, current_index in enumerate(current_indices[:-1])
                    ]
                )
                new_values[start_index_new + new_indices[-1]] = self.values[
                    start_index_current + current_indices[-1]
                ]

        else:
            lower_dim = tuple(list(current_dim)[1:])
            for j in range(current_dim[0]):
                next_element_indices = element_indices[:]
                next_element_indices.append(j)
                self.transpose_dimension(
                    lower_dim,
                    global_dim,
                    switch_indices,
                    new_values,
                    next_element_indices,
                )

        return new_values

    def transpose(self, indices=[0, 1]):
        # Check if indices are possible
        if not (
            len(indices) == len(list(filter(lambda x: 0 <= x < len(self.dim), indices)))
        ):
            raise Exception("Cannot transpose, indices out of range")

        # Make sure the indices are ordered
        indices = sorted(indices)

        # Transpose the values
        transposed_values = self.values[:]
        self.values = self.transpose_dimension(
            self.dim, self.dim, indices, transposed_values, []
        )

        # Update the dimensions
        dim_list = list(self.dim)
        dim_list[indices[0]], dim_list[indices[1]] = (
            dim_list[indices[1]],
            dim_list[indices[0]],
        )
        self.dim = tuple(dim_list)

    def mul_scalar(self, scalar):
        self.values = [value * scalar.values[0] for value in self.values]

    def add_scalar(self, scalar):
        self.values = [value + scalar.values[0] for value in self.values]
