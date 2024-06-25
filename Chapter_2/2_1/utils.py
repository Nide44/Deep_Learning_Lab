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
    def stringify_tensor(dim, values, vertical=False):
        if dim == None:
            return str(values)
        elif len(dim) == 1:
            lower_dim = None
            if vertical:
                values_string = "\n ".join(
                    [Utils.stringify_tensor(lower_dim, value) for value in values]
                )
            else:
                values_string = " ".join(
                    [Utils.stringify_tensor(lower_dim, value) for value in values]
                )
            return values_string if len(values) == 1 else "[" + values_string + "]"
        else:
            lower_dim = tuple(list(dim)[1:])
            new_lines = (len(dim) - 1) * "\n"
            values_string = (new_lines + " ").join(
                [Utils.stringify_tensor(lower_dim, value) for value in values]
            )
            return values_string if len(values) == 1 else "[" + values_string + "]"

    @staticmethod
    def flatten_tensor(dim, values):
        if len(dim) == 1:
            return values
        else:
            lower_dim = tuple(list(dim)[1:])
            return sum(
                [Utils.flatten_tensor(lower_dim, sublist) for sublist in values],
                start=[],
            )

    @staticmethod
    def check_complete_tensor(dim, values):
        if len(dim) == 1:
            return dim[0] == len(values)
        else:
            lower_dim = tuple(list(dim)[1:])
            for subtensor in values:
                if not Utils.check_complete_tensor(lower_dim, subtensor):
                    return False
            return True

    @staticmethod
    def create_zero_tensor(dim):
        if len(dim) == 1:
            return dim[0] * [0]
        else:
            zero_tensor = []
            lower_dim = tuple(list(dim)[1:])
            for i in range(dim[0]):
                zero_tensor.append(Utils.create_zero_tensor(lower_dim))
            return zero_tensor
        
    @staticmethod
    def get_tensor_element_by_list_of_indices(tensor, indices):
        if len(indices) == 0:
            return tensor
        else:
            return Utils.get_tensor_element_by_list_of_indices(tensor[indices[0]], indices[1:])

    @staticmethod
    def set_tensor_with_list_of_indices(tensor, indices, value):
        if len(indices) == 1:
            tensor[indices[0]] = value
        else:
            Utils.set_tensor_with_list_of_indices(tensor[indices[0]], indices[1:], value)

    @staticmethod
    def fill_transposed_tensor(switch_indices, dim, values, tensor, current_indices, new_indices, current_level):
        if dim == None:
            value = Utils.get_tensor_element_by_list_of_indices(values, current_indices)
            Utils.set_tensor_with_list_of_indices(tensor, new_indices, value)
        else:
            if len(dim) == 1:
                lower_dim = None
            else:
                lower_dim = tuple(list(dim)[1:])
            for j in range(dim[0]):
                next_new_indices = new_indices[:]

                if current_level == switch_indices[1]:
                    buffer_value = next_new_indices[switch_indices[0]]
                    next_new_indices[switch_indices[0]] = j
                    next_new_indices.append(buffer_value)
                else:
                    next_new_indices.append(j)

                next_current_indices = current_indices[:]
                next_current_indices.append(j)
                Utils.fill_transposed_tensor(switch_indices, lower_dim, values, tensor, next_current_indices, next_new_indices, current_level + 1)

    @staticmethod
    def transpose_tensor_values(dim, values, indices=[0, 1]):
        # Check if indices are possible
        if not (
            len(indices) == len(list(filter(lambda x: 0 <= x < len(dim), indices)))
        ):
            raise Exception("Cannot transpose, indices out of range")

        # Check that 2 different indices are given
        if indices[0] == indices[1]:
            raise Exception("Cannot transpose an axis with itself")
        
        # Ensure the switching axes are ascending
        indices = sorted(indices)

        # Create the new dimension tuple
        new_dim_list = list(dim)[:]
        new_dim_list[indices[0]] = dim[indices[1]]
        new_dim_list[indices[1]] = dim[indices[0]]
        new_dim = tuple(new_dim_list)

        # Create an empty/zero tensor
        new_tensor = Utils.create_zero_tensor(new_dim)

        # Fill the new tensor
        Utils.fill_transposed_tensor(indices, dim, values, new_tensor, [], [], 0)

        return new_dim, new_tensor
    
    @staticmethod
    def fill_addition_tensor(dim, values1, values2, tensor, current_indices):
        if dim == None:
            value1 = Utils.get_tensor_element_by_list_of_indices(values1, current_indices)
            value2 = Utils.get_tensor_element_by_list_of_indices(values2, current_indices)
            Utils.set_tensor_with_list_of_indices(tensor, current_indices, value1 + value2)
        else:
            if len(dim) == 1:
                lower_dim = None
            else:
                lower_dim = tuple(list(dim)[1:])
            for j in range(dim[0]):
                next_current_indices = current_indices[:]
                next_current_indices.append(j)
                Utils.fill_addition_tensor(lower_dim, values1, values2, tensor, next_current_indices)

    @staticmethod
    def add_tensor_values(dim, values1, values2):
        new_tensor = Utils.create_zero_tensor(dim)
        Utils.fill_addition_tensor(dim, values1, values2, new_tensor, [])
        return new_tensor
    
    @staticmethod
    def fill_scalar_mul_tensor(dim, tensor_values, scalar_value, tensor, current_indices):
        if dim == None:
            value_tensor = Utils.get_tensor_element_by_list_of_indices(tensor_values, current_indices)
            Utils.set_tensor_with_list_of_indices(tensor, current_indices, value_tensor * scalar_value)
        else:
            if len(dim) == 1:
                lower_dim = None
            else:
                lower_dim = tuple(list(dim)[1:])
            for j in range(dim[0]):
                next_current_indices = current_indices[:]
                next_current_indices.append(j)
                Utils.fill_scalar_mul_tensor(lower_dim, tensor_values, scalar_value, tensor, next_current_indices)
    
    @staticmethod
    def scalar_mul(dim, tensor_values, scalar_value):
        new_tensor = Utils.create_zero_tensor(dim)
        Utils.fill_scalar_mul_tensor(dim, tensor_values, scalar_value, new_tensor, [])
        return new_tensor
    
    @staticmethod
    def fill_scalar_add_tensor(dim, tensor_values, scalar_value, tensor, current_indices):
        if dim == None:
            value_tensor = Utils.get_tensor_element_by_list_of_indices(tensor_values, current_indices)
            Utils.set_tensor_with_list_of_indices(tensor, current_indices, value_tensor + scalar_value)
        else:
            if len(dim) == 1:
                lower_dim = None
            else:
                lower_dim = tuple(list(dim)[1:])
            for j in range(dim[0]):
                next_current_indices = current_indices[:]
                next_current_indices.append(j)
                Utils.fill_scalar_add_tensor(lower_dim, tensor_values, scalar_value, tensor, next_current_indices)
    
    @staticmethod
    def scalar_add(dim, tensor_values, scalar_value):
        new_tensor = Utils.create_zero_tensor(dim)
        Utils.fill_scalar_add_tensor(dim, tensor_values, scalar_value, new_tensor, [])
        return new_tensor