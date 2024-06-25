from globals import PRIMITIVE_TYPES

class Utils:
    @staticmethod
    def check_primitive_type(element_type):
        return element_type in PRIMITIVE_TYPES
        
    @staticmethod
    def check_same_type(element_type, values):
        return len(values) == len(list(filter(lambda x: isinstance(x, element_type), values)))
    
    @staticmethod
    def stringify_tensor(dim, values, vertical=False):
        if dim == None:
            return str(values)
        elif len(dim) == 1:
            lower_dim = None
            if vertical:
                values_string = "\n ".join([Utils.stringify_tensor(lower_dim, value) for value in values])
            else:
                values_string = " ".join([Utils.stringify_tensor(lower_dim, value) for value in values])
            return values_string if len(values) == 1 else "[" + values_string + "]"
        else:
            lower_dim = tuple(list(dim)[1:])
            new_lines = (len(dim) - 1) * "\n"
            values_string = (new_lines + " ").join([Utils.stringify_tensor(lower_dim, value) for value in values])
            return values_string if len(values) == 1 else "[" + values_string + "]"
        
    @staticmethod
    def flatten_tensor(dim, values):
        if len(dim) == 1:
            return values
        else:
            lower_dim = tuple(list(dim)[1:])
            return sum([Utils.flatten_tensor(lower_dim, sublist) for sublist in values], start=[])
        
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
    def transpose_tensor(dim, values, indices=[0,1]):
        # Check if indices are possible
        if not (len(indices) == len(list(filter(lambda x: 0 <= x <= len(dim), indices)))):
            raise Exception("Cannot transpose, indices out of range")
