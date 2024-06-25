from utils import Utils

class Tensor:
    def __init__(self, values):
        self.values = values
        self.dim = self.define_dimension(values)
        self.type = self.define_type(self.dim, values)

        is_valid_tensor = self.check_valid_tensor()
        if not is_valid_tensor:
            raise Exception("Tensor is not valid")
        
    def __add__(self, other):
        if self.dim != other.dim:
            raise Exception("Cannot add tensors as they have a different shape")
        
        return Tensor(Utils.add_tensor_values(self.dim, self.values, other.values))
        
    def define_dimension(self, values):
        if isinstance(values[0], list):
            lower_dim = self.define_dimension(values[0])
            return tuple([len(values)] + list(lower_dim))
        else:
            return (len(values),)
        
    def define_type(self, dim, values):
        if len(dim) == 1:
            return type(values[0])
        else:
            lower_dim = tuple(list(dim)[1:])
            return self.define_type(lower_dim, values[0])
        
    def check_valid_tensor(self):
        return Utils.check_primitive_type(self.type) and Utils.check_same_type(self.type, Utils.flatten_tensor(self.dim, self.values)) and Utils.check_complete_tensor(self.dim, self.values)
    
    def transpose(self, indices):
        self.dim, self.values = Utils.transpose_tensor_values(self.dim, self.values, indices)

    def scalar_mul(self, scalar):
        self.values = Utils.scalar_mul(self.dim, self.values, scalar.value)

    def scalar_add(self, scalar):
        self.values = Utils.scalar_add(self.dim, self.values, scalar.value)
    
    def __str__(self):
        return Utils.stringify_tensor(self.dim, self.values)