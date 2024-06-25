from utils import Utils

class Matrix:
    def __init__(self, values):
        self.values = values
        self.dim = (len(values), len(values[0]))
        self.type = type(values[0][0])

        is_valid_matrix = self.check_valid_matrix()
        if not is_valid_matrix:
            raise Exception("Matrix is not valid")
        
    def __add__(self, other):
        if self.dim != other.dim:
            raise Exception("Cannot add matrices as they have a different shape")
        
        return Matrix(Utils.add_tensor_values(self.dim, self.values, other.values))
        
    def check_valid_matrix(self):
        return Utils.check_primitive_type(self.type) and Utils.check_same_type(self.type, Utils.flatten_tensor(self.dim, self.values)) and Utils.check_complete_tensor(self.dim, self.values)
    
    def transpose(self):
        self.dim, self.values = Utils.transpose_tensor_values(self.dim, self.values)

    def scalar_mul(self, scalar):
        self.values = Utils.scalar_mul(self.dim, self.values, scalar.value)

    def scalar_add(self, scalar):
        self.values = Utils.scalar_add(self.dim, self.values, scalar.value)

    def vector_broadcast(self, vector):
        if not vector.vertical or vector.dim[0] != self.dim[1]:
            raise Exception("Cannot perform broadcasting as the dimensions do not match")
        
        for i in range(len(self.values)):
            for j in range(len(self.values[i])):
                self.values[i][j] += vector.values[j]
    
    def __str__(self):
        return Utils.stringify_tensor(self.dim, self.values)