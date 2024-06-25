from utils import Utils

class Matrix:
    def __init__(self, values):
        self.values = values
        self.dim = (len(values), len(values[0]))
        self.type = type(values[0][0])

        is_valid_matrix = self.check_valid_matrix()
        if not is_valid_matrix:
            raise Exception("Matrix is not valid")
        
    def check_valid_matrix(self):
        return Utils.check_primitive_type(self.type) and Utils.check_same_type(self.type, Utils.flatten_tensor(self.dim, self.values)) and Utils.check_complete_tensor(self.dim, self.values)
    
    def __str__(self):
        return Utils.stringify_tensor(self.dim, self.values)