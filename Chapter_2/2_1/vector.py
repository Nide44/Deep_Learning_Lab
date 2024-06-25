from globals import PRIMITIVE_TYPES
from utils import Utils

class Vector:
    def __init__(self, values):
        self.values = values
        self.dim = (len(values),)
        self.type = type(values[0])
        self.vertical = True

        is_valid_vector = self.check_valid_vector()
        if not is_valid_vector:
            raise Exception("Vector is not valid")

    def check_valid_vector(self):
        return Utils.check_primitive_type(self.type) and Utils.check_same_type(self.type, self.values)
    
    def __str__(self):
        return Utils.stringify_tensor(self.dim, self.values, vertical=True)