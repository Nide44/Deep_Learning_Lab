from utils import Utils

class Scalar:
    def __init__(self, value):
        self.value = value
        self.dim = None
        self.type = type(value)

        is_valid_scalar = self.check_valid_scalar()
        if not is_valid_scalar:
            raise Exception("Scalar is not valid")

    def check_valid_scalar(self):
        return Utils.check_primitive_type(self.type)
    
    def transpose(self):
        pass
    
    def __str__(self):
        return Utils.stringify_tensor(self.dim, self.value)