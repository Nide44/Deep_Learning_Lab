from tensor import Tensor


class Vector(Tensor):
    def __init__(self, values, predefined=False, dim=None, element_type=None):
        super().__init__(values, predefined, dim, element_type)
