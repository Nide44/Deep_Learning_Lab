from dl_lib.tensor import Tensor


class Matrix(Tensor):
    def __init__(self, values, predefined=False, dim=None, element_type=None):
        super().__init__(values, predefined, dim, element_type)

    def broadcast_vector(self, vector):
        if vector.dim[0] != self.dim[1]:
            raise Exception("Cannot broadcast vector as dimensions do not match")

        for i in range(0, len(self.values), vector.dim[0]):
            for j in range(vector.dim[0]):
                self.values[i + j] += vector.values[j]
