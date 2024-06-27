from dl_lib.tensor import Tensor


class Matrix(Tensor):
    def __init__(self, values, predefined=False, dim=None, element_type=None):
        super().__init__(values, predefined, dim, element_type)

    def __mul__(self, other):
        if self.dim[1] != other.dim[0]:
            raise Exception("Cannot multiply matrix as the dimensions do not match")

        new_dim = tuple([self.dim[0], other.dim[1]])
        new_values = []
        for i in range(self.dim[0]):
            row_values = self.get_row_values(i)
            for j in range(other.dim[1]):
                col_values = other.get_col_values(j)
                new_values.append(
                    sum(
                        [
                            row_value * col_value
                            for (row_value, col_value) in zip(row_values, col_values)
                        ]
                    )
                )

        return type(self)(new_values, True, new_dim, self.type)

    def get_row_values(self, row_index):
        start_index = row_index * self.dim[1]
        return self.values[start_index : start_index + self.dim[1]]

    def get_col_values(self, col_index):
        start_index = col_index
        return self.values[start_index :: self.dim[1]]

    def broadcast_vector(self, vector):
        if vector.dim[0] != self.dim[1]:
            raise Exception("Cannot broadcast vector as dimensions do not match")

        for i in range(0, len(self.values), vector.dim[0]):
            for j in range(vector.dim[0]):
                self.values[i + j] += vector.values[j]
