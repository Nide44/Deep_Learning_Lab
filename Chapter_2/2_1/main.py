from scalar import Scalar
from vector import Vector
from matrix import Matrix
from tensor import Tensor

# Own implementation
scalar = Scalar(3)
# print(scalar)

vector = Vector([1, 2, 3])
# print(vector)
scalar_vector = Vector([3])
# print(scalar_vector)

matrix = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# print(matrix)
scalar_matrix = Matrix([[3]])
# print(scalar_matrix)
vector_matrix = Matrix([[1], [2], [3]])
# print(vector_matrix)

tensor_3d = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
# print(tensor_3d)
tensor_5d = Tensor(
    [
        [
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
            [[[9, 10], [11, 12]], [[13, 14], [15, 16]]],
        ],
        [
            [[[17, 18], [19, 20]], [[21, 22], [23, 24]]],
            [[[25, 26], [27, 28]], [[29, 30], [31, 32]]],
        ],
    ]
)
# print(tensor_5d)
scalar_tensor = Tensor([[[3]]])
# print(scalar_tensor)
vector_tensor = Tensor([[[1], [2], [3]]])
# print(vector_tensor)
matrix_tensor = Tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
# print(matrix_tensor)

# print(scalar)
scalar.transpose()
# print(scalar)
# print(vector, vector.vertical)
vector.transpose()
# print(vector, vector.vertical)
# print(matrix)
matrix.transpose()
# print(matrix)
# print(tensor_3d)
tensor_3d.transpose([0,2])
# print(tensor_3d)
# print(tensor_5d)
tensor_5d.transpose([2,4])
# print(tensor_5d)
