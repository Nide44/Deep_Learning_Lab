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

scalar1 = Scalar(3)
scalar2 = Scalar(2)
sum_scalar = scalar1 + scalar2
# print(sum_scalar)

vector1 = Vector([1, 2, 3])
vector2 = Vector([4, 5, 6])
sum_vector = vector1 + vector2
# print(sum_vector)

matrix1 = Matrix([[1, 2, 3], [4, 5, 6]])
matrix2 = Matrix([[7, 8, 9], [10, 11, 12]])
sum_matrix = matrix1 + matrix2
# print(sum_matrix)

tensor_3d1 = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
tensor_3d2 = Tensor([[[9, 10], [11, 12]], [[13, 14], [15, 16]]])
sum_tensor = tensor_3d1 + tensor_3d2
# print(sum_tensor)

# print(vector1)
vector1.scalar_mul(scalar1)
# print(vector1)

# print(matrix1)
matrix1.scalar_mul(scalar1)
# print(matrix1)

# print(tensor_3d1)
tensor_3d1.scalar_mul(scalar1)
# print(tensor_3d1)

# print(vector1)
vector1.scalar_add(scalar1)
# print(vector1)

# print(matrix1)
matrix1.scalar_add(scalar1)
# print(matrix1)

# print(tensor_3d1)
tensor_3d1.scalar_add(scalar1)
# print(tensor_3d1)

print(matrix1)
matrix1.vector_broadcast(vector1)
print(matrix1)