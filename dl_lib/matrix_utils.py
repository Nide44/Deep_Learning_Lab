import math

from dl_lib.matrix import Matrix
from dl_lib.vector import Vector


def hadamard_product(matrix1, matrix2):
    if matrix1.dim != matrix2.dim:
        raise Exception(
            "Cannot perform Hadamard multiplication as the matrices have different shapes"
        )

    return Matrix(
        list(map(lambda x, y: x * y, matrix1.values, matrix2.values)),
        True,
        matrix1.dim,
        matrix1.type,
    )


def create_identity_matrix(dim):
    if dim[0] != dim[1]:
        raise Exception("Identity matrices have to be square")
    new_values = []
    for i in range(dim[0]):
        for j in range(dim[1]):
            new_values.append(1) if i == j else new_values.append(0)
    return Matrix(new_values, True, dim, int)


def calculate_frobenius_norm(matrix):
    return math.sqrt(sum([math.pow(value, 2) for value in matrix.values]))


def create_diag_matrix_from_vector(vector, dim=None):
    if dim == None:
        dim = tuple(2 * [vector.dim[0]])
    vector_index = 0
    new_values = []
    for i in range(dim[0]):
        for j in range(dim[1]):
            if i == j:
                new_values.append(vector.values[vector_index])
                vector_index += 1
            else:
                new_values.append(0)
    return Matrix(new_values, True, dim, vector.type)


def check_diag_matrix(matrix):
    for i in range(matrix.dim[0]):
        for j in range(matrix.dim[1]):
            if i == j:
                if matrix.values[i * matrix.dim[1] + j] == 0:
                    return False
            else:
                if matrix.values[i * matrix.dim[1] + j] != 0:
                    return False
    return True


def invert_diag_matrix(matrix):
    if not check_diag_matrix(matrix):
        raise Exception("The provided matrix is not diagonal")

    inverted_original_vector = Vector(
        [1 / value for value in matrix.values if value != 0]
    )
    return create_diag_matrix_from_vector(inverted_original_vector)
