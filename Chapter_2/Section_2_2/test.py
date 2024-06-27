import pytest

from dl_lib.vector import Vector
from dl_lib.matrix import Matrix
from dl_lib.matrix_utils import hadamard_product
from dl_lib.vector_utils import dot_product
from dl_lib.tensor_utils import transpose_tensor


def test_multiply_matrix():
    matrix1 = Matrix([[1, 2, 3], [4, 5, 6]])
    matrix2 = Matrix([[7, 8, 9], [10, 11, 12], [13, 14, 15]])
    mul_matrix = matrix1 * matrix2
    assert (
        mul_matrix.values == [66, 72, 78, 156, 171, 186]
        and mul_matrix.dim == (2, 3)
        and mul_matrix.type == int
    )

    with pytest.raises(Exception):
        mul_matrix = matrix2 * matrix1


def test_hadamard_matrix():
    matrix1 = Matrix([[1, 2, 3], [4, 5, 6]])
    matrix2 = Matrix([[7, 8, 9], [10, 11, 12]])
    hadamard_matrix = hadamard_product(matrix1, matrix2)
    assert (
        hadamard_matrix.values == [7, 16, 27, 40, 55, 72]
        and hadamard_matrix.dim == (2, 3)
        and hadamard_matrix.type == int
    )


def test_dot_product():
    vector1 = Vector([1, 2, 3, 4])
    vector2 = Vector([5, 6, 7, 8])
    dot_vector = dot_product(vector1, vector2)
    assert (
        dot_vector.values == [70]
        and dot_vector.dim == (1, 1)
        and dot_vector.type == int
    )


def test_matrix_mul_distributive():
    matrix1 = Matrix([[1, 2, 3], [4, 5, 6]])
    matrix2 = Matrix([[7, 8, 9], [10, 11, 12], [13, 14, 15]])
    matrix3 = Matrix([[16, 17, 18], [19, 20, 21], [22, 23, 24]])
    assert matrix1 * (matrix2 + matrix3) == matrix1 * matrix2 + matrix1 * matrix3


def test_matrix_mul_associative():
    matrix1 = Matrix([[1, 2, 3], [4, 5, 6]])
    matrix2 = Matrix([[7, 8, 9], [10, 11, 12], [13, 14, 15]])
    matrix3 = Matrix([[16, 17, 18], [19, 20, 21], [22, 23, 24]])
    assert matrix1 * (matrix2 * matrix3) == (matrix1 * matrix2) * matrix3


def test_matrix_mul_not_commutative():
    matrix1 = Matrix([[1, 2, 3], [4, 5, 6]])
    matrix2 = Matrix([[7, 8, 9], [10, 11, 12], [13, 14, 15]])
    with pytest.raises(Exception):
        matrix1 * matrix2 == matrix2 * matrix1


def test_dot_prod_commutative():
    vector1 = Vector([1, 2, 3, 4])
    vector2 = Vector([5, 6, 7, 8])
    assert dot_product(vector1, vector2) == dot_product(vector2, vector1)


def test_transpose_matrix_mul():
    matrix1 = Matrix([[1, 2, 3], [4, 5, 6]])
    matrix2 = Matrix([[7, 8, 9], [10, 11, 12], [13, 14, 15]])
    assert transpose_tensor(matrix1 * matrix2) == transpose_tensor(matrix2) * transpose_tensor(matrix1)