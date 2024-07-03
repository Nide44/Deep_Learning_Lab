import pytest

from dl_lib.matrix import Matrix
from dl_lib.matrix_utils import (
    create_diag_matrix_from_vector,
    create_identity_matrix,
    hadamard_product,
    invert_diag_matrix,
)
from dl_lib.tensor_utils import transpose_tensor
from dl_lib.vector import Vector
from dl_lib.vector_utils import calculate_euclidean_norm


def test_create_diag_matrix_from_vector():
    vector = Vector([1, 2, 3, 4, 5])
    diag_matrix = create_diag_matrix_from_vector(vector)
    assert (
        diag_matrix.dim == (5, 5)
        and diag_matrix.values
        == [
            1,
            0,
            0,
            0,
            0,
            0,
            2,
            0,
            0,
            0,
            0,
            0,
            3,
            0,
            0,
            0,
            0,
            0,
            4,
            0,
            0,
            0,
            0,
            0,
            5,
        ]
        and diag_matrix.type == int
    )

    diag_matrix = create_diag_matrix_from_vector(vector, (6, 4))
    assert (
        diag_matrix.dim == (6, 4)
        and diag_matrix.values
        == [1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0]
        and diag_matrix.type == int
    )

    diag_matrix = create_diag_matrix_from_vector(vector, (2, 4))
    assert (
        diag_matrix.dim == (2, 4)
        and diag_matrix.values == [1, 0, 0, 0, 0, 2, 0, 0]
        and diag_matrix.type == int
    )


def test_diag_hadamard():
    vector1 = Vector([1, 2, 3, 4, 5])
    vector2 = Vector([6, 7, 8, 9, 10])
    assert create_diag_matrix_from_vector(vector1) * vector2 == hadamard_product(
        vector1, vector2
    )


def test_invert_diag_matrix():
    vector = Vector([1, 2, 3, 4, 5, 6])
    diag_matrix = create_diag_matrix_from_vector(vector)
    inverse_matrix = invert_diag_matrix(diag_matrix)
    assert (
        inverse_matrix.values
        == [
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1 / 2,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1 / 3,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1 / 4,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1 / 5,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1 / 6,
        ]
        and inverse_matrix.dim == (6, 6)
        and inverse_matrix.type == float
    )


def test_orthonormal_vectors():
    vector1 = Vector([0, 1])
    vector2 = Vector([1, 0])
    assert (
        (transpose_tensor(vector1) * vector2).values == [0]
        and calculate_euclidean_norm(vector1) == 1
        and calculate_euclidean_norm(vector2) == 1
    )


def test_orthogonal_matrix():
    matrix = Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    transposed_matrix = transpose_tensor(matrix)
    assert (
        matrix * transposed_matrix == transposed_matrix * matrix
        and matrix * transposed_matrix == create_identity_matrix(matrix.dim)
    )
