import pytest

from dl_lib.matrix_utils import calculate_frobenius_norm
from dl_lib.vector_utils import (
    calculate_euclidean_norm,
    calculate_l1_norm,
    calculate_max_norm,
    calculate_squared_l2_norm,
    calculate_lp_norm,
)
from dl_lib.vector import Vector
from dl_lib.matrix import Matrix


def test_calc_lp_norm():
    vector = Vector([1, 2, 3, 4, 5])
    norm = calculate_lp_norm(vector, 3)
    assert pytest.approx(norm, abs=0.01) == 6.08


def test_calc_euclidean_norm():
    vector = Vector([1, 2, 3, 4, 5])
    norm = calculate_euclidean_norm(vector)
    assert pytest.approx(norm, abs=0.01) == 7.41


def test_calc_squared_l2_norm():
    vector = Vector([1, 2, 3, 4, 5])
    norm = calculate_squared_l2_norm(vector)
    assert norm == 55


def test_calc_l1_norm():
    vector = Vector([1, 2, 3, 4, 5])
    norm = calculate_l1_norm(vector)
    assert norm == 15


def test_calc_max_norm():
    vector = Vector([1, 2, 3, 4, 5])
    norm = calculate_max_norm(vector)
    assert norm == 5


def test_calc_frobenius_norm():
    matrix = Matrix([[1, 2], [3, 4]])
    norm = calculate_frobenius_norm(matrix)
    assert pytest.approx(norm, abs=0.01) == 5.48
