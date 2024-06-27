import pytest

from dl_lib.matrix_utils import create_identity_matrix
from dl_lib.vector import Vector


def test_create_identity_matrix():
    i_matrix = create_identity_matrix((4, 4))
    assert (
        i_matrix.values
        == [
            1,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            1,
        ]
        and i_matrix.dim == (4, 4)
        and i_matrix.type == int
    )

    with pytest.raises(Exception):
        i_matrix = create_identity_matrix((2, 3))


def test_identity_mul_vector():
    vector = Vector([1, 2, 3, 4])
    i_matrix = create_identity_matrix((4, 4))
    assert i_matrix * vector == vector
