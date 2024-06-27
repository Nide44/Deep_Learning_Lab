from dl_lib.scalar import Scalar
from dl_lib.vector import Vector
from dl_lib.matrix import Matrix
from dl_lib.tensor import Tensor


def test_create_scalar():
    scalar = Scalar(3)
    assert scalar.values == [3] and scalar.dim == (1, 1) and scalar.type == int


def test_create_vector():
    vector = Vector([1, 2, 3])
    assert vector.values == [1, 2, 3] and vector.dim == (3, 1) and vector.type == int


def test_create_scalar_vector():
    scalar_vector = Vector([3])
    assert (
        scalar_vector.values == [3]
        and scalar_vector.dim == (1, 1)
        and scalar_vector.type == int
    )


def test_create_matrix():
    matrix = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert (
        matrix.values == [1, 2, 3, 4, 5, 6, 7, 8, 9]
        and matrix.dim == (3, 3)
        and matrix.type == int
    )


def test_create_scalar_matrix():
    scalar_matrix = Matrix([[3]])
    assert (
        scalar_matrix.values == [3]
        and scalar_matrix.dim == (1, 1)
        and scalar_matrix.type == int
    )


def test_create_vector_matrix():
    vector_matrix = Matrix([[1], [2], [3]])
    assert (
        vector_matrix.values == [1, 2, 3]
        and vector_matrix.dim == (3, 1)
        and vector_matrix.type == int
    )


def test_create_tensor1():
    tensor_3d = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    assert (
        tensor_3d.values == [1, 2, 3, 4, 5, 6, 7, 8]
        and tensor_3d.dim == (2, 2, 2)
        and tensor_3d.type == int
    )


def test_create_tensor2():
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
    assert (
        tensor_5d.values == [i + 1 for i in range(32)]
        and tensor_5d.dim == (2, 2, 2, 2, 2)
        and tensor_5d.type == int
    )


def test_create_scalar_tensor():
    scalar_tensor = Tensor([[[3]]])
    assert (
        scalar_tensor.values == [3]
        and scalar_tensor.dim == (1, 1, 1)
        and scalar_tensor.type == int
    )


def test_create_vector_tensor():
    vector_tensor = Tensor([[[1], [2], [3]]])
    assert (
        vector_tensor.values == [1, 2, 3]
        and vector_tensor.dim == (1, 3, 1)
        and vector_tensor.type == int
    )


def test_create_matrix_tensor():
    matrix_tensor = Tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
    assert (
        matrix_tensor.values == [i + 1 for i in range(9)]
        and matrix_tensor.dim == (1, 3, 3)
        and matrix_tensor.type == int
    )


def test_scalar_transpose():
    scalar = Scalar(3)
    scalar.transpose()
    assert scalar.values == [3] and scalar.dim == (1, 1) and scalar.type == int


def test_vector_transpose():
    vector = Vector([1, 2, 3])
    vector.transpose()
    assert vector.values == [1, 2, 3] and vector.dim == (1, 3) and vector.type == int


def test_matrix_transpose():
    matrix = Matrix([[1, 2, 3], [4, 5, 6]])
    matrix.transpose()
    assert (
        matrix.values == [1, 4, 2, 5, 3, 6]
        and matrix.dim == (3, 2)
        and matrix.type == int
    )


def test_tensor_transpose1():
    tensor_3d = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    tensor_3d.transpose([1, 2])
    assert (
        tensor_3d.values == [1, 3, 2, 4, 5, 7, 6, 8]
        and tensor_3d.dim == (2, 2, 2)
        and tensor_3d.type == int
    )


def test_tensor_transpose2():
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
    tensor_5d.transpose([3, 4])
    assert (
        tensor_5d.values
        == [
            1,
            3,
            2,
            4,
            5,
            7,
            6,
            8,
            9,
            11,
            10,
            12,
            13,
            15,
            14,
            16,
            17,
            19,
            18,
            20,
            21,
            23,
            22,
            24,
            25,
            27,
            26,
            28,
            29,
            31,
            30,
            32,
        ]
        and tensor_5d.dim == (2, 2, 2, 2, 2)
        and tensor_5d.type == int
    )

    tensor_5d.transpose([3, 4])
    assert (
        tensor_5d.values == [i + 1 for i in range(32)]
        and tensor_5d.dim == (2, 2, 2, 2, 2)
        and tensor_5d.type == int
    )

    tensor_5d.transpose([0, 1])
    assert (
        tensor_5d.values
        == [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
        ]
        and tensor_5d.dim == (2, 2, 2, 2, 2)
        and tensor_5d.type == int
    )


def test_add_scalar_scalar():
    scalar1 = Scalar(3)
    scalar2 = Scalar(2)
    scalar1.add_scalar(scalar2)
    assert scalar1.values == [5] and scalar1.dim == (1, 1) and scalar1.type == int


def test_add_scalar_vector():
    scalar = Scalar(3)
    vector = Vector([1, 2, 3])
    vector.add_scalar(scalar)
    assert vector.values == [4, 5, 6] and vector.dim == (3, 1) and vector.type == int


def test_add_scalar_matrix():
    scalar = Scalar(3)
    matrix = Matrix([[1, 2, 3], [4, 5, 6]])
    matrix.add_scalar(scalar)
    assert (
        matrix.values == [4, 5, 6, 7, 8, 9]
        and matrix.dim == (2, 3)
        and matrix.type == int
    )


def test_add_scalar_tensor():
    scalar = Scalar(3)
    tensor_3d = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    tensor_3d.add_scalar(scalar)
    assert (
        tensor_3d.values == [4, 5, 6, 7, 8, 9, 10, 11]
        and tensor_3d.dim == (2, 2, 2)
        and tensor_3d.type == int
    )


def test_add_scalars():
    scalar1 = Scalar(3)
    scalar2 = Scalar(2)
    sum_scalar = scalar1 + scalar2
    assert (
        sum_scalar.values == [5] and sum_scalar.dim == (1, 1) and sum_scalar.type == int
    )


def test_add_vectors():
    vector1 = Vector([1, 2, 3])
    vector2 = Vector([4, 5, 6])
    sum_vector = vector1 + vector2
    assert (
        sum_vector.values == [5, 7, 9]
        and sum_vector.dim == (3, 1)
        and sum_vector.type == int
    )


def test_add_matrices():
    matrix1 = Matrix([[1, 2, 3], [4, 5, 6]])
    matrix2 = Matrix([[7, 8, 9], [10, 11, 12]])
    sum_matrix = matrix1 + matrix2
    assert (
        sum_matrix.values == [8, 10, 12, 14, 16, 18]
        and sum_matrix.dim == (2, 3)
        and sum_matrix.type == int
    )


def test_add_tensors():
    tensor_3d1 = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    tensor_3d2 = Tensor([[[9, 10], [11, 12]], [[13, 14], [15, 16]]])
    sum_tensor = tensor_3d1 + tensor_3d2
    assert (
        sum_tensor.values == [10, 12, 14, 16, 18, 20, 22, 24]
        and sum_tensor.dim == (2, 2, 2)
        and sum_tensor.type == int
    )


def test_mul_scalar_scalar():
    scalar1 = Scalar(3)
    scalar2 = Scalar(2)
    scalar1.mul_scalar(scalar2)
    assert scalar1.values == [6] and scalar1.dim == (1, 1) and scalar1.type == int


def test_mul_scalar_vector():
    scalar = Scalar(3)
    vector = Vector([1, 2, 3])
    vector.mul_scalar(scalar)
    assert vector.values == [3, 6, 9] and vector.dim == (3, 1) and vector.type == int


def test_mul_scalar_matrix():
    scalar = Scalar(3)
    matrix = Matrix([[1, 2, 3], [4, 5, 6]])
    matrix.mul_scalar(scalar)
    assert (
        matrix.values == [3, 6, 9, 12, 15, 18]
        and matrix.dim == (2, 3)
        and matrix.type == int
    )


def test_mul_scalar_tensor():
    scalar = Scalar(3)
    tensor_3d = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    tensor_3d.mul_scalar(scalar)
    assert (
        tensor_3d.values == [3, 6, 9, 12, 15, 18, 21, 24]
        and tensor_3d.dim == (2, 2, 2)
        and tensor_3d.type == int
    )


def test_broadcast_vector_matrix():
    vector = Vector([1, 2, 3])
    matrix = Matrix([[1, 2, 3], [4, 5, 6]])
    matrix.broadcast_vector(vector)
    assert (
        matrix.values == [2, 4, 6, 5, 7, 9]
        and matrix.dim == (2, 3)
        and matrix.type == int
    )
