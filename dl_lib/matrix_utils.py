from dl_lib.matrix import Matrix


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
