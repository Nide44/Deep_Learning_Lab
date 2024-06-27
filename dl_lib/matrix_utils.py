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


def create_identity_matrix(dim):
    if dim[0] != dim[1]:
        raise Exception("Identity matrices have to be square")
    new_values = []
    for i in range(dim[0]):
        for j in range(dim[1]):
            new_values.append(1) if i == j else new_values.append(0)
    return Matrix(new_values, True, dim, int)
