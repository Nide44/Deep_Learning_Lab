import math

from dl_lib.tensor_utils import transpose_tensor


def dot_product(vector1, vector2):
    if vector1.dim != vector2.dim:
        raise Exception(
            "Cannot calculate the dot product as the vectors have different shapes"
        )

    vector1_T = transpose_tensor(vector1)
    return vector1_T * vector2


def calculate_lp_norm(vector, p):
    if p < 1:
        raise Exception("P needs to be at least 1")

    return math.pow(sum([math.pow(abs(value), p) for value in vector.values]), 1 / p)


def calculate_euclidean_norm(vector):
    return calculate_lp_norm(vector, 2)


def calculate_squared_l2_norm(vector):
    return math.pow(calculate_euclidean_norm(vector), 2)


def calculate_l1_norm(vector):
    return calculate_lp_norm(vector, 1)


def calculate_max_norm(vector):
    return max([abs(value) for value in vector.values])
