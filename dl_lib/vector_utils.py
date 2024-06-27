from dl_lib.tensor_utils import transpose_tensor

def dot_product(vector1, vector2):
    if vector1.dim != vector2.dim:
        raise Exception(
            "Cannot calculate the dot product as the vectors have different shapes"
        )

    vector1_T = transpose_tensor(vector1)
    return vector1_T * vector2
