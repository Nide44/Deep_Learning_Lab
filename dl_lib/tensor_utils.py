import math


def transpose_dimension(
    tensor_values, current_dim, global_dim, switch_indices, new_values, element_indices
):
    if len(current_dim) == 1:
        for i in range(current_dim[0]):
            current_indices = element_indices[:]
            current_indices.append(i)
            new_indices = current_indices[:]
            new_indices[switch_indices[0]], new_indices[switch_indices[1]] = (
                new_indices[switch_indices[1]],
                new_indices[switch_indices[0]],
            )
            new_global_dim = list(global_dim)[:]
            new_global_dim[switch_indices[0]], new_global_dim[switch_indices[1]] = (
                new_global_dim[switch_indices[1]],
                new_global_dim[switch_indices[0]],
            )
            start_index_new = sum(
                [
                    math.prod(new_global_dim[j + 1 :]) * new_index
                    for j, new_index in enumerate(new_indices[:-1])
                ]
            )
            start_index_current = sum(
                [
                    math.prod(global_dim[j + 1 :]) * current_index
                    for j, current_index in enumerate(current_indices[:-1])
                ]
            )
            new_values[start_index_new + new_indices[-1]] = tensor_values[
                start_index_current + current_indices[-1]
            ]

    else:
        lower_dim = tuple(list(current_dim)[1:])
        for j in range(current_dim[0]):
            next_element_indices = element_indices[:]
            next_element_indices.append(j)
            transpose_dimension(
                tensor_values,
                lower_dim,
                global_dim,
                switch_indices,
                new_values,
                next_element_indices,
            )

    return new_values


def transpose_tensor(tensor, indices=[0, 1]):
    # Check if indices are possible
    if not (
        len(indices) == len(list(filter(lambda x: 0 <= x < len(tensor.dim), indices)))
    ):
        raise Exception("Cannot transpose, indices out of range")

    # Make sure the indices are ordered
    indices = sorted(indices)

    # Transpose the values
    transposed_values = tensor.values[:]
    new_values = transpose_dimension(
        tensor.values, tensor.dim, tensor.dim, indices, transposed_values, []
    )

    # Update the dimensions
    dim_list = list(tensor.dim)
    dim_list[indices[0]], dim_list[indices[1]] = (
        dim_list[indices[1]],
        dim_list[indices[0]],
    )
    new_dim = tuple(dim_list)

    return type(tensor)(new_values, True, new_dim, tensor.type)

def mul_scalar(tensor, scalar):
    new_values = [value * scalar.values[0] for value in tensor.values]
    return type(tensor)(new_values, True, tensor.dim, tensor.type)

def add_scalar(tensor, scalar):
    new_values = [value + scalar.values[0] for value in tensor.values]
    return type(tensor)(new_values, True, tensor.dim, tensor.type)
