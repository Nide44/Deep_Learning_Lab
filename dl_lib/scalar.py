from dl_lib.matrix import Matrix


class Scalar(Matrix):
    def __init__(self, values, predefined=False, dim=None, element_type=None):
        super().__init__(values, predefined, dim, element_type)
