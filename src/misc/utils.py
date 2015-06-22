
import numpy as np


# external code from stackoverflow: http://stackoverflow.com/questions/1167617/in-python-how-do-i-indicate-im-overriding-a-method
def overrides(interface_class):
    """
    override decorator. need to specify the super class, unfortunately
    """

    def overrider(method):
        assert  method.__name__ in dir(interface_class)
        return method
    return overrider


def normalize_by_row(matrix):
    """
    normalizes rows of matrix
    returns normalized matrix
    """
    row_sums = matrix.sum(axis=1)
    # replaces 0 with 1 to avoid divide by 0
    np.place(row_sums, row_sums == 0, 1)
    norm_matrix = matrix / row_sums[:, np.newaxis]
    return norm_matrix
