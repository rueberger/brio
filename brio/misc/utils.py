
import numpy as np


# external code from stackoverflow: http://stackoverflow.com/questions/1167617/in-python-how-do-i-indicate-im-overriding-a-method
def overrides(interface_class):
    """ provides an override decorator for python
    simply checks if the decorated method is in interface_class
      and complains if it isnt

    :param interface_class: the class containing the method being overwritten
    """
    def overrider(method):
        """
        Helper function
        """

        if method.__name__.startswith('__'):
            # handles private methods
            priv_name = "_{}{}".format(interface_class.__name__, method.__name__)
            assert priv_name in dir(interface_class)
        else:
            assert method.__name__ in dir(interface_class)
        return method
    return overrider


def normalize_by_row(matrix):
    """
    normalizes rows of matrix
    returns normalized matrix
    """

    row_sums = np.sqrt(np.sum(matrix ** 2, axis=1))
    # replaces 0 with 1 to avoid divide by 0
    np.place(row_sums, row_sums == 0, 1)
    norm_matrix = matrix / row_sums[:, np.newaxis]
    return norm_matrix

def roll_itr(itr, n_elems):
    """ returns a generator that concatenates the next
      n_elems of itr into an array of shape (itr_len, n_elems)
      where itr.next() returns 1d arrays of fixed length itr_len
    2d arrays that might be returned by itr.next() will be flattened

    :param itr: iterator over 1d arrays of fixed length
    :param n_elems: number of elements to concatenate
    :returns: a generator over arrays of shape (itr_len, n_elems)
    :rtype: generator
    """
    concat_arr = []
    for arr in itr:
        concat_arr.append(np.ravel(arr))
        if len(concat_arr) == n_elems:
            yield np.array(concat_arr).T
            concat_arr = []
