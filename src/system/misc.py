import numpy as np


def is_sorted(a):
    """
    Check if an array or list is sorted in non-decreasing order.

    This function verifies that each element in the sequence is less than
    or equal to the next element. It performs an efficient linear scan
    and returns as soon as it finds any violation of the sorted order.

    Parameters
    ----------
    a : array_like
        Input array or list to check for sorted order. Can be any object
        that can be converted to a numpy array.

    Returns
    -------
    bool
        True if the array is sorted in non-decreasing order, False otherwise.

    Examples
    --------
    >>> is_sorted([1, 2, 3, 4, 5])
    True
    >>> is_sorted([1, 3, 2, 4, 5])
    False
    >>> is_sorted([5, 4, 3, 2, 1])
    False
    >>> is_sorted([1, 2, 2, 3, 4])  # Duplicates allowed
    True
    >>> is_sorted(np.array([1.0, 2.5, 3.7]))
    True

    Notes
    -----
    - Uses strict inequality check (`<`) to detect violations
    - Allows duplicate values (non-strictly increasing)
    - Returns True for empty arrays and single-element arrays
    - More efficient than np.all(np.diff(a) >= 0) for early termination
    """
    a = np.asarray(a)
    for i in range(a.size - 1):
        if a[i + 1] < a[i]:
            return False
    return True


def is_unique(a):
    """
    Check if all elements in a sorted array or list are unique.

    This function verifies that no two consecutive elements in the array / list
    are equal. It assumes the input is sorted and checks for duplicate
    consecutive elements. Repeating: For accurate results, the input should be sorted first.

    Parameters
    ----------
    a : array_like
        Input array or list to check for uniqueness. Should be sorted for
        correct results. Can be any object that can be converted to a numpy array.

    Returns
    -------
    bool
        True if all elements are unique, False if any duplicates are found.

    Examples
    --------
    >>> is_unique([1, 2, 3, 4, 5])  # Assumes sorted input
    True
    >>> is_unique([1, 2, 2, 3, 4])  # Duplicate detected
    False
    >>> is_unique([1])  # Single element is always unique
    True
    >>> is_unique([])   # Empty array is considered unique
    True

    Notes
    -----
    - This function assumes the input array is sorted
    - For unsorted arrays, use with sorted array: `is_unique(np.sort(a))`
    - Checks only consecutive elements, so sorting is necessary
    - More efficient than len(set(a)) == len(a) for large arrays (early termination)
    - Returns True for empty and single-element arrays by definition
    """
    a = np.asarray(a)
    for i in range(a.size - 1):
        if a[i + 1] == a[i]:
            return False
    return True
