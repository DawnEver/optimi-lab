import numpy as np


def check_dominated(obj1: np.ndarray, obj2: np.ndarray) -> bool | np.ndarray[bool]:
    """Check whether `obj1` dominates `obj2` (assumes a minimization problem).

    Args:
        obj1: Objective values 1, shape (n_obj) or (pop_size, n_obj)
        obj2: Objective values 2, shape (n_obj) or (pop_size, n_obj)

    Returns:
        bool | np.ndarray: Boolean array; True indicates `obj1[i]` dominates `obj2[i]`,
            shape (pop_size,)

    Example:
        >>> import numpy as np
        >>> obj1 = np.array([1, 2])
        >>> obj2 = np.array([2, 3])
        >>> np.allclose(check_dominated(obj1, obj2), True)
        True
        >>> obj3 = np.array([0.5, 1.5])
        >>> np.allclose(check_dominated(obj1, obj3), False)
        True
        >>> obj4 = np.array([1, 2])
        >>> np.allclose(check_dominated(obj1, obj4), False)
        True
        >>> obj5 = np.array([1.5, 1.5])
        >>> np.allclose(check_dominated(obj1, obj5), False)
        True

    """
    # obj1 dominates obj2 iff: obj1 is no worse than obj2 on all objectives
    # and strictly better on at least one objective
    all_better_or_equal = np.all(obj1 <= obj2, axis=-1)
    any_better = np.any(obj1 < obj2, axis=-1)

    return all_better_or_equal & any_better


def non_dominated_sorting(obj_mat: np.ndarray, only_first_front: bool = True) -> np.ndarray | list[np.ndarray]:
    """Efficient non-dominated sorting using numpy broadcasting in minimization problem.

    Args:
        obj_mat (np.ndarray): 2D array, each row is a point.
        only_first_front (bool): If True, return only first front indices and non front indices; else all fronts.

    Returns:
        np.ndarray or list[np.ndarray]: Indices of first front or all fronts.

    Examples:
        >>> import numpy as np
        >>> mat = np.array([[1, 2], [2, 1], [1, 1], [0, 0]])
        >>> non_dominated_sorting(mat)
        array([3])
        >>> non_dominated_sorting(mat, only_first_front=False)
        [array([3]), array([2]), array([0, 1])]
        >>> non_dominated_sorting(np.array([]))
        array([], dtype=float64)

    """
    n = obj_mat.shape[0]
    if n == 0:
        return np.array([], dtype=np.float64)

    dominates = check_dominated(obj_mat[:, None, :], obj_mat[None, :, :])

    domination_count = np.sum(dominates.T, axis=1)  # how many times each is dominated

    current_front = np.where(domination_count == 0)[0]
    if only_first_front:
        return current_front

    fronts = [current_front]
    dominated_solutions = [np.where(dominates[i])[0].tolist() for i in range(n)]
    domination_count = domination_count.copy()

    while len(current_front) > 0:
        next_front = []
        for i in current_front:
            for j in dominated_solutions[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        if next_front:
            fronts.append(np.array(next_front))
        current_front = next_front

    return fronts


if __name__ == '__main__':
    import doctest

    doctest.testmod()
