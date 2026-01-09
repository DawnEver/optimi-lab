from copy import deepcopy

import numpy as np

from optimi_lab.utils.logger import log

PENALTY_VALUE = np.inf  # 2147483647 # 32bit int max value; used to fill unmet objective values


def sort_obj_name_list(obj_name_list: list[str]) -> tuple[list[list[str]], list[list[float]], list[bool]]:
    """Organize `obj_name_list` by operating condition.

    Args:
        obj_name_list (list[str]): List of objective names, format "key_name@oc_index@flag".

    Returns:
        tuple: A tuple containing:
            - obj_name_matrix (list[list[str]]): Objective names grouped by operating condition.
            - err_value_matrix (list[list[float]]): Error/penalty values corresponding to each objective.
            - max_obj_flags (list[bool]): Flags indicating whether each objective is a maximization.

    Examples:
    --------
        >>> obj_name_list = ['a@oc0@max', 'b@oc0@min', 'a@oc1@max']
        >>> obj_name_matrix, err_value_matrix, max_obj_flags = sort_obj_name_list(obj_name_list)
        >>> obj_name_matrix == [['a', 'b'], ['a']]
        True
        >>> err_value_matrix == [[-np.inf, np.inf], [-np.inf]]
        True
        >>> max_obj_flags == [True, False, True]
        True
        >>> obj_name_list = ['a@oc2@invalid']
        >>> try:
        ...     sort_obj_name_list(obj_name_list)
        ... except KeyError as e:
        ...     str(e) == "'unmatched flag invalid'"
        True


    ------

    Notes:
        ```
        obj_name_list = ["a@oc0@max","b@oc0@min","a@oc1@max"]
        Use dictionary grouping to collect by operating condition:
        obj_name_matrix = {
            0:[a,b],
            1:[a],
        }
        obj_flag_matrix = {
            0:["max","min"],
            1:["max"],
        }
        Then convert to 2D lists:
        obj_name_matrix = [
            [a,b],
            [a],
        ]
        obj_flag_matrix = [
            ["max","min"],
            ["max"],
        ]
        ```
        Where `oc_index` is the operating condition index, `key_name` is the objective name, and `flag` is 'max'/'min'.

    """
    obj_name_matrix = {}
    err_value_matrix = {}
    max_obj_flags = []
    for _obj_name in obj_name_list:
        key_name, oc_index, flag = _obj_name.split('@')
        # oc: operation condition

        # flag max/min
        if flag == 'max':
            err_fill_value = -PENALTY_VALUE
        elif flag == 'min':
            err_fill_value = PENALTY_VALUE
        else:
            msg = f'unmatched flag {flag}'
            raise (KeyError(msg))

        if oc_index not in obj_name_matrix:
            obj_name_matrix[oc_index] = []
            err_value_matrix[oc_index] = []

        obj_name_matrix[oc_index].append(key_name)
        max_obj_flags.append(flag == 'max')
        err_value_matrix[oc_index].append(err_fill_value)

    obj_name_matrix = list(obj_name_matrix.values())
    err_value_matrix = list(err_value_matrix.values())
    return obj_name_matrix, err_value_matrix, max_obj_flags


def generate_kwds_list(
    var_name_list: list[str], var_value_matrix: np.ndarray, base_params_dict_list: list[dict[str, dict]]
) -> list[dict]:
    """Generates a list of arguments dict for the simulation based on the variable value matrix and base parameters.

    Args:
        var_name_list (list[str]): List of variable names to be used in the simulation.
        var_value_matrix (np.ndarray): A matrix containing variable values for the simulation.
        base_params_dict_list (list[dict[str,dict]]): A list of base parameters for the simulation.

    Returns:
        list[dict]: A list of arguments dict for the simulation.

    Examples:
    --------
        >>> var_name_list = ['a.b@Hz@oc0', 'c.0.d.e@mm@oc1', 'c.1.f.1@deg@oc1']
        >>> var_value_matrix = np.array([[50, 0.5, 60], [60, 0.6, 70]])
        >>> base_params_dict = {'a': {'b': '40 Hz'}, 'c': [{'d': {'e': '0.4 mm'}}, {'f': ['35 deg', '55 deg']}]}
        >>> base_params_dict_list = [base_params_dict] * 2
        >>> kwds_list = generate_kwds_list(var_name_list, var_value_matrix, base_params_dict_list)
        >>> len(kwds_list) == 4
        True

    ------

    Notes:
        `var_name_list` is a list where each element is a string formatted as
        "var_path@var_unit@var_oc_id", where:
        - `var_path` is the variable path separated by dots, e.g. 'problem_parameters.sim_frequency' or 'ring_input_list.0.basic.inner_radius_rel'. Numeric parts indicate list indices, strings indicate dict keys.
        - `var_unit` is the variable unit, e.g. 'Hz' or 'm'.
        - `var_oc_id` is the operating condition identifier, e.g. 'oc0'.

    """
    kwds_list = []
    case_id = 0
    for var_value_row in var_value_matrix:
        args_dict = {}
        for i_var, var_str in enumerate(var_name_list):
            var_path, var_unit, var_oc_id = var_str.split('@')
            oc_id = int(var_oc_id[2:])  # convert oc index to integer
            if oc_id not in args_dict:
                args_dict[oc_id] = deepcopy(base_params_dict_list[oc_id])
                args_dict[oc_id]['case_id'] = case_id
                case_id += 1

            # parse var_path
            var_path_sep = var_path.split('.')
            # traverse nested structure to find the target variable
            target = args_dict[oc_id]
            for current_key in var_path_sep[:-1]:
                if current_key.isdigit():
                    current_key = int(current_key)
                target = target[current_key]
            last_key = var_path_sep[-1]
            # digits indicate list indices, strings indicate dict keys
            if last_key.isdigit():
                last_key = int(last_key)
            target[last_key] = f'{var_value_row[i_var]} {var_unit}'

        kwds_list += list(args_dict.values())
    return kwds_list


def parse_outputs(
    obj_name_matrix: list[list[str]],
    err_value_matrix: list[list[float]],
    n_obj: int,
    n_sample: int,
    output_dict_list: list[dict],
) -> np.ndarray:
    """Parse outputs: convert `output_dict_list` into an objective value matrix per operating condition.

    Args:
        obj_name_matrix (list[list[str]]): Matrix of objective function names for each operating condition.
        err_value_matrix (list[list[float]]): Matrix of error/penalty values for each operating condition.
        n_obj (int): Number of objective functions.
        n_sample (int): Number of samples in the variable value matrix.
        output_dict_list (list[dict]): List of output dictionaries from the simulation.

    Returns:
        np.ndarray: A matrix containing the objective function values for each sample.

    Examples:
    --------
        >>> obj_name_matrix = [['a', 'b'], ['c']]
        >>> err_value_matrix = [[-np.inf, np.inf], [np.inf]]
        >>> n_obj = 3
        >>> n_sample = 2
        >>> output_dict_list = [
        ...     {'case_id': 0, 'solution_type': 'SUCCESS', 'a': 1, 'b': 2},
        ...     {'case_id': 1, 'solution_type': 'SUCCESS', 'c': 3},
        ...     {'case_id': 2, 'solution_type': 'ERROR', 'a': 1, 'b': 2},
        ...     {'case_id': 3, 'solution_type': 'SUCCESS', 'c': 3},
        ... ]
        >>> obj_value_matrix = parse_outputs(obj_name_matrix, err_value_matrix, n_obj, n_sample, output_dict_list)
        >>> np.allclose(obj_value_matrix, np.array([[1, 2, 3], [-np.inf, np.inf, 3]]))
        True
        >>> try:
        ...     parse_outputs(
        ...         obj_name_matrix,
        ...         err_value_matrix,
        ...         n_obj,
        ...         n_sample,
        ...         [{'case_id': 0, 'solution_type': 'SUCCESS'}, {'case_id': 0, 'solution_type': 'SUCCESS'}],
        ...     )
        ... except KeyError as e:
        ...     str(e).startswith('"Invalid key name')
        True

    ------

    """
    num_oc = len(obj_name_matrix)
    obj_value_matrix = np.zeros((n_sample, n_obj))
    output = dict.fromkeys(range(num_oc))

    for index, result in enumerate(output_dict_list):
        index_oc = result['case_id'] % num_oc  # index of operating condition

        obj_name_list_oc = obj_name_matrix[index_oc]

        if result['solution_type'] in ['ERROR', 'TIMEOUT', 'NONE']:
            # For ERROR/TIMEOUT/NONE results, use the predefined error/penalty values
            """
            ERROR: simulation runtime error
            TIMEOUT: simulation timed out
            NONE: simulation not run (used for plotting only)
            """
            output[index_oc] = err_value_matrix[index_oc]
        else:
            try:
                output[index_oc] = [result[key_name] for key_name in obj_name_list_oc]
            except KeyError as e:
                msg = f'Invalid key name obj_name_list_oc: {obj_name_list_oc}:{e}'
                log(msg, level='ERROR')
                raise KeyError(msg)

        if index_oc == num_oc - 1:
            obj_value_matrix[index // num_oc] = [value for sublist in output.values() for value in sublist]
    return obj_value_matrix


if __name__ == '__main__':
    import doctest

    doctest.testmod()
