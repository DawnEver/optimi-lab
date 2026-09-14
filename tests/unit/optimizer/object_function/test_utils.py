"""Coverage for the object-function helpers, in the cases the doctests do not reach.

The doctests in ``optimi_lab/object_function/utils.py`` are the richest in the package and
they keep covering each function's happy path -- this file does not restate them, it covers
what they leave out:

- every exception doctest was written as a BARE EXPRESSION inside its ``except`` block
  (``str(e).startswith(...)``), which doctest evaluates and DISCARDS, so the claimed error was
  never actually compared. The doctests now print the value they claim; the tests here assert
  the exception and its message;
- ``parse_outputs``' TIMEOUT/NONE branch, its ``case_id % num_oc`` cycling and its KeyError
  were untested, as was the row left behind by a sample that never completed a round. That
  last case is now a REFUSAL with its own tests: a short cycle used to leave a row of ZEROS
  -- and, when the missing condition was not the last one, a row carrying the PREVIOUS
  sample's values, which is a plausible objective that means the opposite of "not
  evaluated";
- ``generate_kwds_list`` was asserted only by ``len(kwds_list) == 4``: neither the nested-path
  traversal it exists for, nor the deep copy that keeps ``base_params_dict_list`` unmutated
  across a sweep.
"""

import numpy as np
import pytest

from optimi_lab.object_function.utils import generate_kwds_list, parse_outputs, sort_obj_name_list
from optimi_lab.utils.exceptions import ParameterException


class TestSortObjNameList:
    """`sort_obj_name_list` groups objectives by operating condition and flags max/min."""

    def test_groups_by_operating_condition_in_encounter_order(self):
        """Objectives regrouped per operating condition, independently of oc interleaving."""
        obj_name_list = ['m@oc0@max', 'n@oc1@min', 'o@oc0@min']

        obj_name_matrix, err_value_matrix, max_obj_flags = sort_obj_name_list(obj_name_list)

        assert obj_name_matrix == [['m', 'o'], ['n']]
        assert err_value_matrix == [[-np.inf, np.inf], [np.inf]]
        assert max_obj_flags == [True, False, False]

    def test_unmatched_flag_raises_with_the_flag_it_rejected(self):
        """An unknown max/min flag raises KeyError naming the flag, not silently min-ing it."""
        with pytest.raises(KeyError, match='unmatched flag invalid'):
            sort_obj_name_list(['a@oc0@invalid'])

    def test_flags_place_the_penalty_on_the_correct_side(self):
        """A maximization is penalised at -inf and a minimization at +inf, per objective."""
        _, err_value_matrix, max_obj_flags = sort_obj_name_list(['a@oc0@max', 'b@oc0@min', 'c@oc1@max'])

        assert err_value_matrix == [[-np.inf, np.inf], [-np.inf]]
        assert max_obj_flags == [True, False, True]


class TestGenerateKwdsList:
    """`generate_kwds_list` writes swept values into a nested parameter structure."""

    @pytest.fixture
    def base_params_dict(self) -> dict:
        return {
            'a': {'b': '40 Hz'},
            'c': [{'d': {'e': '0.4 mm'}}, {'f': ['35 deg', '55 deg']}],
        }

    @pytest.fixture
    def var_name_list(self) -> list[str]:
        return ['a.b@Hz@oc0', 'c.0.d.e@mm@oc1', 'c.1.f.1@deg@oc1']

    def test_values_land_at_nested_dict_and_list_paths(self, var_name_list, base_params_dict):
        """A dotted path walks dict keys and list indices, and writes 'value unit'.

        The matrix is float (it comes from sampling), so the written strings carry the float
        rendering -- '50.0 Hz', not '50 Hz' -- which is what a consumer parsing them sees.
        """
        var_value_matrix = np.array([[50, 0.5, 60], [60, 0.6, 70]])

        kwds_list = generate_kwds_list(var_name_list, var_value_matrix, [base_params_dict] * 2)

        assert len(kwds_list) == 4
        assert kwds_list[0]['a']['b'] == '50.0 Hz'
        assert kwds_list[0]['c'][0]['d']['e'] == '0.4 mm', 'oc0\'s case must not carry an oc1 variable'
        assert kwds_list[1]['c'][0]['d']['e'] == '0.5 mm'
        assert kwds_list[1]['c'][1]['f'][1] == '60.0 deg'
        assert kwds_list[2]['a']['b'] == '60.0 Hz'
        assert kwds_list[3]['c'][1]['f'][1] == '70.0 deg'

    def test_each_sample_gets_its_own_case_id_in_operating_condition_order(self, var_name_list, base_params_dict):
        """`case_id` counts samples in operating-condition order; that is how outputs map back."""
        var_value_matrix = np.array([[50, 0.5, 60], [60, 0.6, 70]])

        kwds_list = generate_kwds_list(var_name_list, var_value_matrix, [base_params_dict] * 2)

        assert [kwds['case_id'] for kwds in kwds_list] == [0, 1, 2, 3]

    def test_the_base_params_are_not_mutated_by_a_sweep(self, var_name_list, base_params_dict):
        """Every case works on a deep copy, so a sweep cannot accumulate into the base."""
        var_value_matrix = np.array([[50, 0.5, 60], [60, 0.6, 70]])

        generate_kwds_list(var_name_list, var_value_matrix, [base_params_dict] * 2)

        assert base_params_dict['a']['b'] == '40 Hz'
        assert base_params_dict['c'][0]['d']['e'] == '0.4 mm'
        assert base_params_dict['c'][1]['f'] == ['35 deg', '55 deg']

    def test_the_same_case_is_written_once_per_operating_condition(self, var_name_list, base_params_dict):
        """Two variables in one operating condition share one case, not two."""
        var_value_matrix = np.array([[50, 0.5, 60]])

        kwds_list = generate_kwds_list(var_name_list, var_value_matrix, [base_params_dict] * 2)

        assert len(kwds_list) == 2
        assert kwds_list[1]['c'][0]['d']['e'] == '0.5 mm'
        assert kwds_list[1]['c'][1]['f'][1] == '60.0 deg'


class TestParseOutputs:
    """`parse_outputs` turns per-case simulation results into an objective matrix."""

    @pytest.fixture
    def obj_name_matrix(self) -> list[list[str]]:
        return [['a', 'b'], ['c']]

    @pytest.fixture
    def err_value_matrix(self) -> list[list[float]]:
        return [[-np.inf, np.inf], [np.inf]]

    def test_success_rows_are_read_in_operating_condition_order(self, obj_name_matrix, err_value_matrix):
        """Results cycle back through the operating conditions, one matrix row per sample."""
        output_dict_list = [
            {'case_id': 0, 'solution_type': 'SUCCESS', 'a': 1, 'b': 2},
            {'case_id': 1, 'solution_type': 'SUCCESS', 'c': 3},
            {'case_id': 2, 'solution_type': 'SUCCESS', 'a': 4, 'b': 5},
            {'case_id': 3, 'solution_type': 'SUCCESS', 'c': 6},
        ]

        obj_value_matrix = parse_outputs(obj_name_matrix, err_value_matrix, 3, 2, output_dict_list)

        assert np.allclose(obj_value_matrix, np.array([[1, 2, 3], [4, 5, 6]]))

    @pytest.mark.parametrize('solution_type', ['ERROR', 'TIMEOUT', 'NONE'])
    def test_failed_cases_take_the_penalty_values(self, obj_name_matrix, err_value_matrix, solution_type):
        """ERROR, TIMEOUT and NONE all write the per-objective penalty, never a real value.

        Only ERROR was exercised before; TIMEOUT and NONE are handled by the same branch, and a
        failed case that produced a plausible number would be indistinguishable from a run.
        """
        output_dict_list = [
            {'case_id': 0, 'solution_type': solution_type, 'a': 1, 'b': 2},
            {'case_id': 1, 'solution_type': 'SUCCESS', 'c': 3},
            {'case_id': 2, 'solution_type': 'SUCCESS', 'a': 4, 'b': 5},
            {'case_id': 3, 'solution_type': 'SUCCESS', 'c': 6},
        ]

        obj_value_matrix = parse_outputs(obj_name_matrix, err_value_matrix, 3, 2, output_dict_list)

        assert np.allclose(obj_value_matrix, np.array([[-np.inf, np.inf, 3], [4, 5, 6]]))

    def test_a_missing_objective_key_raises(self, obj_name_matrix, err_value_matrix):
        """A SUCCESS result that does not carry every objective raises, naming the keys."""
        output_dict_list = [
            {'case_id': 0, 'solution_type': 'SUCCESS'},
            {'case_id': 1, 'solution_type': 'SUCCESS', 'c': 3},
        ]

        with pytest.raises(KeyError, match='Invalid key name obj_name_list_oc'):
            parse_outputs(obj_name_matrix, err_value_matrix, 3, 2, output_dict_list)

    def test_a_sample_whose_last_condition_never_reports_is_refused(self, obj_name_matrix, err_value_matrix):
        """A cycle that never completes RAISES; it does not leave a row of zeros behind.

        A row of zeros reads as a real objective, so a sample that was never evaluated arrived
        at the caller as the number 0 -- the one thing an objective matrix must not do. The
        refusal names the sample instead, and the matrix is not returned at all, so no partial
        result can be mistaken for a complete one.
        """
        output_dict_list = [
            {'case_id': 0, 'solution_type': 'SUCCESS', 'a': 1, 'b': 2},
            {'case_id': 1, 'solution_type': 'SUCCESS', 'c': 3},
            {'case_id': 2, 'solution_type': 'SUCCESS', 'a': 4, 'b': 5},
        ]

        with pytest.raises(ParameterException, match='never completed'):
            parse_outputs(obj_name_matrix, err_value_matrix, 3, 2, output_dict_list)

    def test_a_missing_condition_cannot_reuse_the_previous_sample_values(self, obj_name_matrix, err_value_matrix):
        """A cycle missing a MIDDLE condition is refused too, and this is the sharper half.

        The staging dict was never reset between cycles, so the missing condition kept the
        PREVIOUS sample's values and the row was written anyway: sample 1 used to come back as
        ``[1, 2, 6]`` -- sample 0's ``a`` and ``b`` -- instead of zeros. A wrong number is
        worse than an absent one, so the completeness check is per condition, not per sample.
        """
        output_dict_list = [
            {'case_id': 0, 'solution_type': 'SUCCESS', 'a': 1, 'b': 2},
            {'case_id': 1, 'solution_type': 'SUCCESS', 'c': 3},
            {'case_id': 3, 'solution_type': 'SUCCESS', 'c': 6},
        ]

        with pytest.raises(ParameterException, match='never completed'):
            parse_outputs(obj_name_matrix, err_value_matrix, 3, 2, output_dict_list)

    def test_an_empty_result_list_is_refused_when_samples_were_declared(self, obj_name_matrix, err_value_matrix):
        """No results at all is the same defect at full size: every declared sample is missing."""
        with pytest.raises(ParameterException, match='never completed'):
            parse_outputs(obj_name_matrix, err_value_matrix, 3, 2, [])

    def test_a_case_id_beyond_the_declared_samples_is_refused(self, obj_name_matrix, err_value_matrix):
        """Results for samples the caller did not declare are refused, naming the count.

        This used to be an ``IndexError`` from the row assignment -- loud, but it named numpy's
        array bounds rather than the disagreement between the results and ``n_sample``.
        """
        output_dict_list = [
            {'case_id': 0, 'solution_type': 'SUCCESS', 'a': 1, 'b': 2},
            {'case_id': 1, 'solution_type': 'SUCCESS', 'c': 3},
            {'case_id': 2, 'solution_type': 'SUCCESS', 'a': 4, 'b': 5},
            {'case_id': 3, 'solution_type': 'SUCCESS', 'c': 6},
            {'case_id': 4, 'solution_type': 'SUCCESS', 'a': 7, 'b': 8},
            {'case_id': 5, 'solution_type': 'SUCCESS', 'c': 9},
        ]

        with pytest.raises(ParameterException, match='n_sample=2'):
            parse_outputs(obj_name_matrix, err_value_matrix, 3, 2, output_dict_list)

    def test_no_declared_samples_is_an_empty_matrix_rather_than_a_refusal(self, obj_name_matrix, err_value_matrix):
        """``n_sample=0`` with no results is consistent, so it returns a 0-row matrix."""
        obj_value_matrix = parse_outputs(obj_name_matrix, err_value_matrix, 3, 0, [])

        assert obj_value_matrix.shape == (0, 3)

    def test_no_declared_objectives_is_refused_rather_than_reported_as_incomplete(self):
        """No operating conditions at all is refused by name, not blamed on the samples.

        The per-condition completeness check has nothing to look at here, so without this guard
        it reported every sample as incomplete with an EMPTY list of missing conditions -- a
        claim about nothing. There is no ``case_id % num_oc`` mapping to derive either.
        """
        with pytest.raises(ParameterException, match='declares no operating condition'):
            parse_outputs([], [], 0, 2, [])
