import numpy as np
import pytest

from cubesat_attitude.quaternion import (
    conjugate_quaternion,
    hamilton_product,
    inverse_quaternion,
    normalize_quaternion,
)


def test_normalize_simple_quaternion() -> None:
    q = np.array([2.0, 0.0, 0.0, 0.0])

    result = normalize_quaternion(q)

    expected = np.array([1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(result, expected)


def test_normalize_general_quaternion() -> None:
    q = np.array([1.0, 2.0, 2.0, 0.0])

    result = normalize_quaternion(q)

    expected = np.array([1 / 3, 2 / 3, 2 / 3, 0.0])
    np.testing.assert_allclose(result, expected)


def test_normalized_quaternion_has_unit_norm() -> None:
    q = np.array([1.0, -4.0, 2.0, 3.0])

    result = normalize_quaternion(q)

    np.testing.assert_allclose(np.linalg.norm(result), 1.0)


def test_normalize_rejects_zero_quaternion() -> None:
    q = np.zeros(4)

    with pytest.raises(ValueError):
        normalize_quaternion(q)


def test_normalize_rejects_invalid_shape() -> None:
    q = np.array([1.0, 0.0, 0.0])

    with pytest.raises(ValueError):
        normalize_quaternion(q)


def test_normalize_does_not_modify_input() -> None:
    q = np.array([1.0, 2.0, 2.0, 0.0])
    q_before = q.copy()

    normalize_quaternion(q)

    np.testing.assert_array_equal(q, q_before)


def test_conjugate_identity_quaternion() -> None:
    q = np.array([1.0, 0.0, 0.0, 0.0])

    result = conjugate_quaternion(q)

    expected = np.array([1.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(result, expected)


def test_conjugate_changes_vector_part_sign() -> None:
    q = np.array([0.0, 1.0, 2.0, 3.0])

    result = conjugate_quaternion(q)

    expected = np.array([0.0, -1.0, -2.0, -3.0])
    np.testing.assert_array_equal(result, expected)


def test_double_conjugation_returns_original_quaternion() -> None:
    q = np.array([1.0, 2.0, 3.0, 4.0])

    result = conjugate_quaternion(conjugate_quaternion(q))

    np.testing.assert_array_equal(result, q)


def test_conjugate_does_not_modify_input() -> None:
    q = np.array([1.0, 2.0, 3.0, 4.0])
    q_before = q.copy()

    conjugate_quaternion(q)

    np.testing.assert_array_equal(q, q_before)


def test_conjugate_raises_value_error_for_invalid_shape() -> None:
    q = np.array([1.0, 0.0, 0.0])

    with pytest.raises(ValueError):
        conjugate_quaternion(q)


def test_conjugate_preserves_norm() -> None:
    q = np.array([1.0, 2.0, 3.0, 4.0])

    result = conjugate_quaternion(q)

    np.testing.assert_allclose(np.linalg.norm(result), np.linalg.norm(q))


def test_hamilton_product_known_result() -> None:
    q1 = np.array([1.0, 2.0, 3.0, 4.0])
    q2 = np.array([5.0, 6.0, 7.0, 8.0])

    result = hamilton_product(q1, q2)

    expected = np.array([-60.0, 12.0, 30.0, 24.0])
    np.testing.assert_array_equal(result, expected)


def test_hamilton_product_with_identity() -> None:
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    q = np.array([2.0, 3.0, 4.0, 5.0])

    left_result = hamilton_product(identity, q)
    right_result = hamilton_product(q, identity)

    np.testing.assert_array_equal(left_result, q)
    np.testing.assert_array_equal(right_result, q)


def test_hamilton_product_is_not_commutative() -> None:
    q1 = np.array([1.0, 2.0, 3.0, 4.0])
    q2 = np.array([5.0, 6.0, 7.0, 8.0])

    result_12 = hamilton_product(q1, q2)
    result_21 = hamilton_product(q2, q1)

    assert not np.array_equal(result_12, result_21)


def test_hamilton_product_is_associative() -> None:
    q = np.array([1.0, 1.0, 0.0, 0.0])
    p = np.array([1.0, 0.0, 1.0, 0.0])
    r = np.array([1.0, 0.0, 0.0, 1.0])

    left = hamilton_product(
        hamilton_product(q, p),
        r,
    )

    right = hamilton_product(
        q,
        hamilton_product(p, r),
    )

    np.testing.assert_allclose(left, right)


def test_hamilton_product_with_conjugate_returns_squared_norm() -> None:
    q = np.array([1.0, 2.0, 3.0, 4.0])
    q_conjugate = conjugate_quaternion(q)

    result = hamilton_product(q, q_conjugate)

    expected = np.array([np.linalg.norm(q) ** 2, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(result, expected)


def test_conjugate_hamilton_product_returns_squared_norm() -> None:
    q = np.array([1.0, 2.0, 3.0, 4.0])
    q_conjugate = conjugate_quaternion(q)

    result = hamilton_product(q_conjugate, q)

    expected = np.array([np.linalg.norm(q) ** 2, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(result, expected)


def test_hamilton_product_does_not_modify_inputs() -> None:
    q1 = np.array([1.0, 2.0, 3.0, 4.0])
    q2 = np.array([5.0, 6.0, 7.0, 8.0])

    q1_before = q1.copy()
    q2_before = q2.copy()

    hamilton_product(q1, q2)

    np.testing.assert_array_equal(q1, q1_before)
    np.testing.assert_array_equal(q2, q2_before)


def test_hamilton_product_raises_value_error_for_invalid_first_shape() -> None:
    q1 = np.array([1.0, 0.0, 0.0])
    q2 = np.array([1.0, 0.0, 0.0, 0.0])

    with pytest.raises(ValueError):
        hamilton_product(q1, q2)


def test_hamilton_product_raises_value_error_for_invalid_second_shape() -> None:
    q1 = np.array([1.0, 0.0, 0.0, 0.0])
    q2 = np.array([1.0, 0.0, 0.0])

    with pytest.raises(ValueError):
        hamilton_product(q1, q2)


def test_inverse_quaternion_known_result() -> None:
    q = np.array([1.0, 2.0, 0.0, 0.0])

    result = inverse_quaternion(q)

    expected = np.array([0.2, -0.4, 0.0, 0.0])
    np.testing.assert_allclose(result, expected)


def test_inverse_quaternion_is_multiplicative_inverse() -> None:
    q = np.array([1.0, 2.0, 3.0, 4.0])
    q_inverse = inverse_quaternion(q)

    left_result = hamilton_product(q, q_inverse)
    right_result = hamilton_product(q_inverse, q)

    identity = np.array([1.0, 0.0, 0.0, 0.0])

    np.testing.assert_allclose(left_result, identity, atol=1e-12)
    np.testing.assert_allclose(right_result, identity, atol=1e-12)


def test_inverse_of_unit_quaternion_equals_conjugate() -> None:
    q = np.array([0.5, 0.5, 0.5, 0.5])
    q_inverse = inverse_quaternion(q)
    q_conjugate = conjugate_quaternion(q)

    np.testing.assert_allclose(q_inverse, q_conjugate)


def test_inverse_quaternion_does_not_modify_input() -> None:
    q = np.array([1.0, 2.0, 3.0, 4.0])
    q_before = q.copy()

    inverse_quaternion(q)

    np.testing.assert_array_equal(q, q_before)


def test_inverse_quaternion_raises_value_error_for_zero_quaternion() -> None:
    q = np.zeros(4)

    with pytest.raises(ValueError):
        inverse_quaternion(q)


def test_inverse_quaternion_raises_value_error_for_invalid_shape() -> None:
    q = np.array([1.0, 0.0, 0.0])

    with pytest.raises(ValueError):
        inverse_quaternion(q)