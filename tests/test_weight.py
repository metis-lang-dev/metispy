from metis.kernel import Clause, falling_factorial, transition_count, \
    choice_weight


def test_falling_factorial():
    assert falling_factorial(3, 2) == 6
    assert falling_factorial(3, 1) == 3
    assert falling_factorial(3, 0) == 1
    assert falling_factorial(2, 3) == 0
    assert falling_factorial(0, 1) == 0


def test_transition_count_linear_and_persistent():
    c = Clause("c", consume={"a": 2}, persist={"k": 2})
    assert transition_count(c, {"a": 3, "k": 3}) == 6 * 9
    assert transition_count(c, {"a": 1, "k": 3}) == 0   # not enabled
    assert transition_count(c, {"a": 3}) == 0           # missing !k


def test_prior_weight_multiplies():
    c = Clause("c", consume={"a": 1}, weight=2.5)
    assert choice_weight(c, {"a": 2}) == 5.0
    assert choice_weight(c, {}) == 0.0                  # disabled stays 0
