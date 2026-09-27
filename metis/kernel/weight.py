"""CHOICE weights — the heart of the measure.

transition_count(clause, counts) is the number of distinct ORDERED
resource choices instantiating the clause in the multiset: the falling
factorial over linear premises (each copy of a resource is a distinct
choice, and consumed copies exclude each other) times countⁿ over
persistent premises (never excluded). This is exactly the ceptre engine's
uniform-picker multiplicity, and it is what makes resource competition
price explaining-away correctly in the projected measure.

choice_weight adds the clause prior on top (parity mode: prior = 1)."""


def falling_factorial(c: int, n: int) -> int:
    w = 1
    for i in range(n):
        w *= c - i
        if w <= 0:
            return 0
    return w


def transition_count(clause, counts) -> int:
    w = 1
    for p, n in clause.consume.items():
        w *= falling_factorial(counts.get(p, 0), n)
        if w == 0:
            return 0
    for p, n in clause.persist.items():
        w *= counts.get(p, 0) ** n
        if w == 0:
            return 0
    return w


def choice_weight(clause, counts) -> float:
    n = transition_count(clause, counts)
    return clause.weight * n if n else 0.0
