from metis.kernel import HornClause, HornDB, fact, pat


def db():
    # a -> b -> c, cycle c -> a; reach = transitive closure
    return HornDB([
        fact("edge", "a", "b"),
        fact("edge", "b", "c"),
        fact("edge", "c", "a"),
        HornClause("r1", pat("reach", "X", "Y"),
                   (pat("edge", "X", "Y"),), vars=("X", "Y")),
        HornClause("r2", pat("reach", "X", "Z"),
                   (pat("edge", "X", "Y"), pat("reach", "Y", "Z")),
                   vars=("X", "Y", "Z")),
    ])


def test_facts_and_chaining():
    d = db()
    assert d.derivable(pat("edge", "a", "b"))
    assert not d.derivable(pat("edge", "b", "a"))
    assert d.derivable(pat("reach", "a", "c"))
    assert d.derivable(pat("reach", "b", "a"))     # via c, through cycle


def test_cycle_terminates_negative():
    # unreachable goal in a cyclic graph: depth bound must terminate
    assert not db().derivable(pat("reach", "a", "d"))


def test_arity_and_predicate_mismatch():
    d = db()
    assert not d.derivable(pat("edge", "a"))
    assert not d.derivable(pat("nosuch", "a", "b"))
