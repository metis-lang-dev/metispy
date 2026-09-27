"""Shared analytic fixtures — small programs with hand-computable
measures. Each returns a fresh Program (tests may mutate weights)."""

from metis.kernel import Clause, Link, Program


def race():
    """Two consumers race for their own tokens. init a:2, b:1.
    Step 1: W(ra)=2, W(rb)=1 -> P(wa hit)=2/3. By step 3 everything is
    consumed regardless of order; step 4 quiesces to done."""
    return Program(
        stages={"run": [
            Clause("ra", consume={"a": 1}, produce={"wa": 1}),
            Clause("rb", consume={"b": 1}, produce={"wb": 1}),
        ]},
        init_stage="run", init={"a": 2, "b": 1})


def ff():
    """Falling factorial: r2 consumes TWO a's out of 3 -> 3*2=6 ordered
    choices vs r1's 2. Step 1: P(p)=6/8."""
    return Program(
        stages={"run": [
            Clause("r2", consume={"a": 2}, produce={"p": 1}),
            Clause("r1", consume={"b": 1}, produce={"q": 1}),
        ]},
        init_stage="run", init={"a": 3, "b": 2})


def persist():
    """!-premise: rp reads k (count 3, squared -> 9) and never decrements
    it. Step 1: P(done)=9/10; k stays 3 on every branch."""
    return Program(
        stages={"s": [
            Clause("rp", consume={"a": 1}, persist={"k": 2},
                   produce={"done": 1}),
            Clause("rq", consume={"a": 1}, produce={"other": 1}),
        ]},
        init_stage="s", init={"k": 3, "a": 1})


def staged():
    """Stage 'work' drains t; at quiescence the link raises flag and
    moves to 'finish', which immediately quiesces to done.
    t=2: steps 1-2 drain, step 3 link, step 4 done."""
    return Program(
        stages={"work": [Clause("w", consume={"t": 1},
                                produce={"d": 1})],
                "finish": []},
        links=[Link("adv", pre="work", post="finish",
                    produce={"flag": 1})],
        init_stage="work", init={"t": 2})


ALL = {"race": (race, 4), "ff": (ff, 3), "persist": (persist, 2),
       "staged": (staged, 5)}
