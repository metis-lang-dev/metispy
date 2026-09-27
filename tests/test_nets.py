"""Stage 3 gate: program identity invariant under presentation (>=1000
shuffle trials), discriminating on real differences; run identity
invariant under interleaving of independent events (with replay
validation); the projection cache exact on a revisited scene stream."""

import random

from pytest import approx

from metis.kernel import Clause, ForwardProjection, Program, \
    ProjectionCache, canonical, independent_events, program_key, \
    run_key, run_trace
from metis.domains.road import base_catalog, scenario
from fixtures import ALL


def shuffled(program, rng):
    stages = {}
    for s, body in program.stages.items():
        b = list(body)
        rng.shuffle(b)
        stages[s] = b
    links = list(program.links)
    rng.shuffle(links)
    return Program(stages=stages, links=links,
                   init_stage=program.init_stage,
                   init=dict(program.init))


# -- program identity ----------------------------------------------------

def test_permutation_invariance_1000_trials():
    rng = random.Random(3)
    road = base_catalog().ground(scenario())
    subjects = [road] + [make() for make, _ in ALL.values()]
    keys = [program_key(p) for p in subjects]
    trials = 0
    while trials < 1000:
        for p, k in zip(subjects, keys):
            assert program_key(shuffled(p, rng)) == k
            trials += 1


def test_name_independence():
    p = base_catalog().ground(scenario())
    renamed = Program(
        stages={s: [Clause(f"x{i}", dict(c.consume), dict(c.produce),
                           dict(c.persist), c.weight)
                    for i, c in enumerate(body)]
                for s, body in p.stages.items()},
        links=list(p.links), init_stage=p.init_stage,
        init=dict(p.init))
    assert program_key(renamed) == program_key(p)


def test_discrimination_corpus():
    def base():
        return base_catalog().ground(scenario())

    mutants = []
    p = base()                                       # reference
    mutants.append(p)

    m = base()                                       # weight nudged
    m.stages["ego"][0].weight += 0.01
    mutants.append(m)

    m = base()                                       # multiplicity
    c = m.stages["ego"][0]
    atom = next(iter(c.consume))
    c.consume[atom] += 1
    mutants.append(m)

    m = base()                                       # stage rewire
    m.stages["others"].append(m.stages["ego"].pop(0))
    mutants.append(m)

    m = base()                                       # link rewire
    m.links[0].post = "check"
    mutants.append(m)

    m = base()                                       # init differs
    m.init["st(ego,l0,c0,v1)"] = m.init.pop("st(ego,l0,c1,v1)")
    mutants.append(m)

    m = base()                                       # clause dropped
    m.stages["check"].pop()
    mutants.append(m)

    keys = [program_key(m) for m in mutants]
    assert len(set(keys)) == len(keys)


# -- run identity --------------------------------------------------------

def diamond():
    """ra and rb independent; rc joins their products."""
    return Program(
        stages={"s": [Clause("ra", {"a": 1}, {"a2": 1}),
                      Clause("rb", {"b": 1}, {"b2": 1}),
                      Clause("rc", {"a2": 1, "b2": 1}, {"c": 1})]},
        init_stage="s", init={"a": 1, "b": 1})


def test_run_key_interleaving_invariance_hand():
    d = diamond()
    assert independent_events(d, 0, 1)
    assert not independent_events(d, 0, 2)
    assert run_key(d, [0, 1, 2]) == run_key(d, [1, 0, 2])
    assert run_key(d, [0, 1]) == run_key(d, [1, 0])
    assert run_key(d, [0, 1, 2]) != run_key(d, [0, 1])
    assert run_key(d, [0]) != run_key(d, [1])


def replay(program, trace):
    events = program.events()
    stage, kappa = program.initial_state()
    counts = dict(kappa)
    for e in trace:
        kind, s, ev = events[e]
        assert s == stage, "replay: event outside active stage"
        for q, n in ev.consume.items():
            counts[q] = counts.get(q, 0) - n
            assert counts[q] >= 0, "replay: negative resource"
        for q, n in ev.persist.items():
            assert counts.get(q, 0) >= n, "replay: missing !premise"
        for q, n in ev.produce.items():
            counts[q] = counts.get(q, 0) + n
        counts = {q: n for q, n in counts.items() if n > 0}
        if kind == "link":
            stage = ev.post
    return stage, frozenset(counts.items())


def same_outcome(replayed, final):
    """The sampler's final state may have taken the quiescence step to
    DONE after the last recorded event; replay stops at the last event."""
    stage, counts = replayed
    return counts == final[1] and (stage == final[0]
                                   or final[0] == "(done)")


def test_run_key_property_road():
    """Sampled road runs: adjacent independent swaps keep the key AND
    replay to the same final state; runs with different event multisets
    get different keys."""
    prog = base_catalog().ground(scenario())
    rng = random.Random(5)
    seen = {}
    for _ in range(60):
        trace, final = run_trace(prog, 6, rng)
        base_key = run_key(prog, list(trace))
        assert same_outcome(replay(prog, trace), final)
        variant = list(trace)
        for _ in range(20):
            i = rng.randrange(max(1, len(variant) - 1))
            if i + 1 < len(variant) and \
                    independent_events(prog, variant[i], variant[i + 1]):
                variant[i], variant[i + 1] = variant[i + 1], variant[i]
        assert run_key(prog, variant) == base_key
        assert same_outcome(replay(prog, variant), final)
        # discrimination is by event CONTENT, not index: symmetric
        # ground instances (collide[X=ego,Y=alfa] vs [X=alfa,Y=ego])
        # are the same semantic event and must share a key
        from metis.kernel.nets import _event_key
        table = prog.events()
        content = sorted(repr(_event_key(*table[e])) for e in variant)
        seen.setdefault(tuple(content), set()).add(base_key)
    for content_keys in seen.values():
        assert len(content_keys) == 1    # same events -> same causal key
    assert len({k for ks in seen.values() for k in ks}) == len(seen)


def test_run_key_cross_presentation():
    d = diamond()
    rng = random.Random(9)
    sh = shuffled(d, rng)
    # map events by content (unique here)
    def content(p):
        return {repr((k, s, (dict(ev.consume), dict(ev.produce)))): i
                for i, (k, s, ev) in enumerate(p.events())}
    cd, cs = content(d), content(sh)
    mapped = [cs[k] for k in
              (repr(("clause", "s", ({"a": 1}, {"a2": 1}))),
               repr(("clause", "s", ({"b": 1}, {"b2": 1}))),
               repr(("clause", "s", ({"a2": 1, "b2": 1}, {"c": 1}))))]
    orig = [cd[k] for k in
            (repr(("clause", "s", ({"a": 1}, {"a2": 1}))),
             repr(("clause", "s", ({"b": 1}, {"b2": 1}))),
             repr(("clause", "s", ({"a2": 1, "b2": 1}, {"c": 1}))))]
    assert run_key(d, orig) == run_key(sh, mapped)


# -- projection cache ----------------------------------------------------

def test_cache_exact_on_scene_stream():
    cat = base_catalog()
    configs = [(("l0", f"c{i}", "v1"), ("l0", f"c{j}", "v0"))
               for i in (0, 1, 2) for j in (2, 3)]
    rng = random.Random(21)
    stream = [rng.choice(configs) for _ in range(30)]
    cache = ProjectionCache()
    for ego, alfa in stream:
        cache.project(cat.ground(scenario(ego=ego, alfa=alfa)), 6)
    distinct = len(set(stream))
    assert cache.misses == distinct
    assert cache.hits == 30 - distinct
    # a cached revisit answers state queries identically to a fresh run
    ego, alfa = stream[0]
    prog = cat.ground(scenario(ego=ego, alfa=alfa))
    cached = cache.project(prog, 6)
    fresh = ForwardProjection(prog, 6)
    assert cached.final_distribution("crash") == approx(
        fresh.final_distribution("crash"), abs=1e-12)
    assert cached.p_done() == approx(fresh.p_done(), abs=1e-12)


def test_policy_without_key_not_cached():
    cat = base_catalog()
    prog = cat.ground(scenario())
    cache = ProjectionCache()
    pol = lambda stage, counts, ws: ws
    cache.project(prog, 6, policy=pol)
    cache.project(prog, 6, policy=pol)
    assert cache.hits == 0 and cache.misses == 2
    cache.project(prog, 6, policy=pol, policy_key="id-pol")
    cache.project(prog, 6, policy=pol, policy_key="id-pol")
    assert cache.hits == 1


def test_cached_projector_under_plan():
    """The stage-4 note closed: plan() over a ProjectionCache with a
    CONTENT-keyed policy. The builder runs on the canonical program
    (index safety) and only on misses; a repeated planning round is
    all hits; results are identical to the uncached loop."""
    from metis.kernel import ProjectionCache, cached_projector, plan
    from metis.kernel.sync import sync_policy
    from metis.domains.sysadmin.model import build_program, \
        expected_running, sysadmin_p_on

    comps = ["a", "b"]
    nbrs = {"a": ["b"], "b": ["a"]}
    built = [0]

    def builder(prog):
        built[0] += 1
        return sync_policy(prog, sysadmin_p_on(nbrs))

    cache = ProjectionCache()
    steps = 9

    def run_round(project):
        return plan([None, "b"],
                    do=lambda a: build_program(comps, {"a"}, a),
                    project=project,
                    score=lambda pr: (
                        (0.75 if any(x.startswith("force(")
                                     for x in pr.program.init)
                         else 0.0) - expected_running(pr),),
                    doomed_action=None)

    proj = cached_projector(cache, steps, policy_builder=builder,
                            policy_key=("sysadmin", 2, "v1"))
    r1 = run_round(proj)
    assert (cache.misses, cache.hits) == (2, 0)
    r2 = run_round(proj)
    assert (cache.misses, cache.hits) == (2, 2)
    assert built[0] == 2                     # builder only on misses
    plain = run_round(lambda prog: __import__(
        "metis.kernel", fromlist=["ForwardProjection"]
        ).ForwardProjection(prog, steps, policy=builder(prog)))
    assert r1.action == r2.action == plain.action
    assert r1.scores == r2.scores == plain.scores
