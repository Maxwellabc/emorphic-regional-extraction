"""Synthetic-only semantics and lazy-construction tests; no benchmark timing."""
import inspect
import random
from types import FunctionType
import unittest
from unittest.mock import Mock, patch

from research.egraph_lab import EGraph, Node, evaluate
from research.feedback_neighborhood import sample_region_candidates as baseline
from research.feedback_neighborhood_filtered import sample_region_candidates as eager
from research.region_cycle_filter import FrozenRegionCycleFilter
from research.region_cycle_scc_reference import FrozenRegionSCCCycleFilter
from research import feedback_neighborhood_lazy as lazy


def graph(edges, roots=("a",)):
    nodes={n:Node(n,c,"synthetic",tuple(children),float(i+1))
           for c,members in edges.items() for i,(n,children) in enumerate(members.items())}
    return EGraph(nodes,{c:tuple(members) for c,members in edges.items()},tuple(roots))


def eager_scc():
    namespace=dict(eager.__globals__,FrozenRegionCycleFilter=FrozenRegionSCCCycleFilter)
    result=FunctionType(eager.__code__,namespace,eager.__name__,eager.__defaults__,eager.__closure__)
    result.__kwdefaults__=dict(eager.__kwdefaults__ or {})
    return result


BACKENDS=(eager,eager_scc(),lazy.sample_region_candidates,lazy.sample_region_candidates_scc)


class LazyNeighborhoodTests(unittest.TestCase):
    def compare(self,g,selection,*,seed=0,size=32,count=3,max_attempts=256,seen_signatures=()):
        old_rng=random.Random(seed)
        kwargs=dict(size=size,count=count,max_attempts=max_attempts,seen_signatures=seen_signatures)
        expected=baseline(g,selection,rng=old_rng,**kwargs)
        original=dict(selection)
        for backend in BACKENDS:
            rng=random.Random(seed)
            result=backend(g,selection,rng=rng,**kwargs)
            self.assertEqual(result,expected,backend.__module__)
            self.assertEqual(rng.getstate(),old_rng.getstate())
            self.assertEqual(selection,original)
        return expected

    def test_signature_and_scc_share_function_code_without_global_mutation(self):
        self.assertEqual(inspect.signature(lazy.sample_region_candidates),inspect.signature(baseline))
        self.assertEqual(inspect.signature(lazy.sample_region_candidates_scc),inspect.signature(baseline))
        self.assertIs(lazy.sample_region_candidates.__code__,lazy.sample_region_candidates_scc.__code__)
        self.assertIsNot(lazy.sample_region_candidates.__globals__,lazy.sample_region_candidates_scc.__globals__)
        self.assertIs(lazy.sample_region_candidates.__globals__["FrozenRegionCycleFilter"],FrozenRegionCycleFilter)
        self.assertIs(lazy.sample_region_candidates_scc.__globals__["FrozenRegionCycleFilter"],FrozenRegionSCCCycleFilter)

    def test_no_compiler_or_precheck_for_zero_invalid_draws_including_duplicates(self):
        g=graph({"a":{f"a{i}":() for i in range(7)}})
        center={"a":"a0"}
        signatures=[evaluate(g,{"a":f"a{i}"}).signature for i in range(1,7)]
        for history in ((),signatures):
            constructor=Mock(side_effect=AssertionError("compiler must remain unused"))
            backend=lazy.with_cycle_filter(constructor)
            rng=random.Random(9); expected_rng=random.Random(9)
            actual=backend(g,center,size=32,count=99,rng=rng,seen_signatures=history)
            expected=baseline(g,center,size=32,count=99,rng=expected_rng,seen_signatures=history)
            self.assertEqual(actual,expected)
            self.assertEqual(actual.stats.invalid,0)
            self.assertEqual(rng.getstate(),expected_rng.getstate())
            constructor.assert_not_called()
        self.assertEqual(actual.stats.duplicates,6)

    def test_one_construction_after_first_full_evaluate_cycle_and_only_later_prechecks(self):
        g=graph({"a":{"a0":(),"a1":("a",),"a2":("a",),"a3":(),"a4":()}})
        center={"a":"a0"}
        for base in (FrozenRegionCycleFilter,FrozenRegionSCCCycleFilter):
            events=[]
            def checked_evaluate(g,selection):
                events.append(("evaluate",selection["a"]))
                return evaluate(g,selection)
            class Counted(base):
                def __init__(self,*args): events.append(("compile",None));super().__init__(*args)
                def has_reachable_cycle(self,overrides):
                    events.append(("precheck",overrides["a"]));return super().has_reachable_cycle(overrides)
            with patch.object(lazy,"evaluate",checked_evaluate):
                backend=lazy.with_cycle_filter(Counted)
                result=backend(g,center,size=32,count=99,rng=random.Random(0))
            self.assertEqual(result.stats.invalid,2)
            self.assertEqual(sum(kind=="compile" for kind,_ in events),1)
            compiled=next(i for i,item in enumerate(events) if item[0]=="compile")
            self.assertIn(events[compiled-1],(("evaluate","a1"),("evaluate","a2")))
            self.assertFalse(any(kind=="precheck" for kind,_ in events[:compiled]))
            self.assertTrue(any(kind=="precheck" for kind,_ in events[compiled+1:]))
        self.compare(g,center,count=99)

    def test_no_constructor_if_first_cycle_exhausts_space_or_attempt_limit(self):
        examples=((graph({"a":{"a0":(),"a1":("a",)}}),256),
                  (graph({"a":{"a0":(),"a1":("a",),"a2":("a",)}}),1))
        for g,limit in examples:
            constructor=Mock(side_effect=AssertionError("no later attempt exists"))
            result=lazy.with_cycle_filter(constructor)(g,{"a":"a0"},size=2,count=3,
                                                       max_attempts=limit,rng=random.Random(0))
            self.assertEqual(result.stats.attempted,1)
            self.assertEqual(result.stats.invalid,1)
            constructor.assert_not_called()
            self.compare(g,{"a":"a0"},size=2,max_attempts=limit)

    def test_latent_frozen_cycles_multiple_roots_and_shared_dependencies(self):
        g=graph({"a":{"a0":(),"a1":("x",),"a2":()},
                 "s":{"s0":("b",)},"b":{"b0":(),"b1":("b",),"b2":()},
                 "x":{"x0":("y",)},"y":{"y0":("x",)}},("a","s"))
        center={c:ns[0] for c,ns in g.classes.items()}
        for seed in range(12):
            for size in (2,8,32): self.compare(g,center,seed=seed,size=size,count=9)

    def test_joint_regional_cycle_through_frozen_paths(self):
        g=graph({"r":{"r0":("a","b")},"a":{"a0":(),"a1":("x",),"a2":("b",)},
                 "b":{"b0":(),"b1":("y",),"b2":()},"x":{"x0":("b",)},"y":{"y0":("a",)}},("r",))
        center={c:ns[0] for c,ns in g.classes.items()}
        for seed in range(8):
            result=self.compare(g,center,seed=seed,size=2,count=99)
            self.assertEqual(set(result.region),{"a","b"})
            self.assertEqual(result.stats.invalid,2)

    def test_random_well_formed_multiroot_graphs_match_full_result_and_rng(self):
        rng=random.Random(51351); invalid=duplicates=0
        for trial in range(100):
            names=tuple(f"c{i}" for i in range(rng.randrange(2,13)))
            edges={c:{f"{c}.0":(),**{f"{c}.{j}":tuple(rng.choice(names) for _ in range(rng.randrange(4)))
                    for j in range(1,rng.randrange(2,5))}} for c in names}
            g=graph(edges,tuple(rng.sample(names,rng.randrange(1,min(3,len(names))+1))))
            center={c:ns[0] for c,ns in g.classes.items()}
            for size in (2,8,32):
                first=self.compare(g,center,seed=trial,size=size,count=1+trial%4,max_attempts=1+trial%25)
                second=self.compare(g,center,seed=trial,size=size,count=1+trial%4,max_attempts=1+trial%25,
                                    seen_signatures=[c.metrics.signature for c in first.candidates])
                invalid+=first.stats.invalid+second.stats.invalid
                duplicates+=second.stats.duplicates
        self.assertGreater(invalid,50)
        self.assertGreater(duplicates,0)

    def test_no_decisions_and_invalid_arguments_do_not_compile_or_consume_rng(self):
        g=graph({"a":{"a0":()}}); center={"a":"a0"}
        constructor=Mock(side_effect=AssertionError("unexpected compiler"))
        backend=lazy.with_cycle_filter(constructor)
        for kwargs in ({"size":True},{"size":3},{"size":2,"count":True},{"size":2,"max_attempts":0}):
            rng=random.Random(7); before=rng.getstate()
            with self.assertRaises(ValueError): backend(g,center,rng=rng,**kwargs)
            self.assertEqual(rng.getstate(),before)
        rng=random.Random(7); before=rng.getstate()
        value=backend(g,center,size=32,rng=rng)
        self.assertEqual(value.stats.stop_reason,"no_reachable_decisions")
        self.assertEqual(rng.getstate(),before)
        constructor.assert_not_called()

    def test_noncycle_errors_are_not_hidden_or_used_as_compile_trigger(self):
        g=graph({"a":{"a0":(),"a1":()}})
        constructor=Mock(side_effect=AssertionError("not a cycle"))
        with patch.object(lazy,"evaluate",side_effect=ValueError("missing choice for eclass named cycle")):
            backend=lazy.with_cycle_filter(constructor)
            with self.assertRaisesRegex(ValueError,"missing choice"):
                backend(g,{"a":"a0"},size=2,rng=random.Random(0))
        constructor.assert_not_called()

    def test_compiler_errors_propagate_and_false_acceptance_is_an_implementation_error(self):
        g=graph({"a":{"a0":(),"a1":("a",),"a2":("a",)}})
        constructor=Mock(side_effect=RuntimeError("compiler failed"))
        with self.assertRaisesRegex(RuntimeError,"compiler failed"):
            lazy.with_cycle_filter(constructor)(g,{"a":"a0"},size=2,count=99,rng=random.Random(0))
        constructor.assert_called_once()
        class FalseAcceptance:
            def __init__(self,*args): pass
            def has_reachable_cycle(self,overrides): return False
        with self.assertRaisesRegex(AssertionError,"admitted"):
            lazy.with_cycle_filter(FalseAcceptance)(g,{"a":"a0"},size=2,count=99,rng=random.Random(0))
        class CheckerFailure:
            def __init__(self,*args): pass
            def has_reachable_cycle(self,overrides): raise RuntimeError("checker failed")
        with self.assertRaisesRegex(RuntimeError,"checker failed"):
            lazy.with_cycle_filter(CheckerFailure)(g,{"a":"a0"},size=2,count=99,rng=random.Random(0))


if __name__=="__main__":
    unittest.main()
