#!/usr/bin/env python

import cProfile
import io
import itertools
import pstats
from argparse import ArgumentParser
from pprint import pprint
from timeit import timeit

import boost_histogram as bh
import numpy as np
from memory_profiler import profile

from bootstraphistogram import BootstrapHistogram


def _main():
    args = _parsecml()
    if args.all:
        summary = {}
        for withweight, withseed, fast in itertools.product([True, False], repeat=3):
            if args.profile:
                t = _profile(
                    withweight=withweight,
                    withseed=withseed,
                    fast=fast,
                    numsamples=args.numsamples,
                    numfills=args.numfills,
                    arraysize=args.arraysize,
                )
            else:
                t = _run(
                    withweight=withweight,
                    withseed=withseed,
                    fast=fast,
                    numsamples=args.numsamples,
                    numfills=args.numfills,
                    arraysize=args.arraysize,
                )
            summary[f"withweight={withweight}, withseed={withseed}, fast={fast}"] = t
        pprint(list(sorted((v, k) for k, v in summary.items())))
    else:
        t = _run(
            withweight=args.withweight,
            withseed=args.withseed,
            fast=not args.slow,
            numsamples=args.numsamples,
            numfills=args.numfills,
            arraysize=args.arraysize,
        )
        print(t)
    return


def _parsecml():
    parser = ArgumentParser()
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--withseed", action="store_true")
    parser.add_argument("--withweight", action="store_true")
    parser.add_argument("--slow", action="store_true")
    parser.add_argument("--numsamples", type=int, default=10000)
    parser.add_argument("--numfills", type=int, default=5)
    parser.add_argument("--arraysize", type=int, default=100)
    return parser.parse_args()


def _profile(**kwargs) -> float:
    print(f"--- Profiling {list(sorted(kwargs.items()))}")
    pr = cProfile.Profile()
    pr.enable()
    t = _run_with_memory_profiler(**kwargs)
    pr.disable()
    s = io.StringIO()
    for sortby in ["cumtime", "time"]:
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats(20)
    print(s.getvalue())
    return t


def _run(
    withweight: bool,
    withseed: bool,
    numsamples: int = 10000,
    arraysize: int = 100,
    numfills: int = 5,
    fast: bool = True,
) -> float:
    hist = BootstrapHistogram(
        bh.axis.Regular(100, 0.0, 1.0), numsamples=numsamples, rng=1234
    )
    data = np.random.uniform(size=arraysize)
    weight = np.random.uniform(size=arraysize) if withweight else None
    seed = np.arange(arraysize) if withseed else None
    if fast:
        return timeit(
            lambda: hist._fill_fast(data, weight=weight, seed=seed), number=numfills
        )
    else:
        return timeit(
            lambda: hist._fill_slow(data, weight=weight, seed=seed), number=numfills
        )


@profile
def _run_with_memory_profiler(*args, **kwargs) -> float:
    return _run(*args, **kwargs)


if __name__ == "__main__":
    _main()
