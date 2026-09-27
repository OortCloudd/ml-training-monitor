"""Uncertainty of an observed work/time rate, not of hardware FLOP accounting."""
from functools import lru_cache
import math
import random
import statistics

@lru_cache(maxsize=32)
def rate_interval(pairs):
    n=len(pairs)
    if n<50:return {'status':'insufficient_window','minimum_steps':50,'observed_steps':n}
    f=[x[0] for x in pairs];t=[x[1] for x in pairs]
    if any(not math.isfinite(x) or x<=0 for x in f+t):return {'status':'invalid_observations'}
    midpoint=n//2
    first=sum(f[:midpoint])/sum(t[:midpoint]);last=sum(f[midpoint:])/sum(t[midpoint:])
    drift=abs(last-first)/((last+first)/2)
    if drift>.10:return {'status':'nonstationary','half_window_relative_change':drift,'threshold':.10}
    block=max(5,min(10,n//10));rng=random.Random(1729);rates=[]
    # Circular moving-block bootstrap retains short-range serial dependence.
    for _ in range(2000):
        nf=nt=0.;remaining=n
        while remaining:
            start=rng.randrange(n);size=min(block,remaining)
            for j in range(size):
                k=(start+j)%n;nf+=f[k];nt+=t[k]
            remaining-=size
        rates.append(nf/nt)
    rates.sort()
    return {'status':'available','level':.95,'lower_rate':rates[49],'upper_rate':rates[1949],
            'method':'circular_moving_block_bootstrap','replicates':2000,'block_length':block,
            'observed_steps':n,'half_window_relative_change':drift,
            'scope':'Local work/time rate conditional on local stationarity; not FLOP-count or peak uncertainty',
            'systematic_uncertainty_included':False}
