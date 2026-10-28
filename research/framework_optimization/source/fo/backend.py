"""Explicit dependency injection; defaults always use the existing FP32 path."""
from dataclasses import dataclass
from functools import partial
from ptcg_lab.rl_core import evaluate_decisions as baseline_evaluate,ppo_step as baseline_step
from .packed import evaluate_decisions
from .precision import Precision
from .ppo import ppo_step


@dataclass(frozen=True)
class Options:
    vectorized: bool=False
    precision: str='fp32'
    archive_queue: bool=False
    immutable_hardlink: bool=False
    workers: int=1


def evaluator(options,device='cpu',trace=None):
    if not options.vectorized and options.precision=='fp32':return baseline_evaluate,None
    if not options.vectorized:raise ValueError('Establish packed FP32 baseline before separate AMP experiment')
    precision=Precision(options.precision,device)
    return partial(evaluate_decisions,precision=precision,trace=trace),precision


def step(options,device='cpu',trace=None):
    evaluate,precision=evaluator(options,device,trace)
    if evaluate is baseline_evaluate:return baseline_step,None
    return partial(ppo_step,evaluator=evaluate,precision=precision),precision
