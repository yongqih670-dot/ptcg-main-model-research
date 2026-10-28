"""Frozen single-opponent unpaired BO3 contrast for the TR11 diagnosis."""
import numpy as np


def contrast(parent,endpoint):
    parent=np.asarray(parent,dtype=float);endpoint=np.asarray(endpoint,dtype=float)
    if len(parent)!=400 or len(endpoint)!=400:raise ValueError('Exactly400 completeBO3 in each arm required')
    if not np.isin(parent,[0.,.5,1.]).all() or not np.isin(endpoint,[0.,.5,1.]).all():raise ValueError('Scores must be completeBO3 win/draw/loss')
    rng=np.random.default_rng(231102)
    draws=rng.choice(endpoint,(20000,400)).mean(1)-rng.choice(parent,(20000,400)).mean(1)
    # Sum exact half-point scores before division so a true20/400 gain is
    # compared to0.05 without subtraction-induced rounding below the boundary.
    gain=float((endpoint.sum()-parent.sum())/400)
    ci=np.quantile(draws,[.025,.975]).tolist()
    return {'gain':gain,'ci95':ci,'learning_signal_passed':bool(ci[0]>0 and gain>=.05),
        'bootstrap_seed':231102,'bootstrap_samples':20000}
