
import nose
import numpy as np

from nengo.objects import LIF, LIFRate, Direct

from ..ra_gemv import ragged_gather_gemv
from  .. import raggedarray as ra
from  ..raggedarray import RaggedArray as RA
from ..clraggedarray import CLRaggedArray as CLRA

from ..clra_nonlinearities import plan_lif

import pyopencl as cl
ctx = cl.create_some_context()

def not_close(a, b, rtol=1e-3, atol=1e-3):
    return np.abs(a - b) > atol + rtol * np.abs(b)

def test_lif_step0():
    test_lif_step(upsample=1, n_elements=0)

def test_lif_step1():
    test_lif_step(upsample=4, n_elements=0)

def test_lif_step2():
    test_lif_step(upsample=4, n_elements=7)

def test_lif_step(upsample=1, n_elements=0):
    dt = 1e-3
    n_neurons = [3, 3, 3]
    # n_neurons = [12345, 23456, 34567]
    N = len(n_neurons)
    J = RA([np.random.normal(scale=1.2, size=n) for n in n_neurons])
    V = RA([np.random.uniform(low=0, high=1, size=n) for n in n_neurons])
    W = RA([np.random.uniform(low=-5*dt, high=5*dt, size=n) for n in n_neurons])
    OS = RA([np.zeros(n) for n in n_neurons])

    ref = 2e-3
    # tau = 20e-3
    # tau_array = RA([tau*np.ones(n) for n in n_neurons])

    # refs = list(np.random.uniform(low=1.7e-3, high=4.2e-3, size=len(n_neurons)))
    taus = list(np.random.uniform(low=15e-3, high=80e-3, size=len(n_neurons)))

    queue = cl.CommandQueue(ctx)
    clJ = CLRA(queue, J)
    clV = CLRA(queue, V)
    clW = CLRA(queue, W)
    clOS = CLRA(queue, OS)
    # clTau = CLRA(queue, tau_array)

    # clRef = CLRA(queue, RA(refs))
    clTau = CLRA(queue, RA(taus))

    ### simulate host
    nls = [LIF(n, tau_ref=ref, tau_rc=taus[i])
           for i, n in enumerate(n_neurons)]
    for i, nl in enumerate(nls):
        if upsample <= 1:
            nl.step_math0(dt, J[i], V[i], W[i], OS[i])
        else:
            s = np.zeros_like(OS[i])
            for j in xrange(upsample):
                nl.step_math0(dt/upsample, J[i], V[i], W[i], s)
                OS[i] = (OS[i] > 0.5) | (s > 0.5)

    ### simulate device
    plan = plan_lif(queue, clJ, clV, clW, clV, clW, clOS, ref, clTau, dt,
                    n_elements=n_elements, upsample=upsample)
    # plan = plan_lif(queue, clJ, clV, clW, clV, clW, clOS, ref, clTau, dt)
    plan()

    if 1:
        a, b = V, clV
        for i in xrange(len(a)):
            nc, _ = not_close(a[i], b[i]).nonzero()
            if len(nc) > 0:
                j = nc[0]
                print "i", i, "j", j
                print "J", J[i][j], clJ[i][j]
                print "V", V[i][j], clV[i][j]
                print "W", W[i][j], clW[i][j]
                print "...", len(nc) - 1, "more"

    print "number of spikes", np.sum([np.sum(OS[i]) for i in xrange(len(OS))])
    assert ra.allclose(J, clJ.to_host())
    assert ra.allclose(V, clV.to_host())
    assert ra.allclose(W, clW.to_host())
    assert ra.allclose(OS, clOS.to_host())

def test_lif_speed():
    # import time

    dt = 1e-3
    # t_final = 1.
    # t = dt * np.arange(np.round(t_final / dt))
    # nt = len(t)

    ref = 2e-3
    tau = 20e-3

    # n_neurons = [1.1e5] * 5
    n_neurons = [1.0e5] * 5 + [1e3]*50
    J = RA([np.random.randn(n) for n in n_neurons])
    V = RA([np.random.uniform(low=0, high=1, size=n) for n in n_neurons])
    W = RA([np.random.uniform(low=-10*dt, high=10*dt, size=n) for n in n_neurons])
    OS = RA([np.zeros(n) for n in n_neurons])

    queue = cl.CommandQueue(
        ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

    clJ = CLRA(queue, J)
    clV = CLRA(queue, V)
    clW = CLRA(queue, W)
    clOS = CLRA(queue, OS)

    n_elements = [0, 2, 5, 10]
    for i, nel in enumerate(n_elements):
        plan = plan_lif(queue, clJ, clV, clW, clV, clW, clOS, ref, tau, dt,
                        n_elements=nel)

        for j in range(1000):
            plan(profiling=True)

        print "plan %d: n_elements = %d" % (i, nel)
        print 'n_calls         ', plan.n_calls
        print 'queued -> submit', plan.atime
        print 'submit -> start ', plan.btime
        print 'start -> end    ', plan.ctime

