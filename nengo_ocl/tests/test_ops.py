"""Ensure that OCL ops perform the same operations as Nengo ops."""

import nengo
from nengo.builder.signal import Signal
from nengo.builder.operator import DotInc, ElementwiseInc, PreserveValue, Reset
import numpy as np

import nengo_ocl


def test_dotinc_matrix_vector(rng):
    model = nengo.builder.Model(dt=0.001)
    Y = Signal(rng.rand(5))
    A = Signal(rng.rand(5, 2))
    X = Signal(rng.rand(2))
    model.add_op(PreserveValue(Y))
    model.add_op(DotInc(A, X, Y))
    step1 = np.dot(A.value, X.value) + Y.value
    step2 = np.dot(A.value, X.value) + step1

    sim = nengo_ocl.Simulator(None, model=model)
    sim.step()
    assert np.allclose(sim.signals[Y], step1)
    sim.step()
    assert np.allclose(sim.signals[Y], step2)


def test_dotinc_matrix_matrix(rng):
    model = nengo.builder.Model(dt=0.001)
    Y = Signal(rng.rand(2, 3))
    A = Signal(rng.rand(2, 4))
    X = Signal(rng.rand(4, 3))
    model.add_op(PreserveValue(Y))
    model.add_op(DotInc(A, X, Y))
    step1 = np.dot(A.value, X.value) + Y.value
    step2 = np.dot(A.value, X.value) + step1

    sim = nengo_ocl.Simulator(None, model=model)
    sim.step()
    assert np.allclose(sim.signals[Y], step1)
    sim.step()
    assert np.allclose(sim.signals[Y], step2)


def test_elementwiseinc(rng):
    model = nengo.builder.Model(dt=0.001)
    Y = Signal(rng.rand(2, 4))
    A = Signal(rng.rand(2, 4))
    X = Signal(rng.rand(2, 1))
    model.add_op(Reset(Y))
    model.add_op(ElementwiseInc(A, X, Y))
    step1 = A.value * X.value + Y.value
    step2 = A.value * X.value + step1

    sim = nengo_ocl.Simulator(None, model=model)
    sim.step()
    assert np.allclose(sim.signals[Y], step1)
    sim.step()
    assert np.allclose(sim.signals[Y], step2)


def test_reset(rng):
    model = nengo.builder.Model(dt=0.001)
    sig = Signal(rng.rand(2, 3))
    model.add_op(Reset(sig))

    sim = nengo_ocl.Simulator(None, model=model)
    sim.step()
    assert np.allclose(sim.signals[sig], 0.0)
    sim.step()
    assert np.allclose(sim.signals[sig], 0.0)
