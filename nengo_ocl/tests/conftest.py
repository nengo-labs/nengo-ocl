import nengo
import nengo.tests.test_synapses
from nengo.utils.testing import signals_allclose
import pyopencl as cl
import pytest

from pytest_plt.plugin import plt
from pytest_rng.plugin import rng


@pytest.fixture(scope="session")
def ctx(request):
    return cl.create_some_context()


# --- Change allclose tolerences for some Nengo tests
def allclose_tol(*args, **kwargs):
    """Use looser tolerance"""
    kwargs.setdefault('atol', 2e-7)
    return signals_allclose(*args, **kwargs)


nengo.tests.test_synapses.signals_allclose = allclose_tol  # looser tolerances
