from pfilter import ParticleFilter, squared_error
import numpy as np


def test_init():
    # silly initialisation, but uses all parameters
    pf = ParticleFilter(
        prior_fn=lambda x: np.random.normal(0, 1, (1, 100)),
        observe_fn=lambda x: x,
        n_particles=100,
        dynamics_fn=lambda x: x,
        noise_fn=lambda x: x,
        weight_fn=lambda x: x,
        resample_proportion=0.2,
        column_names=["test"],
        internal_weight_fn=lambda x: x,
        n_eff_threshold=1.0,
    )


def test_squared_error():
    for shape in [1, 1], [1, 10], [10, 1], [10, 10], [200, 10], [10, 200]:
        x = np.random.normal(0, 1, shape)
        y = np.random.normal(0, 1, shape)        
        assert np.allclose(squared_error(x, y, sigma=1), squared_error(x, y))
        assert np.all(squared_error(x, y, sigma=0.5) < squared_error(x, y))
        assert np.all(squared_error(x, y, sigma=2.0) > squared_error(x, y))
    

def test_heat_kernel():
    pass
