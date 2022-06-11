from pfilter import (
    ParticleFilter,
    squared_error,
    t_noise,
    gaussian_noise,
    cauchy_noise,
    make_heat_adjusted,
    systematic_resample,
    stratified_resample,
    multinomial_resample,
    residual_resample,
)
import numpy as np


def test_init():
    # silly initialisation, but uses all parameters
    pf = ParticleFilter(
        prior_fn=lambda n: np.random.normal(0, 1, (n, 1)),
        observe_fn=lambda x: x,
        n_particles=100,
        dynamics_fn=lambda x: x,
        noise_fn=lambda x: x,
        weight_fn=lambda x, y: np.ones(len(x)),
        resample_proportion=0.2,
        column_names=["test"],
        internal_weight_fn=lambda x, y: np.ones(len(x)),
        n_eff_threshold=1.0,
    )
    pf.update(np.array([1]))


# pure, basic multinomial sampling
def basic_resample(weights):
    return np.random.choice(np.arange(len(weights)), p=weights, size=len(weights))

def test_copy():
    pf = ParticleFilter(
        prior_fn=lambda n: np.random.normal(0, 1, (n, 1)),
        observe_fn=lambda x: x,
        n_particles=100,
        dynamics_fn=lambda x: x,
        noise_fn=lambda x: x,
        weight_fn=lambda x, y: np.ones(len(x)),
        resample_proportion=0.2,
        column_names=["test"],
        internal_weight_fn=lambda x, y: np.ones(len(x)),
        n_eff_threshold=1.0,
    )

    original = np.array(pf.particles)
    original_weights = np.array(pf.weights)

    a_copy = pf.copy()

    pf.update() 

    b_copy = pf.copy()

    pf.update(np.array([1]))

    c_copy = pf.copy()

    assert np.allclose(a_copy.particles, original)
    assert np.allclose(a_copy.weights, original_weights)
    assert np.allclose(c_copy.mean_hypothesis, pf.mean_hypothesis)
    assert not c_copy.mean_hypothesis is pf.mean_hypothesis
    



def test_predict():
    # silly initialisation, but uses all parameters
    pf = ParticleFilter(
        prior_fn=lambda n: np.random.normal(0, 1, (n, 1)),
        observe_fn=lambda x: x,
        n_particles=100,
        dynamics_fn=lambda x: x + 1,
        noise_fn=lambda x: x + np.random.normal(0, 1, x.shape),
        weight_fn=lambda x, y: np.sum(y**2, axis=1),
        resample_proportion=0.2,
        column_names=["test"],
        internal_weight_fn=lambda x, y: np.ones(len(x)),
        n_eff_threshold=1.0,
    )
    old_weights = np.array(pf.original_weights)
    old_particles = np.array(pf.original_particles)
    states = []
    for i in pf.predictor(10):
        states.append([pf.weights, pf.particles])
    
    assert len(states)==10
    # make sure the state hasn't changed
    assert np.allclose(old_weights, pf.original_weights)
    assert np.allclose(old_particles, pf.original_particles)
    pf.update(np.array([1]))
    #assert not np.allclose(old_weights, pf.original_weights)
    assert not np.allclose(old_particles, pf.original_particles)
    
    



def test_resampler():
    pf = ParticleFilter(
        prior_fn=lambda n: np.random.normal(0, 1, (n, 1)),
        n_particles=100,
        resample_fn=None,  # should use default
    )
    for i in range(10):
        pf.update(np.array([1]))

    pf = ParticleFilter(
        prior_fn=lambda n: np.random.normal(0, 1, (n, 1)),
        n_particles=100,
        resample_fn=basic_resample,
    )
    for i in range(10):
        pf.update(np.array([1]))

    for sampler in [
        stratified_resample,
        systematic_resample,
        residual_resample,
        multinomial_resample,
    ]:
        pf = ParticleFilter(
            prior_fn=lambda n: np.random.normal(0, 1, (n, 1)),
            n_particles=100,
            resample_fn=sampler,
        )
        for i in range(10):
            pf.update(np.array([1]))


def test_weights():
    # verify weights sum to 1.0
    pf = ParticleFilter(
        prior_fn=lambda n: np.random.normal(0, 1, (n, 1)), n_particles=100
    )
    for i in range(10):
        pf.update(np.array([1]))
        assert len(pf.weights) == len(pf.particles) == 100
        assert pf.particles.shape == (100, 1)
        assert np.allclose(np.sum(pf.weights), 1.0)


def test_no_observe():
    # check that
    pf = ParticleFilter(
        prior_fn=lambda n: np.random.normal(0, 1, (n, 1)), n_particles=10
    )
    for i in range(10):
        pf.update(None)
        assert len(pf.weights) == len(pf.particles) == 10
        assert pf.particles.shape == (10, 1)
        assert np.allclose(np.sum(pf.weights), 1.0)


import numpy.ma as ma


def test_partial_missing():
    # check that
    pf = ParticleFilter(
        prior_fn=lambda n: np.random.normal(0, 1, (n, 4)), n_particles=100
    )
    for i in range(10):
        masked_input = ma.masked_equal(np.array([1, 999, 0, 999]), 999)
        pf.update(masked_input)
        pf.update(np.array([1, 1, 1, 1]))
        assert np.allclose(np.sum(pf.weights), 1.0)
        assert len(pf.weights) == len(pf.particles) == 100


def test_transform_fn():
    # silly initialisation, but uses all parameters
    pf = ParticleFilter(
        prior_fn=lambda n: np.random.normal(0, 1, (n, 1)),
        observe_fn=lambda x: x,
        n_particles=100,
        dynamics_fn=lambda x: x,
        transform_fn=lambda x, w: 2 * x,
        noise_fn=lambda x: x,
        weight_fn=lambda x, y: np.ones(len(x)),
        resample_proportion=0.2,
        column_names=["test"],
        internal_weight_fn=lambda x, y: np.ones(len(x)),
        n_eff_threshold=1.0,
    )
    for i in range(10):
        pf.update(np.array([1]))
        assert np.allclose(pf.original_particles * 2.0, pf.transformed_particles)


def test_kwargs():
    def check_kwargs(x, **kwargs):
        assert "test_1" in kwargs
        assert "t" in kwargs
        assert kwargs["test_1"] == "ok"
        assert kwargs["t"] == 1.0
        return x

    # silly initialisation, but uses all parameters
    pf = ParticleFilter(
        prior_fn=lambda n: np.random.normal(0, 1, (n, 1)),
        observe_fn=lambda x, **kwargs: check_kwargs(x, **kwargs),
        n_particles=100,
        dynamics_fn=lambda x, **kwargs: check_kwargs(x, **kwargs),
        transform_fn=lambda x, w, **kwargs: check_kwargs(x, **kwargs),
        noise_fn=lambda x, **kwargs: check_kwargs(x, **kwargs),
        weight_fn=lambda x, y, **kwargs: check_kwargs(np.ones(len(x)), **kwargs),
        resample_proportion=0.2,
        column_names=["test"],
        internal_weight_fn=lambda x, y, **kwargs: check_kwargs(
            np.ones(len(x)), **kwargs
        ),
        n_eff_threshold=1.0,
    )
    pf.update(np.array([[1]]), test_1="ok", t=1.0)


def test_gaussian_noise():
    np.random.seed(2012)
    for shape in [10, 10], [100, 1000], [500, 50]:
        val = np.random.normal(0, 10)
        x = np.full(shape, val)
        noisy = gaussian_noise(x, np.ones(shape[1]))
        assert (np.mean(noisy) - np.mean(x)) ** 2 < 1.0
        assert (np.std(noisy) - 1.0) ** 2 < 0.1
        noisy = gaussian_noise(x, np.full(shape[1], 10.0))
        assert (np.std(noisy) - 10.0) ** 2 < 0.1


def test_cauchy_noise():
    np.random.seed(2012)
    for shape in [10, 10], [100, 1000], [500, 50]:
        val = np.random.normal(0, 10)
        x = np.full(shape, val)
        noisy = cauchy_noise(x, np.ones(shape[1]))

def test_t_noise():
    np.random.seed(2012)
    for shape in [10, 10], [100, 1000], [500, 50]:
        val = np.random.normal(0, 10)
        x = np.full(shape, val)         
        noisy = t_noise(x, sigmas=np.ones(shape[1]), df=1.0)
        noisy = t_noise(x, sigmas=np.ones(shape[1]), df=10.0)
        noisy = t_noise(x, sigmas=np.ones(shape[1]), df=0.1)


def test_squared_error():
    for shape in [1, 1], [1, 10], [10, 1], [10, 10], [200, 10], [10, 200]:
        x = np.random.normal(0, 1, shape)
        y = np.random.normal(0, 1, shape)
        assert np.allclose(squared_error(x, y, sigma=1), squared_error(x, y))
        assert np.all(squared_error(x, y, sigma=0.5) < squared_error(x, y))
        assert np.all(squared_error(x, y, sigma=2.0) > squared_error(x, y))


def test_heat_kernel():
    kernel = make_heat_adjusted(1.0)
    assert kernel(0) == 1.0
    assert kernel(1) < 1.0
    assert kernel(1000) < 1e-4
    assert np.allclose(kernel(3), np.exp(-3 ** 2 / 2.0))
    assert kernel(-1) == kernel(1)
    assert kernel(2) < kernel(1)
    a = np.zeros((10, 10))
    b = np.ones((10, 10))
    assert np.all(kernel(a) == 1.0)
    assert np.all(kernel(b) < 1.0)
    kernel_small = make_heat_adjusted(0.5)
    kernel_large = make_heat_adjusted(2.0)
    for k in -10, -5, -1, -0.5, 0.5, 1, 5, 10:
        assert kernel_small(k) < kernel(k) < kernel_large(k)

