import numpy as np
import numpy.ma as ma

# return a new function that has the heat kernel (given by delta) applied.
def make_heat_adjusted(sigma):
    def heat_distance(d):
        return np.exp(-(d**2) / (2.0 * sigma**2))

    return heat_distance


## Resampling based on the examples at: https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/12-Particle-Filters.ipynb
## originally by Roger Labbe, under an MIT License
def systematic_resample(weights):
    n = len(weights)
    positions = (np.arange(n) + np.random.uniform(0, 1)) / n
    return create_indices(positions, weights)


def stratified_resample(weights):
    n = len(weights)
    positions = (np.random.uniform(0, 1, n) + np.arange(n)) / n
    return create_indices(positions, weights)


def residual_resample(weights):
    n = len(weights)
    indices = np.zeros(n, np.uint32)
    # take int(N*w) copies of each weight
    num_copies = (n * weights).astype(np.uint32)
    k = 0
    for i in range(n):
        for _ in range(num_copies[i]):  # make n copies
            indices[k] = i
            k += 1
    # use multinormial resample on the residual to fill up the rest.
    residual = weights - num_copies  # get fractional part
    residual /= np.sum(residual)
    cumsum = np.cumsum(residual)
    cumsum[-1] = 1
    indices[k:n] = np.searchsorted(cumsum, np.random.uniform(0, 1, n - k))
    return indices


def create_indices(positions, weights):
    n = len(weights)
    indices = np.zeros(n, np.uint32)
    cumsum = np.cumsum(weights)
    i, j = 0, 0
    while i < n:
        if positions[i] < cumsum[j]:
            indices[i] = j
            i += 1
        else:
            j += 1

    return indices


### end rlabbe's resampling functions


def multinomial_resample(weights):
    return np.random.choice(np.arange(len(weights)), p=weights, size=len(weights))


# resample function from http://scipy-cookbook.readthedocs.io/items/ParticleFilter.html
def resample(weights):
    n = len(weights)
    indices = []
    C = [0.0] + [np.sum(weights[: i + 1]) for i in range(n)]
    u0, j = np.random.random(), 0
    for u in [(u0 + i) / n for i in range(n)]:
        while u > C[j]:
            j += 1
        indices.append(j - 1)
    return indices


# identity function for clearer naming
identity = lambda x: x


def squared_error(x, y, sigma=1):
    """
    RBF kernel, supporting masked values in the observation
    Parameters:
    -----------
    x : array (N,D) array of values
    y : array (N,D) array of values

    Returns:
    -------

    distance : scalar
        Total similarity, using equation:

            d(x,y) = e^((-1 * (x - y) ** 2) / (2 * sigma ** 2))

        summed over all samples. Supports masked arrays.
    """
    dx = (x - y) ** 2
    d = np.ma.sum(dx, axis=1)
    return np.exp(-d / (2.0 * sigma**2))


def gaussian_noise(x, sigmas):
    """Apply diagonal covaraiance normally-distributed noise to the N,D array x.
    Parameters:
    -----------
        x : array
            (N,D) array of values
        sigmas : array
            D-element vector of std. dev. for each column of x
    """
    n = np.random.normal(np.zeros(len(sigmas)), sigmas, size=(x.shape[0], len(sigmas)))
    return x + n


def t_noise(x, sigmas, df=1.0):
    """Apply diagonal covaraiance t-distributed noise to the N,D array x.
    Parameters:
    -----------
        x : array
            (N,D) array of values
        sigmas : array
            D-element vector of std. dev. for each column of x
        df : degrees of freedom (shape of the t distribution)
            Must be a scalar
    """
    n = np.random.standard_t(df, size=(x.shape[0], len(sigmas))) * sigmas
    return x + n


def cauchy_noise(x, sigmas):
    """Apply diagonal covaraiance Cauchy-distributed noise to the N,D array x.
    Parameters:
    -----------
        x : array
            (N,D) array of values
        sigmas : array
            D-element vector of std. dev. for each column of x
    """
    n = np.random.standard_cauchy(size=(x.shape[0], len(sigmas))) * np.array(sigmas)
    return x + n


def independent_sample(fn_list):
    """Take a list of functions that each draw n samples from a distribution
    and concatenate the result into an n, d matrix
    Parameters:
    -----------
        fn_list: list of functions
                A list of functions of the form `sample(n)` that will take n samples
                from a distribution.
    Returns:
    -------
        sample_fn: a function that will sample from all of the functions and concatenate
        them
    """

    def sample_fn(n):
        return np.stack([fn(n) for fn in fn_list]).T

    return sample_fn


class ParticleFilter(object):
    """A particle filter object which maintains the internal state of a population of particles, and can
    be updated given observations.

    Attributes:
    -----------

    n_particles : int
        number of particles used (N)
    d : int
        dimension of the internal state
    resample_proportion : float
        fraction of particles resampled from prior at each step
    particles : array
        (N,D) array of particle states
    original_particles : array
        (N,D) array of particle states *before* any random resampling replenishment
        This should be used for any computation on the previous time step (e.g. computing
        expected values, etc.)
    mean_hypothesis : array
        The current mean hypothesized observation
    mean_state : array
        The current mean hypothesized internal state D
    map_hypothesis:
        The current most likely hypothesized observation
    map_state:
        The current most likely hypothesized state
    n_eff:
        Normalized effective sample size, in range 0.0 -> 1.0
    weight_informational_energy:
        Informational energy of the distribution (Onicescu's)
    weight_entropy:
        Entropy of the weight distribution (in nats)
    hypotheses : array
        The (N,...) array of hypotheses for each particle
    weights : array
        N-element vector of normalized weights for each particle.
    """

    def __init__(
        self,
        prior_fn,
        observe_fn=None,
        resample_fn=None,
        n_particles=200,
        dynamics_fn=None,
        noise_fn=None,
        weight_fn=None,
        resample_proportion=None,
        column_names=None,
        internal_weight_fn=None,
        transform_fn=None,
        n_eff_threshold=1.0,
    ):
        """

        Parameters:
        -----------

        prior_fn : function(n) = > states
                a function that generates N samples from the prior over internal states, as
                an (N,D) particle array
        observe_fn : function(states) => observations
                    transformation function from the internal state to the sensor state. Takes an (N,D) array of states
                    and returns the expected sensor output as an array (e.g. a (N,W,H) tensor if generating W,H dimension images).
        resample_fn: A resampling function weights (N,) => indices (N,)
        n_particles : int
                     number of particles in the filter
        dynamics_fn : function(states) => states
                      dynamics function, which takes an (N,D) state array and returns a new one with the dynamics applied.
        noise_fn : function(states) => states
                    noise function, takes a state vector and returns a new one with noise added.
        weight_fn :  function(hypothesized, real) => weights
                    computes the distance from the real sensed variable and that returned by observe_fn. Takes
                    a an array of N hypothesised sensor outputs (e.g. array of dimension (N,W,H)) and the observed output (e.g. array of dimension (W,H)) and
                    returns a strictly positive weight for the each hypothesis as an N-element vector.
                    This should be a *similarity* measure, with higher values meaning more similar, for example from an RBF kernel.
        internal_weight_fn :  function(states, observed) => weights
                    Reweights the particles based on their *internal* state. This is function which takes
                    an (N,D) array of internal states and the observation and
                    returns a strictly positive weight for the each state as an N-element vector.
                    Typically used to force particles inside of bounds, etc.
        transform_fn: function(states, weights) => transformed_states
                    Applied at the very end of the update step, if specified. Updates the attribute
                    `transformed_particles`. Useful when the particle state needs to be projected
                    into a different space.
        resample_proportion : float
                    proportion of samples to draw from the initial on each iteration.
        n_eff_threshold=1.0: float
                    effective sample size at which resampling will be performed (0.0->1.0). Values
                    <1.0 will allow samples to propagate without the resampling step until
                    the effective sample size (n_eff) drops below the specified threshold.
        column_names : list of strings
                    names of each the columns of the state vector

        """
        self.resample_fn = resample_fn or resample
        self.column_names = column_names
        self.prior_fn = prior_fn
        self.n_particles = n_particles
        self.init_filter()
        self.n_eff_threshold = n_eff_threshold
        self.d = self.particles.shape[1]
        self.observe_fn = observe_fn or identity
        self.dynamics_fn = dynamics_fn or identity
        self.noise_fn = noise_fn or identity
        self.weight_fn = weight_fn or squared_error
        self.weights = np.ones(self.n_particles) / self.n_particles
        self.transform_fn = transform_fn
        self.transformed_particles = None
        self.resample_proportion = resample_proportion or 0.0
        self.internal_weight_fn = internal_weight_fn
        self.original_particles = np.array(self.particles)
        self.original_weights = np.array(self.weights)

    def copy(self):
        """Copy this filter at its current state. Returns
        an exact copy, that can be run forward indepedently of the first.
        Beware that if your passed in functions (e.g. dynamics) are stateful, behaviour
        might not be independent! (tip: write stateless functions!)

        Returns:
        ---------
            A new, independent copy of this filter.
        """
        # construct the filter
        new_copy = ParticleFilter(
            observe_fn=self.observe_fn,
            resample_fn=self.resample_fn,
            n_particles=self.n_particles,
            prior_fn=self.prior_fn,
            dynamics_fn=self.dynamics_fn,
            weight_fn=self.weight_fn,
            resample_proportion=self.resample_proportion,
            column_names=self.column_names,
            internal_weight_fn=self.internal_weight_fn,
            transform_fn=self.transform_fn,
            n_eff_threshold=self.n_eff_threshold,
        )

        # copy particle state
        for array in ["particles", "original_particles", "original_weights", "weights"]:
            setattr(new_copy, array, np.array(getattr(self, array)))

        # copy any attributes
        for array in [
            "mean_hypothesis",
            "mean_state",
            "map_state",
            "map_hypothesis",
            "hypotheses",
            "n_eff",
            "weight_informational_energy",
            "weight_entropy",
        ]:
            if hasattr(self, array):
                setattr(new_copy, array, getattr(self, array).copy())

        return new_copy

    def predictor(self, n=None, observed=None):
        """Return an generator that runs a copy of the filter forward for prediction.
        Yields the copied filter object at each step. Useful for making predictions
        without inference.

        By default, the filter will run without observations. Pass observed to set the initial observation.
        Use send() to send new observations to the filter. If no send() is used on any given iteration, the filter
        will revert to prediction without observation.

        If n is specified, runs for n steps; otherwise, runs forever.

        Parameters:
        ----------

        n: integer
            Number of steps to run for. If None, run forever.

        observed: array
            The initial observed output, in the same format as observe_fn() will produce. This is typically the
            input from the sensor observing the process (e.g. a camera image in optical tracking).
            If None, then the observation step is skipped

        """
        copy = self.copy()
        observed = None
        if n is not None:
            for i in range(n):
                copy.update(observed)
                observed = yield copy
        else:
            while True:
                copy.update(observed)
                observed = yield copy


    def init_filter(self, mask=None):
        """Initialise the filter by drawing samples from the prior.

        Parameters:
        -----------
        mask : array, optional
            boolean mask specifying the elements of the particle array to draw from the prior. None (default)
            implies all particles will be resampled (i.e. a complete reset)
        """
        new_sample = self.prior_fn(self.n_particles)

        # resample from the prior
        if mask is None:
            self.particles = new_sample
        else:
            self.particles[mask, :] = new_sample[mask, :]

    def update(self, observed=None, **kwargs):
        """Update the state of the particle filter given an observation.

        Parameters:
        ----------

        observed: array
            The observed output, in the same format as observe_fn() will produce. This is typically the
            input from the sensor observing the process (e.g. a camera image in optical tracking).
            If None, then the observation step is skipped, and the filter will run one step in prediction-only mode.

        kwargs: any keyword arguments specified will be passed on to:
            observe_fn(y, **kwargs)
            weight_fn(x, **kwargs)
            dynamics_fn(x, **kwargs)
            noise_fn(x, **kwargs)
            internal_weight_function(x, y, **kwargs)
            transform_fn(x, **kwargs)
        """

        # apply dynamics and noise
        self.particles = self.noise_fn(
            self.dynamics_fn(self.particles, **kwargs), **kwargs
        )

        # hypothesise observations
        self.hypotheses = self.observe_fn(self.particles, **kwargs)

        if observed is not None:
            # compute similarity to observations
            # force to be positive
            if type(observed)==list or  type(observed)==tuple or type(observed)==float or type(observed)==int:
                observed = np.array(observed, dtype=np.float64)

            weights = np.clip(
                self.weights
                * np.array(
                    self.weight_fn(
                        self.hypotheses.reshape(self.n_particles, -1),
                        observed.reshape(1, -1),
                        **kwargs
                    )
                ),
                0,
                np.inf,
            )
        else:
            # we have no observation, so all particles weighted the same
            weights = self.weights * np.ones((self.n_particles,))

        # apply weighting based on the internal state
        # most filters don't use this, but can be a useful way of combining
        # forward and inverse models
        if self.internal_weight_fn is not None:
            internal_weights = self.internal_weight_fn(
                self.particles, observed, **kwargs
            )
            internal_weights = np.clip(internal_weights, 0, np.inf)
            internal_weights = internal_weights / np.sum(internal_weights)
            weights *= internal_weights

        # normalise weights to resampling probabilities
        self.weight_normalisation = np.sum(weights)
        self.weights = weights / self.weight_normalisation

        # Compute effective sample size and entropy of weighting vector.
        # These are useful statistics for adaptive particle filtering.
        self.n_eff = (1.0 / np.sum(self.weights**2)) / self.n_particles
        self.weight_informational_energy = np.sum(self.weights**2)
        self.weight_entropy = np.sum(self.weights * np.log(self.weights))

        # preserve current sample set before any replenishment
        self.original_particles = np.array(self.particles)

        # store mean (expected) hypothesis
        self.mean_hypothesis = np.sum(self.hypotheses.T * self.weights, axis=-1).T
        self.mean_state = np.sum(self.particles.T * self.weights, axis=-1).T
        self.cov_state = np.cov(self.particles, rowvar=False, aweights=self.weights)

        # store MAP estimate
        argmax_weight = np.argmax(self.weights)
        self.map_state = self.particles[argmax_weight]
        self.map_hypothesis = self.hypotheses[argmax_weight]
        self.original_weights = np.array(self.weights)  # before any resampling

        # apply any post-processing
        if self.transform_fn:
            self.transformed_particles = self.transform_fn(
                self.original_particles, self.weights, **kwargs
            )
        else:
            self.transformed_particles = self.original_particles

        # resampling (systematic resampling) step
        if self.n_eff < self.n_eff_threshold:
            indices = self.resample_fn(self.weights)
            self.particles = self.particles[indices, :]
            self.weights = np.ones(self.n_particles) / self.n_particles

        # randomly resample some particles from the prior
        if self.resample_proportion > 0:
            random_mask = (
                np.random.random(size=(self.n_particles,)) < self.resample_proportion
            )
            self.resampled_particles = random_mask
            self.init_filter(mask=random_mask)
