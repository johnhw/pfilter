import numpy as np

# return a new function that has the heat kernel (given by delta) applied.
def make_heat_adjusted(sigma):
    def heat_distance(d):
        return np.exp(-d ** 2 / (2.0 * sigma ** 2))
    return heat_distance


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

# identity functions for clearer naming
identity = lambda x: x


def squared_error(x, y, sigma=1):
    # RBF kernel
    d = np.sum((x - y) ** 2, axis=(1, 2))
    return np.exp(-d / (2.0 * sigma ** 2))


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
    hypotheses : array
        The (N,...) array of hypotheses for each particle
    weights : array
        N-element vector of normalized weights for each particle.
    """

    def __init__(
        self,
        prior_fn,
        observe_fn,
        n_particles=200,
        dynamics_fn=None,
        noise_fn=None,
        weight_fn=None,
        resample_proportion=None,
        column_names=None,
        internal_weight_fn=None,
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
        n_particles : int 
                     number of particles in the filter
        dynamics_fn : function(states) => states
                      dynamics function, which takes an (N,D) state array and returns a new one with the dynamics applied.
        noise_fn : function(states) => states
                    noise function, takes a state vector and returns a new one with noise added.
        weight_fn :  function(real, hypothesized) => weights
                    computes the distance from the real sensed variable and that returned by observe_fn. Takes
                    a an array of N hypothesised sensor outputs (e.g. array of dimension (N,W,H)) and the observed output (e.g. array of dimension (W,H)) and 
                    returns a strictly positive weight for the each hypothesis as an N-element vector. 
                    This should be a *similarity* measure, with higher values meaning more similar, for example from an RBF kernel.
        internal_weight_fn :  function(states, observed) => weights
                    Reweights the particles based on their *internal* state. This is function which takes
                    an (N,D) array of internal states and the observation and 
                    returns a strictly positive weight for the each state as an N-element vector. 
                    Typically used to force particles inside of bounds, etc.                                        
        resample_proportion : float
                    proportion of samples to draw from the initial on each iteration.
        column_names : list of strings
                    names of each the columns of the state vector
        
        """
        self.column_names = column_names
        self.prior_fn = prior_fn                
        self.n_particles = n_particles
        # perform initial sampling
        self.init_filter()

        self.d = self.particles.shape[1]
        self.observe_fn = observe_fn
        self.dynamics_fn = dynamics_fn or identity
        self.noise_fn = noise_fn or identity
        self.weight_fn = weight_fn or squared_error
        self.resample_proportion = resample_proportion or 0.0
        self.particles = np.zeros((self.n_particles, self.d))
        self.internal_weight_fn = internal_weight_fn
        
        self.original_particles = np.array(self.particles)

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
            self.particles[mask,:] = new_sample[mask,:]

    def update(self, observed=None):
        """Update the state of the particle filter given an observation.
        
        Parameters:
        ----------
        
        observed: array
            The observed output, in the same format as observe_fn() will produce. This is typically the
            input from the sensor observing the process (e.g. a camera image in optical tracking).
            If None, then the observation step is skipped, and the filter will run one step in prediction-only mode.
        """

        # apply dynamics and noise
        self.particles = self.dynamics_fn(self.particles)
        self.particles = self.noise_fn(self.particles)

        # hypothesise observations
        self.hypotheses = self.observe_fn(self.particles)

        if observed is not None:
            # compute similarity to observations
            # force to be positive
            weights = np.clip(
                np.array(self.weight_fn(self.hypotheses, observed)), 0, np.inf
            )
        else:
            # we have no observation, so all particles weighted the same
            weights = np.ones((self.n_particles,))

        # apply weighting based on the internal state
        # most filters don't use this, but can be a useful way of combining
        # forward and inverse models
        if self.internal_weight_fn is not None:
            internal_weights = self.internal_weight_fn(self.particles, observed)
            internal_weights = np.clip(internal_weights, 0, np.inf)
            internal_weights = internal_weights / np.sum(internal_weights)
            weights *= internal_weights

        # normalise weights to resampling probabilities
        self.weights = weights / np.sum(weights)

        # resampling (systematic resampling) step
        indices = resample(self.weights)
        self.particles = self.particles[indices, :]

        # store mean (expected) hypothesis
        self.mean_hypothesis = np.sum(self.hypotheses.T * self.weights, axis=-1).T
        self.mean_state = np.sum(self.particles.T * self.weights, axis=-1).T

        # store MAP estimate 
        argmax_weight = np.argmax(self.weights)
        self.map_state = self.particles[argmax_weight]
        self.map_hypothesis = self.hypotheses[argmax_weight]

        # preserve current sample set before any replineshment
        self.original_particles = np.array(self.particles)

        # randomly resample some particles from the prior
        random_mask = (
            np.random.random(size=(self.n_particles,)) < self.resample_proportion
        )
        self.resampled_particles = random_mask
        self.init_filter(mask=random_mask)

