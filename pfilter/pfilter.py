import numpy as np

# return a new function that has the heat kernel (given by delta) applied.
def make_heat_adjusted(sigma):
    def heat_distance(d):
        return np.exp(-d**2 / (2.0*sigma**2))
    return heat_distance

# resample function from http://scipy-cookbook.readthedocs.io/items/ParticleFilter.html    
def resample(weights):
  n = len(weights)
  indices = []
  C = [0.] + [np.sum(weights[:i+1]) for i in range(n)]
  u0, j = np.random.random(), 0
  for u in [(u0+i)/n for i in range(n)]:
    while u > C[j]:
      j+=1
    indices.append(j-1)
  return indices    
  
def no_dynamics(x):
    return x
    
def no_noise(x):
    return x
    
def squared_error(x,y,sigma=1):
    # RBF kernel
    d = np.sum((x-y)**2, axis=(1,2))
    return np.exp(-d / (2.0*sigma**2))
    
def gaussian_noise(x, sigmas):    
    """Apply normally-distributed noise to the N,D array x.
    Parameters:
    -----------
        x : array
            (N,D) array of values
        sigmas : array
            D-element vector of std. dev. for each column of x
    """
    n = np.random.normal(np.zeros(len(sigmas)), sigmas, size=(x.shape[0], len(sigmas)))
    return x+n

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
    mean_hypothesis : array 
        The current mean hypothesized observation
    mean_state : array
        The current mean hypothesized internal state D
    hypotheses : array
        The (N,...) array of hypotheses for each particle
    weights : array
        N-element vector of normalized weights for each particle.
    """
    
    def __init__(self, priors,  inverse_fn, n_particles=200, dynamics_fn=None, noise_fn=None, 
                weight_fn=None,  resample_proportion=0.05, column_names=None, internal_weight_fn=None):
        """
        
        Parameters:
        -----------
        
        priors : list
                sequence of prior distributions; should be a frozen distribution from scipy.stats; 
                e.g. scipy.stats.norm(loc=0,scale=1) for unit normal
        inverse_fn : function(states) => observations
                    transformation function from the internal state to the sensor state. Takes an (N,D) array of states 
                    and returns the expected sensor output as an array (e.g. a (N,W,H) tensor if generating W,H dimension images).
        n_particles : int 
                     number of particles in the filter
        dynamics_fn : function(states) => states
                      dynamics function, which takes an (N,D) state array and returns a new one with the dynamics applied.
        noise_fn : function(states) => states
                    noise function, takes a state vector and returns a new one with noise added.
        weight_fn :  function(real, hypothesized) => weights
                    computes the distance from the real sensed variable and that returned by inverse_fn. Takes
                    a an array of N hypothesised sensor outputs (e.g. array of dimension (N,W,H)) and the observed output (e.g. array of dimension (W,H)) and 
                    returns a strictly positive weight for the each hypothesis as an N-element vector. 
                    This should be a *similarity* measure, with higher values meaning more similar, for example from an RBF kernel.
        internal_weight_fn :  function(states, observed) => weights
                    Reweights the particles based on their *internal* state. This is function which takes
                    an (N,D) array of internal states and the observation and 
                    returns a strictly positive weight for the each state as an N-element vector. 
                    Typically used to force particles inside of bounds, etc.                                        
        resample_proportion : float
                    proportion of samples to draw from the prior on each iteration.
        column_names : list of strings
                    names of each the columns of the state vector
        
        """
        self.column_names = column_names
        self.priors = priors
        self.d = len(self.priors)
        self.n_particles = n_particles
        self.inverse_fn = inverse_fn
        self.dynamics_fn = dynamics_fn or no_dynamics
        self.noise_fn = noise_fn or no_noise
        self.weight_fn = weight_fn or squared_error
        self.resample_proportion = resample_proportion
        self.particles = np.zeros((self.n_particles, self.d))
        self.internal_weight_fn = internal_weight_fn
        
    def init_filter(self, mask=None):
        """Initialise the filter by drawing samples from the prior.
        
        Parameters:
        -----------
        mask : array, optional
            boolean mask specifying the elements of the particle array to draw from the prior. None (default)
            implies all particles will be resampled (i.e. a complete reset)
        """
        
        # resample from the prior
        if mask is None:
            for i,prior in enumerate(self.priors):
                self.particles[:,i] = prior.rvs(self.n_particles)
        else:
            for i,prior in enumerate(self.priors):
                self.particles[mask,i] = prior.rvs(self.n_particles)[mask]
    
    def update(self, observed=None):
        """Update the state of the particle filter given an observation.
        
        Parameters:
        ----------
        
        observed: array
            The observed output, in the same format as inverse_fn() will produce. This is typically the
            input from the sensor observing the process (e.g. a camera image in optical tracking).
            If None, then the observation step is skipped, and the filter will run one step in prediction-only mode.
        """
            
        # apply dynamics and noise
        self.particles = self.dynamics_fn(self.particles)
        self.particles = self.noise_fn(self.particles)
        # invert to hypothesise observations
        self.hypotheses = self.inverse_fn(self.particles)             
        
        if observed is not None:
            # compute similarity to observations
            # force to be positive 
            weights = np.clip(self.weight_fn(self.hypotheses, observed), 0, np.inf)                   
        else:
            # we have no observation, so all particles weighted the same
            weights = np.ones((self.n_particles,))
        
        # apply weighting based on the internal state
        if self.internal_weight_fn is not None:
            internal_weights = self.internal_weight_fn(self.particles, observed)            
            internal_weights = np.clip(internal_weights, 0, np.inf)        
            internal_weights = internal_weights / np.sum(internal_weights)
            weights *= internal_weights
            
        # normalise probabilities to "probabilities"
        self.weights = weights / np.sum(weights)
                      
        # resampling step
        indices = resample(self.weights)
        self.particles = self.particles[indices, :]
        
        # mean hypothesis
        self.mean_hypothesis = np.sum(self.hypotheses.T * self.weights, axis=-1).T
        self.mean_state = np.sum(self.particles.T * self.weights, axis=-1).T
        
        # randomly resample some particles from the prior
        random_mask = np.random.random(size=(self.n_particles,))<self.resample_proportion
        self.resampled_particles = random_mask
        self.init_filter(mask=random_mask)
        
