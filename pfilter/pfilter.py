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
    n = np.random.normal(np.zeros(len(sigmas)), sigmas, size=(x.shape[0], len(sigmas)))
    return x+n

class ParticleFilter(object):
    def __init__(self, priors,  inverse_fn, n_particles=200, dynamics_fn=None, noise_fn=None, 
                weight_fn=None,  resample_proportion=0.05, column_names=None, internal_weight_fn=None):
        """
        
        Parameters:
        ---
        
        priors: sequence of prior distributions; should be a frozen distribution from scipy.stats; 
                e.g. scipy.stats.norm(loc=0,scale=1) for unit normal
        inverse_fn: transformation function from the internal state to the sensor state. Takes an (N,D) array of states 
                    and returns the expected sensor output as an array (e.g. a tensor).
        n_particles: number of particles in the filter
        dynamics_fn: dynamics function, which takes a state vector and returns a new one with the dynamics applied.
        noise_fn: noise function, takes a state vector and returns a new one with noise added.
        weight_fn: computes the distance from the real sensed variable and that returned by inverse_fn. Takes
                  a an array of N sensor outputs and the observed output (x,y) and 
                  returns a strictly positive weight for the output. This should be a *similarity* measure, 
                  with higher values meaning more similar        
        internal_weight_fn: reweights the particles based on their *internal* state. This is function which takes
                         an array of internal states and returns weights for each. Typically used to force
                         particles inside of bounds.
        resample_proportion: proportion of samples to draw from the prior on each iteration
        column_names: names of each the columns of the state vector
        
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
        # resample from the prior
        if mask is None:
            for i,prior in enumerate(self.priors):
                self.particles[:,i] = prior.rvs(self.n_particles)
        else:
            for i,prior in enumerate(self.priors):
                self.particles[mask,i] = prior.rvs(self.n_particles)[mask]
    
    def update(self, observed):
        # apply dynamics and noise
        self.particles = self.dynamics_fn(self.particles)
        self.particles = self.noise_fn(self.particles)
        
        # invert to hypothesise observations
        hypotheses = self.inverse_fn(self.particles)
        self.hypotheses = hypotheses
        
        # compute similarity to observations
        weights = self.weight_fn(hypotheses, observed)
                
        # force to be positive and normalise to "probabilities"
        weights = np.clip(weights, 0, np.inf)                
        
        # apply weighting based on the internal state
        if self.internal_weight_fn is not None:
            internal_weights = self.internal_weight_fn(self.particles)            
            internal_weights = np.clip(internal_weights, 0, np.inf)        
            internal_weights = internal_weights / np.sum(internal_weights)
            weights *= internal_weights
            
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
        
