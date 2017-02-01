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
    d = np.sum((x-y)**2, axis=1)
    return np.exp(-d / (2.0*sigma**2))
    
def gaussian_noise(x, sigmas):
    n = np.random.normal(np.zeros(len(sigmas)), sigmas, size=(x.shape[0], len(sigmas)))
    return x+n

class ParticleFilter(object):
    def __init__(self, priors,  inverse_fn, n_particles=200, dynamics_fn=None, noise_fn=None, 
                score_fn=None, resample_proportion=0.05):
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
        score_fn: computes the distance from the real sensed variable and that returned by inverse_fn. Takes
                  a an array of N sensor outputs and the observed output (x,y) and 
                  returns a strictly positive score for the output. This should be a *similarity* measure, 
                  with higher values meaning more similar
        
        """
        self.priors = priors
        self.d = len(self.priors)
        self.n_particles = n_particles
        self.inverse_fn = inverse_fn
        self.dynamics_fn = dynamics_fn or no_dynamics
        self.noise_fn = noise_fn or no_noise
        self.score_fn = score_fn or squared_error
        self.resample_proportion = resample_proportion
        self.particles = np.zeros((self.d, self.n_particles))
        
    def init_filter(self, mask=None):
        # resample from the prior
        if mask is None:
            for i,prior in enumerate(self.prior):
                self.particles[:,i] = prior.rvs(self.n_particles)
        else:
            for i,prior in enumerate(self.prior):
                self.particles[mask,i] = prior.rvs(self.n_particles)
    
    def update(self, observed):
        
        self.particles = self.dynamics_fn(self.particles)
        self.particles = self.noise_fn(self.particles)
        
        sensors = self.inverse_fn(self.particles)
        scores = self.score_fn(sensors, observed)
        
        # force to be positive and normalise to "probabilities"
        scores = np.clip(scores, 0, np.inf)        
        scores = scores / np.sum(scores)
        
        # resampling step
        indices = resample(scores)
        self.particles = self.particles[indices, :]
        
        # randomly resample some particles from the prior
        random_mask = np.random.random(size=(self.n_particles,))<self.resample_proportion
        self.init_filter(mask=random_mask)
        
from scipy.stats import norm        
priors = [norm(loc=0, scale=1), norm(loc=0, scale=1), gamma(x=1,a=1,loc=0,scale=1)]

def blob(x):
    y = np.zeros(x.shape[0], 32, 32)
    for particle in x:
        

pf = ParticleFilter(priors=priors, 
                    inverse_fn=blob,
                    n_particles=200,
                    noise_fn=gaussian_noise(sigmas=[0.1, 0.1, 0.05]),
                    score_fn=lambda x:squared_error(x, sigma=2),
                    resample_proportion=0.1)