# pfilter
Basic Python particle filter. Depends on [NumPy](http://numpy.org) only. 

## Usage
Create a `ParticleFilter` object, then call `update(observation)` with an observation array to update the state of the particle filter.

You need to specify at the minimum:
* an **observation function** `observe_fn(state)=>observation` which will return a predicted observation for an internal state.
* a set of **initial distributions** `initial` which is a list of the initial  distributions for all of the internal state variables. These are usually distributions from `scipy.stats`, but any object that has a compatible `rvs()` call to generate random variates will also work 
* a **weight function** `weight_fn(real_observed, hyp_observed_array)=>weights` which specifies how well each of the hyp_observed arrays match the real observation `real_observed`. This must produce a strictly positive weight value, where larger means more similar.

Typically, you would also specify a `dynamics_fn` to update the state based on internal (prediction) dynamics, and a `noise_fn` to add diffusion into the sampling process.

For example, assuming there is a function `blob` which draws a blob on an image of some size (the same size as the observation):

    from pfilter import ParticleFilter, gaussian_noise, squared_error
    columns = ["x", "y", "radius", "dx", "dy"]
    from scipy.stats import norm, gamma, uniform 
    
    # priors for each variable
    # (assumes x and y are coordinates in the range 0-32)
    priors = [  uniform(loc=0, scale=32), 
                uniform(loc=0, scale=32), 
                gamma(a=2,loc=0,scale=10),
                norm(loc=0, scale=0.5),
                norm(loc=0, scale=0.5)]
                                
    # very simple linear dynamcics: x += dx
    def velocity(x):
        xp = np.array(x)
        xp[0:2] += xp[3:5]        
    return xp
    
    # create the filter
    pf = pfilter.ParticleFilter(
                    initial=priors, 
                    observe_fn=blob,
                    n_particles=200,
                    dynamics_fn=velocity,
                    noise_fn=lambda x: 
                                gaussian_noise(x, sigmas=[0.2, 0.2, 0.1, 0.05, 0.05]),
                    weight_fn=lambda x,y:squared_error(x, y, sigma=2),
                    resample_proportion=0.1,
                    column_names = columns)
                    
    # assuming image of the same dimensions/type as blob will produce
    pf.update(image) 
    
See the notebook [PFilterTest.ipynb](PFilterTest.ipynb) for a working example using `skimage` and `OpenCV` which tracks a moving white circle.
    
    
