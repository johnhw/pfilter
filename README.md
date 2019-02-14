# pfilter
Basic Python particle filter. Plain SIR filtering, with systematic resampling. Written to be simple and clear; not necessarily most efficient or most flexible implementation. Depends on [NumPy](http://numpy.org) only. 

## Installation

Available via PyPI:

    pip install pfilter
    
Or install the git version:

    pip install git+https://github.com/johnhw/pfilter.git

## Usage
Create a `ParticleFilter` object, then call `update(observation)` with an observation array to update the state of the particle filter.

You need to specify at the minimum:
* an **observation function** `observe_fn(state) => observation matrix` which will return a predicted observation for an internal state.
* a function that samples from an **initial distributions** `prior_fn=>(n,d) state matrix` for all of the internal state variables. These are usually distributions from `scipy.stats`. The utility function `independent_sample` makes it easy to concatenate sampling functions to sample the whole state vector.
* a **weight function** `weight_fn(real_observed, hyp_observed_array) => weight vector` which specifies how well each of the `hyp_observed` arrays match the real observation `real_observed`. This must produce a strictly positive weight value, where larger means more similar.

Typically, you would also specify:
*  a `dynamics_fn(state) => predicted_state` to update the state based on internal (prediction) dynamics, and a 
* `noise_fn(predicted_state) => noisy_state` to add diffusion into the sampling process. 

For example, assuming there is a function `blob` which draws a blob on an image of some size (the same size as the observation):

```python
        from pfilter import ParticleFilter, gaussian_noise, squared_error, independent_sample
        columns = ["x", "y", "radius", "dx", "dy"]
        from scipy.stats import norm, gamma, uniform 
        
        # prior sampling function for each variable
        # (assumes x and y are coordinates in the range 0-32)    
        prior_fn = independent_sample([uniform(loc=0, scale=32).rvs, 
                    uniform(loc=0, scale=32).rvs, 
                    gamma(a=2,loc=0,scale=10).rvs,
                    norm(loc=0, scale=0.5).rvs,
                    norm(loc=0, scale=0.5).rvs])
                                    
        # very simple linear dynamics: x += dx
        def velocity(x):
            xp = np.array(x)
            xp[0:2] += xp[3:5]        
        return xp
        
        # create the filter
        pf = pfilter.ParticleFilter(
                        prior_fn=prior_fn, 
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
 ```

See the notebook [examples/test_filter.py](examples/test_filter.py) for a working example using `skimage` and `OpenCV` which tracks a moving white circle.
    
    
