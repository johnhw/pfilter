from setuptools import setup
 
setup(
     name='pfilter',    # This is the name of your PyPI-package.
     version='0.1',                          # Update the version number for new releases
     packages=['pfilter'],                  # The name of your scipt, and also the command you'll be using for calling it
     description = 'A basic particle filter',
     author = 'John H Williamson',
     author_email = 'johnhw@gmail.com',
     url = 'https://github.com/johnhw/pfilter', # use the URL to the github repo
    download_url = 'https://github.com/johnhw/pfilter/tarball/0.1',
    keywords=["particle", "probabilistic", "stochastic", "filter", "filtering"]
 )