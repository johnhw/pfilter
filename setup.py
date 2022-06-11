from setuptools import setup

with open("README.md") as f:
    readme = f.read()


setup(
    name="pfilter",  # This is the name of your PyPI-package.
    version="0.2.5",  # Update the version number for new releases
    install_requires=['numpy'],
    packages=[
        "pfilter"
    ],  # The name of your scipt, and also the command you'll be using for calling it
    description="A basic particle filter",
    author="John H Williamson",
    long_description_content_type="text/markdown",
    long_description=readme,
    author_email="johnhw@gmail.com",
    url="https://github.com/johnhw/pfilter",  # use the URL to the github repo
    download_url="https://github.com/johnhw/pfilter/tarball/0.2.5",
    keywords=["particle", "probabilistic", "stochastic", "filter", "filtering"],
)

