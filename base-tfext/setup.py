from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'Helpful Tensorflow functionality'
LONG_DESCRIPTION = 'Helpful Tensorflow functionality including new layers, losses/metrics, and more'

# Setting up
setup(
    name="tfext", 
    version=VERSION,
    author="Jacob Daniels",
    author_email="jacobdaniels2@my.unt.edu",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    # Included so .egg file is not created, as the .egg file
    # is not compatible with some editor's documentation viewers
    zip_safe=False,
    packages=find_packages(),
    # add any additional packages that 
    # needs to be installed along with your package.
    install_requires=[], 
)