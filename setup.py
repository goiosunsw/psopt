
from setuptools import setup

setup(name='psopt',
      version='0.1',
      description='optimizer wrapers for scipy.optimize',
      url='http://github.com/goiosunw/psopt',
      author='Andre Goios',
      author_email='a.almeida@unsw.edu.au',
      license='GPL v3',
      packages=['psopt'],
      install_requires=[
          'numpy',
          'scipy',
          'matplotlib',
      ],
      zip_safe=False)
