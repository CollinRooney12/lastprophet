from setuptools import setup

setup(name = 'lastprophet',
      version = '0.0.1',
      description = "Long and Short Term Time Series Forecasting with Facebook's Prophet tool",
      url = "https://github.com/CollinRooney12/lastprophet",
      author = "Collin Rooney",
      author_email = 'CollinRooney12@gmail.com',
      license = 'MIT',
      keywords='last MAPA thief temporal time series hierarchy forecast Prophet',
      packages = ['lastprophet'],
      zip_safe = False,
      test_suite='nose.collector',
      tests_require=['nose'],
      install_requires = [
              'matplotlib',
              'pandas>=0.18.1',
              'numpy',
              'fbprophet',
              'calendar'],
       )