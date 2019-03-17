from setuptools import setup
setup(
    name='geoplot',
    packages=['geoplot'], # this must be the same as the name above
    install_requires=['matplotlib', 'seaborn', 'pandas', 'geopandas', 'cartopy', 'descartes'],
    py_modules=['geoplot', 'crs', 'utils', 'quad'],
    version='0.2.4',
    description='High-level geospatial plotting for Python.',
    author='Aleksey Bilogur',
    author_email='aleksey.bilogur@gmail.com',
    url='https://github.com/ResidentMario/geoplot',
    download_url='https://github.com/ResidentMario/geoplot/tarball/0.2.4',
    keywords=['data', 'data visualization', 'data analysis', 'data science', 'pandas', 'geospatial data',
              'geospatial analytics'],
    classifiers=[],
)
