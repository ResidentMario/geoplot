from setuptools import setup
setup(
    name='geoplot',
    packages=['geoplot'],
    install_requires=[
        'matplotlib', 'seaborn', 'pandas', 'geopandas>=0.9.0', 'cartopy', 'mapclassify>=2.1',
        'contextily>=1.0.0'
    ],
    extras_require={'develop': [
        'pytest', 'pytest-mpl', 'scipy', 'pylint', 'jupyter', 'sphinx', 'sphinx-gallery',
        'sphinx_rtd_theme', 'mplleaflet'
    ]},
    py_modules=['geoplot', 'crs', 'utils', 'ops'],
    version='0.4.4',
    python_requires='>=3.6.0',
    description='High-level geospatial plotting for Python.',
    author='Aleksey Bilogur',
    author_email='aleksey.bilogur@gmail.com',
    url='https://github.com/ResidentMario/geoplot',
    download_url='https://github.com/ResidentMario/geoplot/tarball/0.4.4',
    keywords=[
        'data', 'data visualization', 'data analysis', 'data science', 'pandas', 'geospatial data',
        'geospatial analytics'
    ],
    classifiers=['Framework :: Matplotlib'],
)
