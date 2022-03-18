from setuptools import setup


doc_requires = [
    'sphinx', 'sphinx-gallery', 'sphinx_rtd_theme', 'nbsphinx', 'ipython',
    'mplleaflet', 'scipy',
]
test_requires = ['pytest', 'pytest-mpl', 'scipy']

setup(
    name='geoplot',
    packages=['geoplot'],
    install_requires=[
        'matplotlib>=3.1.2',  # seaborn GH#1773
        'seaborn', 'pandas', 'geopandas>=0.9.0', 'cartopy', 'mapclassify>=2.1',
        'contextily>=1.0.0'
    ],
    extras_require={
        'doc': doc_requires,
        'test': test_requires,
        'develop': [*doc_requires, *test_requires, 'pylint'],
    },
    py_modules=['geoplot', 'crs', 'utils', 'ops'],
    version='0.5.1',
    python_requires='>=3.7.0',
    description='High-level geospatial plotting for Python.',
    author='Aleksey Bilogur',
    author_email='aleksey.bilogur@gmail.com',
    url='https://github.com/ResidentMario/geoplot',
    download_url='https://github.com/ResidentMario/geoplot/tarball/0.5.1',
    keywords=[
        'data', 'data visualization', 'data analysis', 'data science', 'pandas', 'geospatial data',
        'geospatial analytics'
    ],
    classifiers=['Framework :: Matplotlib'],
)
