from setuptools import setup
setup(
    name='geoplot',
    packages=['geoplot'],
    install_requires=[
        'matplotlib', 'seaborn', 'pandas', 'geopandas', 'cartopy', 'descartes', 'mapclassify',
        'contextily>=1.0rc2'
    ],
    extras_require={'develop': ['pytest', 'pytest-mpl', 'scipy']},
    py_modules=['geoplot', 'crs', 'utils', 'ops'],
    version='0.4.0',
    python_requires='>=3.6.0',
    description='High-level geospatial plotting for Python.',
    author='Aleksey Bilogur',
    author_email='aleksey.bilogur@gmail.com',
    url='https://github.com/ResidentMario/geoplot',
    download_url='https://github.com/ResidentMario/geoplot/tarball/0.4.0',
    keywords=[
        'data', 'data visualization', 'data analysis', 'data science', 'pandas', 'geospatial data',
        'geospatial analytics'
    ],
    classifiers=[],
)
