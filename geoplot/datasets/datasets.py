"""
User-facing utility module for downloading example datasets. Similar to the `geopandas.datasets` namespace.
"""

import geopandas as gpd
import pandas as pd
import fiona
from zipfile import ZipFile
from pkg_resources import resource_stream, resource_listdir
from io import BytesIO


def load(dname):
    """
    Retrieves a dataset by name.

    Parameters
    ----------
    dname : str
        Name of the dataset to retrieve.

    Returns
    -------
    GeoDataFrame instance
        The dataset being referenced.
    """
    try:
        assert 'examples.zip' in resource_listdir('geoplot.datasets', '')
    except AssertionError:
        raise IOError("The 'examples.zip' file packaging geoplot example datasets was not found.")

    z = ZipFile(resource_stream('geoplot.datasets', 'examples.zip'))

    if dname in ['boston-airbnb-listings', 'boston-zip-codes', 'contiguous-usa', 'dc-roads', 'la-flights',
                 'nyc-boroughs', 'napoleon-troop-movements',
                 'ny-census-partial', 'nyc-boroughs', 'nyc-collision-factors', 'nyc-fatal-collisions',
                 'nyc-injurious-collisions', 'nyc-parking-tickets-sample', 'nyc-police-precincts', 'usa-cities']:
        with fiona.BytesCollection(z.read('geoplot-data/{0}.geojson'.format(dname))) as f:
            crs = f.crs
            gdf = gpd.GeoDataFrame.from_features(f, crs=crs)
            return gdf
    elif dname in ['obesity-by-state']:
        return pd.read_csv(BytesIO(z.read("geoplot-data/{0}.tsv".format(dname))), sep='\t')
    else:
        raise ValueError('The provided dataset is not in the example datasets provided with geoplot.')
