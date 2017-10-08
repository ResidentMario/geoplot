"""
User-facing utility module for downloading example datasets. Similar to the `geopandas.datasets` namespace.
"""

import geopandas as gpd
import fiona
from zipfile import ZipFile


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
    z = ZipFile("examples.zip")

    with fiona.BytesCollection(z.read('geoplot-data/{0}.geojson'.format(dname))) as f:
        crs = f.crs
        gdf = gpd.GeoDataFrame.from_features(f, crs=crs)
        return gdf
