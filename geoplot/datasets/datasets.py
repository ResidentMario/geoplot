"""
User-facing utility module for downloading example datasets. Similar to the `geopandas.datasets` namespace.

WIP!
"""
# TODO: Just using GitHub as a filehost for now, move to AWS for file hosting later on.

import geopandas as gpd
import requests


def get_path(dname):
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
    dmap = {
        "nyc_boroughs": "https://github.com/ResidentMario/geoplot-data/raw/master/nyc_boroughs/boroughs.geojson",
    }

    if dname not in dmap:
        raise ValueError("No dataset by the name")

    return gpd.GeoDataFrame(requests.get(dmap[dname]))
