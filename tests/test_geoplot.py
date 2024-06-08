import geopandas as gpd
import geoplot as gplt
import geoplot.crs as gcrs
import matplotlib.pyplot as plt
import pytest


def test_kdeplot_projection_when_shade_true():
    boston_airbnb_listings = gpd.read_file(gplt.datasets.get_path('boston_airbnb_listings'))

    ax = gplt.kdeplot(
        boston_airbnb_listings, cmap='viridis', projection=gcrs.WebMercator(), figsize=(12, 12),
        fill=True
    )