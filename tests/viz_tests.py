from geoplot import utils
import geopandas as gpd
from geoplot import pointplot
import unittest
import pytest
import numpy as np

np.random.seed(42)
p_srs = gpd.GeoSeries(utils.gaussian_points(n=100))
p_df = gpd.GeoDataFrame(geometry=p_srs)
p_df = p_df.assign(hue_var=p_df.geometry.map(lambda p: abs(p.y) + abs(p.x)))
p_df = p_df.assign(hue_var_cat=np.floor(p_df.hue_var % (p_df.hue_var.max() / 7)).astype(str))

@pytest.mark.mpl_image_compare
@pytest.mark.parametrize("df", [p_srs, p_df])
def test_pointplot_bare(df):
    return pointplot(df).get_figure()

@pytest.mark.mpl_image_compare
@pytest.mark.parametrize("kwargs,df", [
    [{'hue': 'hue_var', 'linewidth': 0, 's': 100, 'k': 5}, p_df],
    [{'hue': 'hue_var', 'linewidth': 0, 's': 100, 'k': None}, p_df],
    [{'hue': 'hue_var', 'linewidth': 0, 's': 100, 'k': 5, 'scheme': 'fisher_jenks'}, p_df],
    [{'hue': 'hue_var', 'linewidth': 0, 's': 100, 'k': 5, 'scheme': 'quantiles'}, p_df],
    [{'hue': 'hue_var', 'linewidth': 0, 's': 100, 'k': 5, 'scheme': 'equal_interval'}, p_df],
    [{'hue': 'hue_var', 'linewidth': 0, 's': 100, 'k': 3}, p_df],
    [{'hue': 'hue_var_cat', 'linewidth': 0, 's': 100, 'k': 5}, p_df],
    [{'hue': 'hue_var_cat', 'linewidth': 0, 's': 100, 'k': 3}, p_df],
])
def test_hue_params(kwargs, df):
    return pointplot(df, **kwargs).get_figure()
