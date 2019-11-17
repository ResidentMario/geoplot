import unittest
import pytest
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt

from geoplot import utils
from geoplot import (
    pointplot, voronoi, kdeplot, polyplot, webmap, choropleth, cartogram, quadtree,
    sankey
)
from geoplot.crs import AlbersEqualArea, WebMercator


np.random.seed(42)
p_srs = gpd.GeoSeries(utils.gaussian_points(n=100))
p_df = gpd.GeoDataFrame(geometry=p_srs)
p_df = p_df.assign(var=p_df.geometry.map(lambda p: abs(p.y) + abs(p.x)))
p_df = p_df.assign(var_cat=np.floor(p_df['var'] // (p_df['var'].max() / 5)).astype(str))

poly_df = gpd.GeoDataFrame(geometry=utils.gaussian_polygons(p_srs.geometry, n=10))
poly_df = poly_df.assign(
    var=poly_df.geometry.centroid.x.abs() + poly_df.geometry.centroid.y.abs()
)

ls_df = gpd.GeoDataFrame(geometry=utils.gaussian_linestrings(p_srs.geometry))
ls_df = ls_df.assign(var=ls_df.geometry.centroid.x.abs() + ls_df.geometry.centroid.y.abs())

clip_geom = gpd.GeoSeries(Polygon([[-10, -10], [10, -10], [10, 10], [-10, 10]]))
non_clip_geom = gpd.GeoSeries(Polygon([[-30, -30], [30, -30], [30, 30], [-30, 30]]))

def identity_scale(minval, maxval):
    def scalar(val):
        return 10
    return scalar


def axis_initializer(f):
    def wrapped(_self):
        try:
            f(_self)
        finally:
            plt.close('all')
    return wrapped


@pytest.mark.mpl_image_compare
@pytest.mark.parametrize("kwargs", [
    {'hue': 'var', 'linewidth': 0, 's': 10},
    {'hue': 'var', 'linewidth': 0, 's': 10, 'scheme': 'fisher_jenks'},
    {'hue': 'var', 'linewidth': 0, 's': 10, 'scheme': 'quantiles'},
    {'hue': 'var', 'linewidth': 0, 's': 10, 'scheme': 'equal_interval'},
    {'hue': 'var_cat', 'linewidth': 0, 's': 10},
    {'hue': 'var_cat', 'linewidth': 0, 's': 10, 'scheme': 'categorical'},
    {'hue': 'var', 'linewidth': 0, 's': 10, 'cmap': 'Greens', 'scheme': 'quantiles'},
    {'hue': 'var', 'linewidth': 0, 's': 10, 'cmap': 'Greens'},
    {'hue': p_df['var'], 'linewidth': 0, 's': 10},
    {'hue': np.array(p_df['var']), 'linewidth': 0, 's': 10},
    {'hue': list(p_df['var']), 'linewidth': 0, 's': 10}
])
def test_hue_params(kwargs):
    return pointplot(p_df, **kwargs).get_figure()


# xfail due to seaborn#1773
@pytest.mark.mpl_image_compare
@pytest.mark.parametrize("kwargs", [
    pytest.param({'cmap': 'Reds'}, marks=pytest.mark.xfail),
    pytest.param({'cmap': 'Blues', 'shade': True}, marks=pytest.mark.xfail),
    pytest.param({'cmap': 'Greens', 'shade': True, 'shade_lowest': True}, marks=pytest.mark.xfail)
])
def test_hue_params_kdeplot(kwargs):
    return kdeplot(p_df, **kwargs).get_figure()


@pytest.mark.mpl_image_compare
@pytest.mark.parametrize("kwargs", [
    {'scale': 'var', 'linewidth': 0, 'limits': (5, 10)},
    {'scale': 'var', 'linewidth': 0, 'limits': (5, 10), 'scale_func': identity_scale}
])
def test_scale_params(kwargs):
    return pointplot(p_df, **kwargs).get_figure()


@pytest.mark.mpl_image_compare
@pytest.mark.parametrize("kwargs", [
    {'clip': clip_geom, 'edgecolor': 'white', 'facecolor': 'lightgray'},
    {'clip': non_clip_geom, 'edgecolor': 'white', 'facecolor': 'lightgray'},
    {'clip': clip_geom, 'edgecolor': 'white', 'facecolor': 'lightgray',
     'projection': AlbersEqualArea()},
    {'clip': non_clip_geom, 'edgecolor': 'white', 'facecolor': 'lightgray',
     'projection': AlbersEqualArea()}
])
def test_clip_params_geometric(kwargs):
    return voronoi(p_df, **kwargs).get_figure()


# xfail due to seaborn#1773
@pytest.mark.mpl_image_compare
@pytest.mark.parametrize("kwargs", [
    pytest.param({'clip': clip_geom}, marks=pytest.mark.xfail),
    pytest.param({'clip': non_clip_geom}, marks=pytest.mark.xfail),
    pytest.param({'clip': clip_geom, 'projection': AlbersEqualArea()}, marks=pytest.mark.xfail),
    pytest.param({'clip': non_clip_geom, 'projection': AlbersEqualArea()}, marks=pytest.mark.xfail)
])
def test_clip_params_overlay(kwargs):
    return kdeplot(p_df, **kwargs).get_figure()


@pytest.mark.mpl_image_compare
@pytest.mark.parametrize("kwargs", [
    {'hue': 'var', 'linewidth': 0, 's': 10, 'legend': True},
    {'hue': 'var', 'linewidth': 0, 's': 10, 'scheme': 'quantiles', 'legend': True},
    {'hue': 'var', 'linewidth': 0, 's': 10, 'legend': False},
    {'scale': 'var', 'linewidth': 0, 'limits': (5, 10), 'legend': True},
    {'scale': 'var', 'linewidth': 0, 'limits': (5, 10), 'legend': False},
    {'scale': 'var', 'linewidth': 0, 'limits': (5, 10), 'scale_func': identity_scale,
     'legend': True},
    {'hue': 'var', 'scale': 'var', 'linewidth': 0, 'limits': (5, 10), 'legend': True,
     'legend_var': 'hue'},
    {'hue': 'var', 'scale': 'var', 'linewidth': 0, 'limits': (5, 10), 'legend': True,
     'legend_var': 'hue', 'scheme': 'quantiles'},
    {'hue': 'var', 'scale': 'var', 'linewidth': 0, 'limits': (5, 10), 'legend': True,
     'legend_var': 'scale'},
    {'hue': 'var', 'linewidth': 0, 's': 10, 'scheme': 'quantiles', 'legend': True,
     'legend_labels': list('ABCDE')},
    # kwargs[10], this one is broken (below)
    {'hue': 'var', 'linewidth': 0, 's': 10, 'scheme': 'quantiles', 'legend': True,
     'legend_values': [1, 1, 2, 4, 4]},
    # kwargs[11], this one is also broken
    {'hue': 'var', 'linewidth': 0, 's': 10, 'scheme': 'quantiles', 'legend': True,
     'legend_labels': list('ABCDE'), 'legend_values': [1, 1, 2, 4, 4]},
    {'scale': 'var', 'linewidth': 0, 'limits': (5, 10), 'legend': True,
     'legend_labels': list('ABCDE')},
    {'scale': 'var', 'linewidth': 0, 'limits': (5, 10), 'legend': True,
     'legend_values': [1, 1, 2, 4, 4]},
    {'scale': 'var', 'linewidth': 0, 'limits': (5, 10), 'legend': True,
     'legend_labels': list('ABCDE'), 'legend_values': [1, 1, 2, 4, 4]},
    {'hue': 'var', 'linewidth': 0, 's': 10, 'norm': Normalize(vmin=0, vmax=10), 'legend': True},
    {'hue': 'var', 'linewidth': 0, 's': 10, 'legend': True,
     'legend_kwargs': {'orientation': 'horizontal'}},
    {'hue': 'var', 'linewidth': 0, 's': 10, 'legend': True, 'scheme': 'quantiles',
     'legend_kwargs': {'bbox_to_anchor': (1.0, 1.2)}},
    {'hue': 'var', 'linewidth': 0, 's': 10, 'legend': True, 'scheme': 'quantiles',
     'legend_kwargs': {'markeredgecolor': 'purple', 'markeredgewidth': 5}},
    {'scale': 'var', 'linewidth': 0,'limits': (5, 10),
     'legend': True, 'legend_kwargs': {'markerfacecolor': 'purple'}}
])
def test_legend_params(kwargs):
    return pointplot(p_df, **kwargs).get_figure()


@pytest.mark.mpl_image_compare
@pytest.mark.parametrize("func,df,kwargs", [
    [pointplot, p_df, {'hue': 'var', 'linewidth': 0, 's': 10, 'legend': True}],
    [pointplot, p_df,
     {'hue': 'var', 'linewidth': 0, 's': 10, 'legend': True,
      'projection': AlbersEqualArea()}],
    # xfail due to seaborn#1773
    pytest.param(*[kdeplot, p_df, {}], marks=pytest.mark.xfail),
    pytest.param(*[kdeplot, p_df, {'projection': AlbersEqualArea()}], marks=pytest.mark.xfail),
    [polyplot, poly_df, {}],
    [polyplot, poly_df, {'projection': AlbersEqualArea()}],
    # xfail because webmap tiles are subject to remote change
    pytest.param(*[webmap, p_df, {'projection': WebMercator()}], marks=pytest.mark.xfail),
    [choropleth, poly_df, {'hue': 'var', 'linewidth': 0, 'legend': True}],
    [choropleth, poly_df, 
     {'hue': 'var', 'linewidth': 0, 'legend': True,
      'projection': AlbersEqualArea()}],
    [cartogram, poly_df, {'scale': 'var', 'linewidth': 0, 'legend': True}],
    [cartogram, poly_df,
     {'scale': 'var', 'linewidth': 0, 'legend': True,
      'projection': AlbersEqualArea()}],
    [voronoi, p_df, {'facecolor': 'lightgray', 'edgecolor': 'white'}],
    [voronoi, p_df, 
     {'facecolor': 'lightgray', 'edgecolor': 'white',
      'projection': AlbersEqualArea()}],
    [quadtree, p_df, {'facecolor': 'lightgray', 'edgecolor': 'white'}],
    [quadtree, p_df, 
     {'facecolor': 'lightgray', 'edgecolor': 'white', 'projection': AlbersEqualArea()}],
    [sankey, ls_df, {'scale': 'var', 'legend': True}],
    [sankey, ls_df, {'scale': 'var', 'legend': True, 'projection': AlbersEqualArea()}]
])
def test_plot_basic(func, df, kwargs):
    return func(df, **kwargs).get_figure()


@pytest.mark.mpl_image_compare
def test_param_extent_unproj():
    # invalid extent: raise
    with pytest.raises(ValueError):
        pointplot(p_df, extent=(-181, 0, 1, 1))
    with pytest.raises(ValueError):
        pointplot(p_df, extent=(0, -91, 1, 1))
    with pytest.raises(ValueError):
        pointplot(p_df, extent=(0, 0, 181, 1))
    with pytest.raises(ValueError):
        pointplot(p_df, extent=(0, 0, 1, 91))

    # valid extent: set
    return pointplot(p_df, hue='var', linewidth= 0, s=10, extent=(-10, -10, 10, 10)).get_figure()


@pytest.mark.mpl_image_compare
def test_param_extent_proj():
    # invalid extent: raise
    with pytest.raises(ValueError):
        pointplot(p_df, extent=(-181, 0, 1, 1))
    with pytest.raises(ValueError):
        pointplot(p_df, extent=(0, -91, 1, 1))
    with pytest.raises(ValueError):
        pointplot(p_df, extent=(0, 0, 181, 1))
    with pytest.raises(ValueError):
        pointplot(p_df, extent=(0, 0, 1, 91))

    # valid extent: set
    return pointplot(
        p_df, hue='var', linewidth= 0, s=10, extent=(-10, -10, 10, 10),
        projection=AlbersEqualArea()
    ).get_figure()
