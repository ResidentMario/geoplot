"""
This test file runs a series of property-validating tests using Hypothesis. These tests handle the general
properties of geoplot functions, but don't go over some more involved and/or time-consuming input formatting and
output formatting, which are checked manually elsewhere.

This is the first set of tests that a geoplot iteration should pass. It demonstrates that, at a minimum,
the most common code path works for a variety of inputs.

Testing assumptions:

1.  Data input is pre-validated.
2.  Projections besides the one used for testing projections work equivalently to the one that is being tested.
3.  Extent works as expected (not true: https://github.com/ResidentMario/geoplot/issues/21).
4.  Colormap (cmap, vmin, vmax) variables are vendored, and work as expected.
5.  The figsize variable is vendored, and works as expected.
6.  The limits variable, where defined, works as expected (it's simply float multiplication).
7.  The scale_func variable, where defined, works as expected (this test is covered elsewhere).
8.  Various keyword argument paths work as expected.
9.  Axis stacking works.
10. legend_values and legend_labels works as expected...not true, it's not implemented for hue yet!
11. Different data input combinations work.
"""
# TODO: Number 8 should be a separate test.
# TODO: Number 9 should be a separate test.

import sys; sys.path.insert(0, '../')
import geoplot as gplt
import geoplot.crs as gcrs
import unittest
from hypothesis import given
import hypothesis.strategies as st
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import shapely
import numpy as np


# Define strategies.

# Static inputs.
gaussian_points = gplt.utils.gaussian_points(n=10)
gaussian_polys = gplt.utils.gaussian_polygons(gplt.utils.gaussian_points(n=100), n=2).append(
                 gplt.utils.gaussian_multi_polygons(gplt.utils.gaussian_points(n=100), n=2)
)

# Projections.
projections = st.sampled_from((None, gcrs.PlateCarree()))

# Hue data.
schemes = st.sampled_from((None, "equal_interval", "quantiles", "fisher_jenks"))
k = st.integers(min_value=4, max_value=9).map(lambda f: None if f == 4 else f)
datasets_numeric = st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=10**-10, max_value=10**10),
                            min_size=10, max_size=10, unique=True)
datasets_categorical = st.lists(st.text(st.characters(max_codepoint=1000, blacklist_categories=('Cc', 'Cs')),
                                                      max_size=20), min_size=10, max_size=10)
categorical = st.booleans()
use_hue = st.booleans()


@st.composite
def hue_vars(draw, required=False):
    kwargs = dict()
    if draw(use_hue) or required:
        if draw(categorical):
            kwargs['hue'] = draw(datasets_categorical)
            kwargs['categorical'] = True
        else:
            kwargs['hue'] = draw(datasets_numeric)
            kwargs['scheme'] = draw(schemes)
            kwargs['k'] = draw(k)
    return kwargs


# Scale data
use_scale = st.booleans()
scale_data = datasets_numeric


@st.composite
def scale_vars(draw, required=False):
    kwargs = dict()
    if draw(use_scale) or required:
        kwargs['scale'] = draw(datasets_numeric)
    return kwargs


# Legend
legend = st.booleans()
legend_var = st.sampled_from(("scale", "hue"))


@st.composite
def legend_vars(draw, has_legend_var=True):
    kwargs = dict()
    if draw(legend):
        kwargs['legend'] = True
        if has_legend_var:
            kwargs['legend_var'] = draw(legend_var)
    return kwargs


class TestPointPlot(unittest.TestCase):

    @given(projections, hue_vars(), scale_vars(), legend_vars())
    def test_pointplot(self, projection,
                       hue_vars,
                       scale_vars,
                       legend_vars):
        kwargs = {'projection': projection}
        kwargs = {**kwargs, **hue_vars, **scale_vars, **legend_vars}
        try: gplt.pointplot(gaussian_points, **kwargs)
        finally: plt.close()


class TestPolyPlot(unittest.TestCase):

    # Just two code paths.
    def test_polyplot(self):
        gplt.polyplot(gaussian_polys, projection=None)
        gplt.polyplot(gaussian_polys, projection=gcrs.PlateCarree())
        plt.close()


# Additional strategies.
trace = st.booleans()
# The default scaling function in the cartogram is defined in a way that will raise a ValueError for certain very
# small differences in values, so it needs a custom dataset.
scale_datasets = st.lists(
    st.floats(allow_nan=False, allow_infinity=False, min_value=10**-10, max_value=10**10).map(
        lambda f: f + np.random.random_sample()
    ), min_size=10, max_size=10
)


class TestCartogram(unittest.TestCase):

    @given(projections, scale_datasets, hue_vars(), legend_vars(has_legend_var=False), trace)
    def test_cartogram(self, projection, scale_dataset, hue_vars, legend_vars, trace):
        kwargs = {'projection': projection, 'scale': scale_dataset, 'trace': trace}
        kwargs = {**kwargs, **hue_vars, **legend_vars}
        try: gplt.cartogram(gaussian_polys, **kwargs)
        finally: plt.close()


class TestChoropleth(unittest.TestCase):

    @given(projections, hue_vars(required=True), legend_vars(has_legend_var=False))
    def test_choropleth(self, projection,
                       hue_vars,
                       legend_vars):
        kwargs = {'projection': projection}
        kwargs = {**kwargs, **hue_vars, **legend_vars}
        try: gplt.choropleth(gaussian_polys, **kwargs)
        finally: plt.close()


class TestKDEPlot(unittest.TestCase):

    def test_kdeplot(self):

        # Just four code paths.
        try:
            gplt.kdeplot(gaussian_points, projection=None, clip=None)
            gplt.kdeplot(gaussian_points, projection=None, clip=gaussian_polys)
            gplt.kdeplot(gaussian_points, projection=gcrs.PlateCarree(), clip=gaussian_polys)
            gplt.kdeplot(gaussian_points, projection=gcrs.PlateCarree(), clip=gaussian_polys)
        finally:
            plt.close()


# Additional strategies.
network = gplt.utils.uniform_random_global_network(n=10)
data_kwargs = st.sampled_from(({'start': 'from', 'end': 'to'},
                               {'start': 'from', 'end': 'to', 'path': ccrs.PlateCarree()},
                               {'path': gpd.GeoSeries(
                                    [
                                        shapely.geometry.LineString(
                                            [(a.x, a.y), (b.x, b.y)]
                                        ) for a, b in zip(network['from'], network['to'])
                                    ]
                               )}))


class TestSankey(unittest.TestCase):

    @given(projections,
           hue_vars(),
           scale_vars(),
           legend_vars(),
           data_kwargs)
    def test_sankey(self, projection, hue_vars, scale_vars, legend_vars, data_kwargs):
        kwargs = {'projection': projection}
        kwargs = {**kwargs, **hue_vars, **scale_vars, **legend_vars, **data_kwargs}

        try: gplt.sankey(network, **kwargs)
        finally: plt.close()


# Additional strategies
# Static inputs.
agg_gaussian_points = gplt.utils.gaussian_points(n=100)
agg_gaussian_polys = gplt.utils.gaussian_polygons(gplt.utils.gaussian_points(n=100), n=2)
agg_gaussian_multipolys = gplt.utils.gaussian_multi_polygons(gplt.utils.gaussian_points(n=100), n=2)
agg_data = gpd.GeoDataFrame(geometry=agg_gaussian_points)

# Data input.
# For quadtree mode.
mode = st.integers(min_value=0, max_value=2)
nmin = st.integers(min_value=10, max_value=21).map(lambda v: v if v < 21 else None)
nmax = st.integers(min_value=20, max_value=51).map(lambda v: v if v < 51 else None)
nsig = st.integers(min_value=0, max_value=10)

# For by mode.
cats = st.lists(st.integers(min_value=0, max_value=1), min_size=100, max_size=100)

# For geometry mode.
indexed_geometry = gpd.GeoSeries({0: agg_gaussian_polys[0], 1: agg_gaussian_multipolys[1]})

# For hue.
sankey_datasets_numeric = st.lists(st.floats(allow_nan=False, allow_infinity=False,
                                             min_value=10 ** -10, max_value=10 ** 10),
                                   min_size=100, max_size=100)
sankey_hue = sankey_datasets_numeric


@st.composite
def sankey_data_inputs(draw):
    kwargs = dict()
    md = draw(mode)
    if md == 0:
        kwargs['nmin'] = draw(nmin); kwargs['nmax'] = draw(nmax); kwargs['nsig'] = draw(nsig)
    elif md == 1:
        kwargs['by'] = draw(cats)
    elif md == 2:
        kwargs['by'] = draw(cats); kwargs['geometry'] = indexed_geometry
    return kwargs


class TestAggPlot(unittest.TestCase):

    @given(projections, sankey_hue, legend_vars(has_legend_var=False), sankey_data_inputs())
    def test_aggplot(self, projection,
                     sankey_hue,
                     legend_vars,
                     sankey_data_inputs):
        kwargs = {'projection': projection, 'hue': sankey_hue}
        kwargs = {**kwargs, **legend_vars, **sankey_data_inputs}
        try: gplt.aggplot(agg_data, **kwargs)
        finally: plt.close()