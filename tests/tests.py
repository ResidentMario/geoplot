"""
Testing assumptions:

1.  Data input is pre-validated.
2.  Projections besides the one used for testing projections work equivalently to the one that is being tested.
3.  Extent works as expected (not true: https://github.com/ResidentMario/geoplot/issues/21).
4.  Colormap (cmap, vmin, vmax) variables are vendored, and work as expected.
5.  The figsize variable is vendored, and works as expected.
6.  The limits variable, where defined, works as expected (it's simply float multiplication).
7.  The scale_func varaible, where defined, works as expected (this test is covered elsewhere).
8.  Various keyword argument paths work as expected.
9.  Axis stacking works.
10. legend_values and legend_labels works as expected...not true, it's not implemented for hue yet!
"""
# TODO: Number 8 should be a separate test.
# TODO: Number 9 should be a separate test.

import sys; sys.path.insert(0, '../')
import geoplot as gplt
import geoplot.crs as gcrs
import unittest
# import hypothesis as hyp
from hypothesis import given
import hypothesis.strategies as st
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
# import math
# import matplotlib as mpl
# import cartopy
# import cartopy.crs as ccrs
# import shapely
# import random


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
datasets_numeric = st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=10, max_size=10, unique=True)
datasets_categorical = st.lists(st.text(max_size=20), min_size=10, max_size=10)
is_categorical = st.booleans()
use_hue = st.booleans()

# Scale data
use_scale = st.booleans()
scale_data = datasets_numeric

# Legend
use_legend = st.booleans()
legend_var = st.sampled_from(("scale", "hue"))


# class TestPointPlot(unittest.TestCase):
#
#     @given(projections,
#            datasets_categorical, datasets_numeric,
#            use_hue, is_categorical, schemes, k,
#            use_scale,
#            use_legend, legend_var)
#     def test_pointplot(self, projection,
#                        data_categorical, data_numeric,
#                        use_hue, categorical, scheme, k,
#                        use_scale,
#                        use_legend, legend_var):
#         kwargs = {'projection': projection, 'categorical': categorical}
#
#         # Hue.
#         if use_hue and categorical:  kwargs['hue'] = data_categorical
#         elif use_hue and (not categorical):
#             kwargs['hue'] = data_numeric; kwargs['scheme'] = scheme; kwargs['k'] = k
#
#         # Scale.
#         if use_scale: kwargs['scale'] = data_numeric  # No harm or decreased coverage in reuse.
#
#         # Legend.
#         if use_legend: kwargs['legend'] = True; kwargs['legend_var'] = legend_var
#
#         try:
#             gplt.pointplot(gaussian_points, **kwargs)
#        except:
#             import pdb; pdb.set_trace()
#             gplt.pointplot(gaussian_points, **kwargs)
#         finally:
#             plt.close()


# class TestPolyPlot(unittest.TestCase):
#
#     @given(projections)
#     def test_polyplot(self, projection):
#         gplt.polyplot(gaussian_polys, projection=projection)
#         plt.close()


# Additional strategies.
trace = st.booleans()
# The dataset used for the scale variable is defined such that it avoids the "infinite slope" error caught and raised
# in the method.
scale_datasets = st.lists(
    st.floats(allow_nan=False, allow_infinity=False).map(lambda f: f + np.random.random_sample()),
    min_size=10, max_size=10#, unique=True
)


class TestCartogram(unittest.TestCase):

    @given(projections,
           datasets_categorical, datasets_numeric,
           scale_datasets,
           use_hue, is_categorical, schemes, k,
           use_legend, legend_var,
           trace)
    def test_cartogram(self, projection,
                       data_categorical, data_numeric,
                       scale_dataset,
                       use_hue, categorical, scheme, k,
                       use_legend, legend_var,
                       trace):
        kwargs = {'projection': projection, 'categorical': categorical, 'scale': scale_dataset, 'trace': trace}

        # Hue.
        if use_hue and categorical:
            kwargs['hue'] = data_categorical
        elif use_hue and (not categorical):
            kwargs['hue'] = data_numeric; kwargs['scheme'] = scheme; kwargs['k'] = k

        # Legend.
        if use_legend: kwargs['legend'] = True; kwargs['legend_var'] = legend_var

        try:
            gplt.cartogram(gaussian_polys, **kwargs)
        # except:
        #     import pdb; pdb.set_trace()
        #     gplt.cartogram(gaussian_polys, **kwargs)
        finally:
            plt.close()
