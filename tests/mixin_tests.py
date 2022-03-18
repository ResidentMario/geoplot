from geoplot.geoplot import (Plot, HueMixin, ScaleMixin, ClipMixin, LegendMixin, webmap)
import geoplot.utils as utils

import pandas as pd
import geopandas as gpd
import unittest
import pytest
import matplotlib.pyplot as plt
from matplotlib.axes._subplots import SubplotBase
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxesSubplot
import geoplot.crs as gcrs
from shapely.geometry import Point, Polygon
import numpy as np


def figure_cleanup(f):
    def wrapped(_self):
        try:
            f(_self)
        finally:
            plt.close('all')
    return wrapped


class TestPlot(unittest.TestCase):
    def setUp(self):
        self.kwargs = {'figsize': (8, 6), 'ax': None, 'extent': None, 'projection': None}
        self.gdf = gpd.GeoDataFrame(geometry=[])
        self.nonempty_gdf = gpd.GeoDataFrame(geometry=[Point(-1, -1), Point(1, 1)])

    @figure_cleanup
    def test_base_init(self):
        """Test the base init all plotters pass to Plot."""
        plot = Plot(self.gdf, **self.kwargs)
        assert plot.figsize == (8, 6)
        assert isinstance(plot.ax, SubplotBase)  # SO 11690597
        assert plot.extent is None
        assert plot.projection is None

        plot = Plot(self.gdf, **{**self.kwargs, **{'projection': gcrs.PlateCarree()}})
        assert plot.figsize == (8, 6)
        assert isinstance(plot.ax, GeoAxesSubplot)
        assert plot.extent is None
        assert isinstance(plot.projection, ccrs.PlateCarree)

    @figure_cleanup
    def test_no_geometry_col(self):
        """Test the requirement that the geometry column is set."""
        with pytest.raises(ValueError):
            Plot(gpd.GeoDataFrame(), **self.kwargs)

    @figure_cleanup
    def test_init_ax(self):
        """Test that the passed Axes is set."""
        _, ax = plt.subplots(figsize=(2, 2))
        plot = Plot(self.gdf, **{**self.kwargs, **{'ax': ax}})
        assert plot.figsize == (2, 2)

        ax = plt.axes(projection=ccrs.PlateCarree())
        plot = Plot(self.gdf, **{**self.kwargs, **{'ax': ax}})
        assert plot.ax == ax

        # non-default user-set figure sizes are ignored with a warning when ax is also set
        with pytest.warns(UserWarning):
            Plot(self.gdf, **{**self.kwargs, **{'figsize': (1, 1), 'ax': ax}})

    @figure_cleanup
    def test_init_projection(self):
        """Test that passing a projection works as expected."""
        plot = Plot(self.gdf, **{**self.kwargs, 'projection': gcrs.PlateCarree()})
        assert isinstance(plot.projection, ccrs.PlateCarree)

    @figure_cleanup
    def test_init_extent_axes(self):
        """Test the extent setter code in the Axes case."""
        # default, empty geometry case: set extent to default value of (0, 1)
        plot = Plot(self.gdf, **self.kwargs)
        assert plot.ax.get_xlim() == plot.ax.get_ylim() == (0, 1)

        # default, non-empty geometry case: use a (relaxed) geometry envelope
        plot = Plot(gpd.GeoDataFrame(geometry=[Point(-1, -1), Point(1, 1)]), **self.kwargs)
        xmin, xmax = plot.ax.get_xlim()
        ymin, ymax = plot.ax.get_ylim()
        assert xmin < -1
        assert xmax > 1
        assert ymin < -1
        assert ymax > 1

        # empty geometry, valid extent case: reuse prior extent, which is (0, 1) by default
        plot = Plot(self.gdf, **{**self.kwargs, **{'extent': (-1, -1, 1, 1)}})
        assert plot.ax.get_xlim() == plot.ax.get_ylim() == (0, 1)

        # nonempty geometry, valid extent case: use extent
        plot = Plot(self.nonempty_gdf, **{**self.kwargs, **{'extent': (-1, -1, 1, 1)}})
        xmin, xmax = plot.ax.get_xlim()
        ymin, ymax = plot.ax.get_ylim()
        assert xmin == -1
        assert xmax == 1
        assert ymin == -1
        assert ymax == 1

        # nonempty geometry, numerically invalid extent case: raise
        with pytest.raises(ValueError):
            Plot(self.nonempty_gdf, **{**self.kwargs, **{'extent': (-181, 0, 1, 1)}})
        with pytest.raises(ValueError):
            Plot(self.nonempty_gdf, **{**self.kwargs, **{'extent': (0, -91, 1, 1)}})
        with pytest.raises(ValueError):
            Plot(self.nonempty_gdf, **{**self.kwargs, **{'extent': (0, 0, 181, 1)}})
        with pytest.raises(ValueError):
            Plot(self.nonempty_gdf, **{**self.kwargs, **{'extent': (0, 0, 1, 91)}})

        # nonempty geometry, zero extent case: warn and relax (cartopy behavior)
        with pytest.warns(UserWarning):
            Plot(self.nonempty_gdf, **{**self.kwargs, **{'extent': (0, 0, 0, 0)}})

    @figure_cleanup
    def test_init_extent_geoaxes(self):
        """Test the extent setter code in the GeoAxes case."""
        # default, empty geometry case: set extent to default value of (0, 1)
        plot = Plot(self.gdf, **{**self.kwargs, **{'projection': gcrs.PlateCarree()}})
        assert plot.ax.get_xlim() == plot.ax.get_ylim() == (0, 1)

        # default, non-empty geometry case: use a (relaxed) geometry envelope
        plot = Plot(
            gpd.GeoDataFrame(geometry=[Point(-1, -1), Point(1, 1)]),
            **{**self.kwargs, **{'projection': gcrs.PlateCarree()}}
        )
        xmin, xmax = plot.ax.get_xlim()
        ymin, ymax = plot.ax.get_ylim()
        assert xmin < -1
        assert xmax > 1
        assert ymin < -1
        assert ymax > 1

        # empty geometry, valid extent case: reuse prior extent, which is (0, 1) by default
        plot = Plot(self.gdf, **{
            **self.kwargs, **{'extent': (-1, -1, 1, 1), 'projection': gcrs.PlateCarree()}
        })
        assert plot.ax.get_xlim() == plot.ax.get_ylim() == (0, 1)

        # nonempty geometry, valid extent case: use extent
        plot = Plot(self.nonempty_gdf, **{
            **self.kwargs, **{'extent': (-1, -1, 1, 1), 'projection': gcrs.PlateCarree()}
        })
        xmin, xmax = plot.ax.get_xlim()
        ymin, ymax = plot.ax.get_ylim()
        assert xmin == -1
        assert xmax == 1
        assert ymin == -1
        assert ymax == 1

        # nonempty geometry, unsatisfiable extent case: warn and fall back to default
        with pytest.warns(UserWarning):
            # Orthographic can only show one half of the world at a time
            Plot(self.nonempty_gdf, **{
                **self.kwargs,
                **{'extent': (-180, -90, 180, 90), 'projection': gcrs.Orthographic()}
            })


class TestHue(unittest.TestCase):
    def setUp(self):
        def create_huemixin():
            huemixin = HueMixin()
            # set props the mixin is responsible for
            huemixin.kwargs = {'hue': 'foo', 'cmap': 'viridis', 'norm': None}
            # set props set by the plot object initializer
            huemixin.ax = None
            huemixin.figsize = (8, 6)
            huemixin.extent = None
            huemixin.projection = None

            np.random.seed(42)
            huemixin.df = gpd.GeoDataFrame(
                {'foo': np.random.random(100), 'geometry': utils.gaussian_points(n=100)}
            )
            return huemixin

        self.create_huemixin = create_huemixin

    def test_hue_init_defaults(self):
        huemixin = self.create_huemixin()
        huemixin.set_hue_values(supports_categorical=False)
        assert len(huemixin.colors) == 100
        assert isinstance(huemixin.hue, pd.Series) and len(huemixin.hue) == 100
        assert huemixin.scheme is None
        assert huemixin.k is None
        assert isinstance(huemixin.mpl_cm_scalar_mappable, ScalarMappable)
        assert huemixin.k is None
        assert huemixin.categories is None
        assert huemixin.color_kwarg == 'color'
        assert huemixin.default_color == 'steelblue'

    def test_hue_init_hue(self):
        # hue is initialized as a string: source from the backing GeoDataFrame
        huemixin = self.create_huemixin()
        huemixin.set_hue_values(supports_categorical=False)
        assert (huemixin.hue == huemixin.df['foo']).all()

        # hue is initialized as a Series: pass that directly to the param
        huemixin = self.create_huemixin()
        hue = pd.Series(np.random.random(100))
        huemixin.kwargs['hue'] = hue
        huemixin.set_hue_values(supports_categorical=False)
        assert(huemixin.hue == hue).all()

        # hue is initialized as a list: transform that into a GeoSeries
        huemixin = self.create_huemixin()
        hue = list(np.random.random(100))
        huemixin.kwargs['hue'] = hue
        huemixin.set_hue_values(supports_categorical=False)
        assert(huemixin.hue == hue).all()

        # hue is initialized as an array: transform that into a GeoSeries
        huemixin = self.create_huemixin()
        hue = np.random.random(100)
        huemixin.kwargs['hue'] = hue
        huemixin.set_hue_values(supports_categorical=False)
        assert(huemixin.hue == hue).all()

    def test_hue_init_cmap(self):
        # cmap is None: 'viridis' is used
        expected = self.create_huemixin()
        expected.kwargs['cmap'] = 'viridis'
        result = self.create_huemixin()
        result.kwargs['cmap'] = None
        expected.set_hue_values(supports_categorical=False)
        result.set_hue_values(supports_categorical=False)
        assert result.colors == expected.colors

        # cmap is the name of a colormap: its value is propogated
        huemixin = self.create_huemixin()
        huemixin.kwargs['cmap'] = 'jet'
        huemixin.set_hue_values(supports_categorical=False)
        assert huemixin.mpl_cm_scalar_mappable.cmap.name == 'jet'

        # cmap is a Colormap instance: it is propogated
        # Colormap is an abstract class, LinearSegmentedColormap stands in as a test object
        huemixin = self.create_huemixin()
        colors = [(215 / 255, 193 / 255, 126 / 255), (37 / 255, 37 / 255, 37 / 255)]
        cm = LinearSegmentedColormap.from_list('test_colormap', colors)
        huemixin.kwargs['cmap'] = cm
        huemixin.set_hue_values(supports_categorical=False)
        assert huemixin.mpl_cm_scalar_mappable.cmap.name == 'test_colormap'

        # cmap is not None but hue is None: raise
        huemixin = self.create_huemixin()
        huemixin.kwargs['cmap'] = 'viridis'
        huemixin.kwargs['hue'] = None
        with pytest.raises(ValueError):
            huemixin.set_hue_values(supports_categorical=False)

    def test_hue_init_norm(self):
        # norm is None: a Normalize instance is used with vmin, vmax boundaries
        huemixin = self.create_huemixin()
        huemixin.set_hue_values(supports_categorical=False)
        assert huemixin.mpl_cm_scalar_mappable.norm.vmin == np.min(huemixin.hue)
        assert huemixin.mpl_cm_scalar_mappable.norm.vmax == np.max(huemixin.hue)

        # norm is not None: it is propogated
        huemixin = self.create_huemixin()
        norm = Normalize(vmin=-0.1, vmax=0.1)
        huemixin.kwargs['norm'] = norm
        huemixin.set_hue_values(supports_categorical=False)
        assert huemixin.mpl_cm_scalar_mappable.norm == norm

    def test_hue_init_color_kwarg(self):
        # color_kwarg in keyword arguments and hue is not None: raise
        huemixin = self.create_huemixin()
        huemixin.kwargs['color'] = 'black'
        huemixin.kwargs['hue'] = 'viridis'
        with pytest.raises(ValueError):
            huemixin.set_hue_values(supports_categorical=False)

        # color_kwarg in keyword arguments and hue is None: set color
        huemixin = self.create_huemixin()
        huemixin.kwargs['color'] = 'black'
        huemixin.kwargs['hue'] = None
        huemixin.kwargs['cmap'] = None
        huemixin.set_hue_values(supports_categorical=False)
        huemixin.colors == ['black'] * 100

        # non-default color_kwarg case
        huemixin = self.create_huemixin()
        huemixin.color_kwarg = 'foofacecolor'
        huemixin.kwargs['foofacecolor'] = 'black'
        huemixin.kwargs['hue'] = None
        huemixin.kwargs['cmap'] = None
        huemixin.set_hue_values(supports_categorical=False)
        huemixin.colors == ['black'] * 100

        # no hue non-default color case
        huemixin = self.create_huemixin()
        huemixin.kwargs['hue'] = None
        huemixin.kwargs['cmap'] = None
        huemixin.set_hue_values(supports_categorical=False, default_color='black')
        huemixin.colors == ['black'] * 100

    def test_hue_init_scheme_kwarg(self):
        # k is not None, scheme is not None, hue is None: raise
        huemixin = self.create_huemixin()
        huemixin.kwargs['k'] = 5
        huemixin.kwargs['scheme'] = 'FisherJenks'
        huemixin.kwargs['hue'] = None
        huemixin.kwargs['cmap'] = None
        with pytest.raises(ValueError):
            huemixin.set_hue_values(supports_categorical=True)


class TestScale(unittest.TestCase):
    def setUp(self):
        def create_scalemixin():
            scalemixin = ScaleMixin()
            # set props the mixin is responsible for
            scalemixin.kwargs = {'scale': 'foo', 'limits': (1, 5), 'scale_func': None}
            # set props set by the plot object initializer
            scalemixin.ax = None
            scalemixin.figsize = (8, 6)
            scalemixin.extent = None
            scalemixin.projection = None

            np.random.seed(42)
            scalemixin.df = gpd.GeoDataFrame(
                {'foo': np.random.random(100), 'geometry': utils.gaussian_points(n=100)}
            )
            return scalemixin

        self.create_scalemixin = create_scalemixin

    def test_scale_init_defaults(self):
        scalemixin = self.create_scalemixin()
        scalemixin.set_scale_values()
        assert scalemixin.limits == (1, 5)
        assert len(scalemixin.scale) == 100
        assert len(scalemixin.sizes) == 100
        assert (scalemixin.sizes <= 5).all()
        assert (scalemixin.sizes >= 1).all()
        assert scalemixin.scale_func is None
        assert scalemixin.dscale is not None  # dscale is the calibrated internal scale

    def test_scale_init_scale_dtypes(self):
        # scale is initialized as a str: transform to GeoSeries
        scalemixin = self.create_scalemixin()
        scale = np.random.random(100)
        scalemixin.kwargs['scale'] = scale
        scalemixin.set_scale_values()
        assert(scalemixin.scale == scale).all()

        # scale is initialized as a GeoSeries: set as-is
        scalemixin = self.create_scalemixin()
        scale = pd.Series(np.random.random(100))
        scalemixin.kwargs['scale'] = scale
        scalemixin.set_scale_values()
        assert(scalemixin.scale == scale).all()

        # scale is initialized as a list: transform to GeoSeries
        scalemixin = self.create_scalemixin()
        scale = pd.Series(np.random.random(100))
        scalemixin.kwargs['scale'] = scale
        scalemixin.set_scale_values()
        assert(scalemixin.scale == scale).all()

        # scale is initialized as an array: transform to GeoSeries
        scalemixin = self.create_scalemixin()
        scale = np.random.random(100)
        scalemixin.kwargs['scale'] = scale
        scalemixin.set_scale_values()
        assert(scalemixin.scale == scale).all()

    def test_scale_init_scale_func(self):
        # if scale is None and scale_func is not None, raise
        scalemixin = self.create_scalemixin()
        scalemixin.kwargs['scale'] = None
        scalemixin.kwargs['scale_func'] = lambda v: v
        with pytest.raises(ValueError):
            scalemixin.set_scale_values()

        # if scale is not None and scale_func is not None, apply that func
        def identity_scale(minval, maxval):
            def scalar(val):
                return 2
            return scalar

        scalemixin = self.create_scalemixin()
        scalemixin.kwargs['scale_func'] = identity_scale
        scalemixin.set_scale_values()
        assert (scalemixin.sizes == 2).all()

    def test_scale_init_param_size_kwarg(self):
        scalemixin = self.create_scalemixin()
        scalemixin.kwargs['scale'] = None
        scalemixin.kwargs['scale_func'] = None
        scalemixin.kwargs['foosize'] = 2
        scalemixin.set_scale_values(size_kwarg='foosize')
        assert scalemixin.sizes == [2] * 100

    def test_scale_init_param_default_size(self):
        scalemixin = self.create_scalemixin()
        scalemixin.kwargs['scale'] = None
        scalemixin.kwargs['scale_func'] = None
        scalemixin.set_scale_values(default_size=2)
        assert scalemixin.sizes == [2] * 100


class TestLegend(unittest.TestCase):
    def setUp(self):
        def create_legendmixin(legend_vars):
            legendmixin = LegendMixin()
            # set props the mixin is responsible for
            legendmixin.kwargs = {
                'legend': True, 'legend_labels': None, 'legend_values': None,
                'legend_kwargs': None, 'legend_var': None
            }
            # set props controlled by the plot object initializer
            _, ax = plt.subplots(figsize=(8, 6))
            legendmixin.ax = ax
            legendmixin.figsize = (8, 6)
            legendmixin.extent = None
            legendmixin.projection = None

            # set data prop
            np.random.seed(42)
            legendmixin.df = gpd.GeoDataFrame(
                {'foo': np.random.random(100), 'geometry': utils.gaussian_points(n=100)}
            )

            # set props controlled by the hue initializer, if appropriate
            if 'hue' in legend_vars:
                legendmixin.colors = ['black'] * 100
                legendmixin.hue = legendmixin.df.foo
                legendmixin.mpl_cm_scalar_mappable = ScalarMappable(cmap='Reds')
                legendmixin.k = None
                legendmixin.categorical = False
                legendmixin.categories = None
                legendmixin.color_kwarg = 'color'
                legendmixin.default_color = 'steelblue'

            # set props controlled by the scale initializer, if appropriate
            if 'scale' in legend_vars:
                legendmixin.scale = legendmixin.df.foo
                legendmixin.limits = (1, 5)
                legendmixin.scale_func = None
                legendmixin.dscale = lambda v: 1
                legendmixin.sizes = [1] * 100

            return legendmixin

        self.create_legendmixin = create_legendmixin

    @figure_cleanup
    def test_legend_init_defaults(self):
        legendmixin = self.create_legendmixin(['hue'])
        legendmixin.paint_legend()
        # legendmixin is a painter, not a value-setter, so this is a smoke test

    @figure_cleanup
    def test_legend_invalid_inputs(self):
        # no hue or scale, but legend is True: raise
        legendmixin = self.create_legendmixin(['hue', 'scale'])
        legendmixin.hue = None
        legendmixin.scale = None
        with pytest.raises(ValueError):
            legendmixin.paint_legend(supports_scale=True, supports_hue=True)

        # only hue, but legend_var is scale: raise
        legendmixin = self.create_legendmixin(['hue', 'scale'])
        legendmixin.scale = None
        legendmixin.kwargs['legend_var'] = 'scale'
        with pytest.raises(ValueError):
            legendmixin.paint_legend(supports_scale=True, supports_hue=True)

        # # only scale, but legend_var is hue: raise
        legendmixin = self.create_legendmixin(['hue', 'scale'])
        legendmixin.hue = None
        legendmixin.kwargs['legend_var'] = 'hue'
        with pytest.raises(ValueError):
            legendmixin.paint_legend(supports_scale=True, supports_hue=True)

        # legend_var is set to an invalid input: raise
        legendmixin = self.create_legendmixin(['hue', 'scale'])
        legendmixin.kwargs['legend_var'] = 'foovar'
        with pytest.raises(ValueError):
            legendmixin.paint_legend(supports_scale=True, supports_hue=True)

        # legend if False and legend_var is not None: raise
        legendmixin = self.create_legendmixin(['hue', 'scale'])
        legendmixin.kwargs['legend'] = False
        legendmixin.kwargs['legend_var'] = 'hue'
        with pytest.raises(ValueError):
            legendmixin.paint_legend(supports_scale=True, supports_hue=True)

        # legend if False and legend_values is not None: raise
        legendmixin = self.create_legendmixin(['hue', 'scale'])
        legendmixin.kwargs['legend'] = False
        legendmixin.kwargs['legend_values'] = [1] * 5
        with pytest.raises(ValueError):
            legendmixin.paint_legend(supports_scale=True, supports_hue=True)

        # legend if False and legend_labels is not None: raise
        legendmixin = self.create_legendmixin(['hue', 'scale'])
        legendmixin.kwargs['legend'] = False
        legendmixin.kwargs['legend_labels'] = [1] * 5
        with pytest.raises(ValueError):
            legendmixin.paint_legend(supports_scale=True, supports_hue=True)

        # legend if False and legend_kwargs is not None: raise
        legendmixin = self.create_legendmixin(['hue', 'scale'])
        legendmixin.kwargs['legend'] = False
        legendmixin.kwargs['legend_kwargs'] = {'fancybox': True}
        with pytest.raises(ValueError):
            legendmixin.paint_legend(supports_scale=True, supports_hue=True)

        # legend is True, but legend_labels and legend_values are different lengths: raise
        legendmixin = self.create_legendmixin(['hue', 'scale'])
        legendmixin.kwargs['legend_values'] = [1] * 5
        legendmixin.kwargs['legend_labels'] = [1] * 4
        with pytest.raises(ValueError):
            legendmixin.paint_legend(supports_scale=True, supports_hue=True)

        # legend is in hue mode, and the user passes a markerfacecolor: raise
        legendmixin = self.create_legendmixin(['hue', 'scale'])
        legendmixin.kwargs['legend_var'] = 'hue'
        legendmixin.kwargs['legend_kwargs'] = {'markerfacecolor': 'black'}
        with pytest.raises(ValueError):
            legendmixin.paint_legend(supports_scale=True, supports_hue=True)

        # legend is in scale mode, and the user passes a markersize: raise
        legendmixin = self.create_legendmixin(['hue', 'scale'])
        legendmixin.kwargs['legend_var'] = 'scale'
        legendmixin.kwargs['legend_kwargs'] = {'markersize': 12}
        with pytest.raises(ValueError):
            legendmixin.paint_legend(supports_scale=True, supports_hue=True)

        # legend is in colorbar hue mode (e.g. k = None, legend=True, legend_var = 'hue') but
        # legend_kwargs includes marker* parameters, which can only be applied to the marker
        # style legend: raise
        legendmixin = self.create_legendmixin(['hue', 'scale'])
        legendmixin.kwargs['legend_var'] = 'hue'
        legendmixin.k = None
        legendmixin.kwargs['legend_kwargs'] = {'markerfacecolor': 'black'}
        with pytest.raises(ValueError):
            legendmixin.paint_legend(supports_scale=True, supports_hue=True)

        # legend is in colorbar mode, but legend_values is specified: raise NotImplementedError
        legendmixin = self.create_legendmixin(['hue', 'scale'])
        legendmixin.kwargs['legend_var'] = 'hue'
        legendmixin.k = None
        legendmixin.kwargs['legend_values'] = [1] * 5
        with pytest.raises(NotImplementedError):
            legendmixin.paint_legend(supports_scale=True, supports_hue=True)

        # legend is in colorbar mode, but legend_labels is specified: raise NotImplementedError
        legendmixin = self.create_legendmixin(['hue', 'scale'])
        legendmixin.kwargs['legend_var'] = 'hue'
        legendmixin.k = None
        legendmixin.kwargs['legend_labels'] = [1] * 5
        with pytest.raises(NotImplementedError):
            legendmixin.paint_legend(supports_scale=True, supports_hue=True)


class TestClip(unittest.TestCase):
    def setUp(self):
        def create_clipmixin():
            clipmixin = ClipMixin()
            clipmixin.kwargs = {
                'clip': gpd.GeoSeries(Polygon([[0, 0], [0, 100], [100, 100], [100, 0]]))
            }
            clipmixin.ax = None
            clipmixin.figsize = (8, 6)
            clipmixin.extent = None
            clipmixin.projection = None

            np.random.seed(42)
            points = utils.gaussian_points(n=2000)
            geoms = np.hstack([
                utils.gaussian_polygons(points, n=50), utils.gaussian_multi_polygons(points, n=50)
            ])
            clipmixin.df = gpd.GeoDataFrame({
                'foo': np.random.random(len(geoms)),
                'geometry': geoms,
            })
            return clipmixin

        self.create_clipmixin = create_clipmixin

    def test_clip_init_default(self):
        clipmixin = self.create_clipmixin()

        # UserWarning because we have a narrow clip
        with pytest.warns(UserWarning):
            df_result = clipmixin.set_clip(clipmixin.df)
        expected = Polygon([[0, 0], [0, 100], [100, 100], [100, 0]])
        result = df_result.geometry.unary_union.envelope
        assert expected.contains(result)


class TestWebmapInput(unittest.TestCase):
    # TODO: stub out network requests to the tile service

    def setUp(self):
        np.random.seed(42)
        p_srs = gpd.GeoSeries(utils.gaussian_points(n=100))
        self.p_df = gpd.GeoDataFrame(geometry=p_srs)

    @figure_cleanup
    def test_webmap_input_restrictions(self):
        """Test webmap-specific plot restrictions."""
        with pytest.raises(ValueError):
            webmap(self.p_df, projection=gcrs.AlbersEqualArea())

        _, ax = plt.subplots(figsize=(2, 2))
        with pytest.raises(ValueError):
            webmap(self.p_df, ax=ax)

        ax = plt.axes(projection=ccrs.PlateCarree())
        with pytest.raises(ValueError):
            webmap(self.p_df, ax=ax)

        with pytest.warns(UserWarning):
            webmap(self.p_df)
