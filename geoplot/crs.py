"""
This module defines the ``geoplot`` coordinate reference system classes, wrappers on
``cartopy.crs`` objects meant to be used as parameters to the ``projection`` parameter of all
front-end ``geoplot`` outputs. For the list of Cartopy CRS objects this module derives from,
refer to http://scitools.org.uk/cartopy/docs/latest/crs/projections.html.
"""

import cartopy.crs as ccrs
import geopandas as gpd


class Base:
    # TODO: RotatedPole
    """
    Generate instances of ``cartopy.crs``.*name* where *name* matches the instance's class name.

    Parameters
    ----------
    `load` : Return a Cartopy CRS initialized with defaults from the `centerings` dictionary,
    overridden by initialization parameters.

    `_as_mpl_axes` : Return the result of calling cartopy's ``_as_mpl_axes`` for `self.load`
    called with empty `df` and `centerings`.
    """
    def __init__(self, **kwargs):
        """Save parameters that initialize Cartopy CRSs."""
        self.args = kwargs

    def load(self, df, centerings):
        """
        A meta-method which abstracts the internals of individual projections' load procedures.

        Parameters
        ----------
        df : GeoDataFrame
            The GeoDataFrame which has been passed as input to the plotter at the top level.
            This data is needed to calculate reasonable centering variables in cases in which the
            user does not already provide them; which is, incidentally, the reason behind all of
            this funny twice-instantiation loading in the first place.
        centerings: dict
            A dictionary containing names and centering methods. Certain projections have certain
            centering parameters whilst others lack them. For example, the geospatial projection
            contains both ``central_longitude`` and ``central_latitude`` instance parameter, which
            together control the center of the plot, while the North Pole Stereo projection has
            only a ``central_longitude`` instance parameter, implying that latitude is fixed (as
            indeed it is, as this projection is centered on the North Pole!).

            A top-level centerings method is provided in each of the ``geoplot`` top-level plot
            functions; each of the projection wrapper classes defined here in turn selects the
            functions from this list relevent to this particular instance and passes them to
            the ``_generic_load`` method here.

            We then in turn execute these functions to get defaults for our ``df`` and pass them
            off to our output ``cartopy.crs`` instance.

        Returns
        -------
        crs : ``cartopy.crs`` object instance
            Returns a ``cartopy.crs`` object instance whose appropriate instance variables have
            been set to reasonable defaults wherever not already provided by the user.
        """
        return getattr(ccrs, self.__class__.__name__)(**{**centerings, **self.args})

    def _as_mpl_axes(self):
        """
        When ``matplotlib`` is provided a projection via a ``projection`` keyword argument, it
        expects to get something with a callable ``as_mpl_axes`` method. The precise details of
        what this method does, exactly, are not important: it suffices to know that every
        ``cartopy`` coordinate reference system object has one.

        When we pass a ``geoplot.crs`` crs object to a ``geoplot`` function, the loading and
        centering of the data occurs automatically (using the function defined immediately above).
        Since we control what ``geoplot`` does at execution, we gracefully integrate this two-step
        procedure into the function body.

        But there are also use cases outside of our control in which we are forced to pass a
        ``geoplot.crs`` object without having first called ``load``: most prominently, when
        creating a plot containing subplots, the "overall" projection must be pre-loaded. It's
        possible to get around this by using ``cartopy.crs`` objects instead, but this is
        inelegant. This method is a better way: when a ``geoplot.crs`` object called by
        ``matplotlib``, it silently swaps itself out for a vanilla version of its ``cartopy.crs``
        mirror, and calls that function's ``_as_mpl_axes`` instead.

        Parameters
        ----------
        proj : geoplot.crs projection instance
            The instance in question (self, in the method body).

        Returns
        -------
        Mutates into a ``cartopy.crs`` object and returns the result of executing ``_as_mpl_axes``
        on that object instead.
        """
        proj = self.load(gpd.GeoDataFrame(), dict())
        return proj._as_mpl_axes()


class Filtering(Base):
    """CRS that `load`s with `centering` restricted to keys in `self.filter_`."""

    def load(self, df, centerings):
        """Call `load` method with `centerings` filtered to keys in `self.filter_`."""
        return super().load(
            df,
            {key: value
             for key, value in centerings.items()
             if key in self.filter_}
        )


class LongitudeCentering(Filtering):
    """Form a CRS that centers by longitude."""
    filter_ = {'central_longitude'}


class LatitudeCentering(Filtering):
    """For a CRS that centers by latitude."""
    filter_ = {'central_latitude'}


PlateCarree,\
    LambertCylindrical,\
    Mercator,\
    Miller,\
    Mollweide,\
    Robinson,\
    Sinusoidal,\
    InterruptedGoodeHomolosine,\
    Geostationary,\
    NorthPolarStereo,\
    SouthPolarStereo = tuple(
        type(name, (LongitudeCentering,), {})
        for name in ('PlateCarree',
                     'LambertCylindrical',
                     'Mercator',
                     'Miller',
                     'Mollweide',
                     'Robinson',
                     'Sinusoidal',
                     'InterruptedGoodeHomolosine',
                     'Geostationary',
                     'NorthPolarStereo',
                     'SouthPolarStereo')
)

Gnomonic = type('Gnomonic', (LatitudeCentering,), {})

AlbersEqualArea,\
    AzimuthalEquidistant,\
    LambertConformal,\
    Orthographic,\
    Stereographic,\
    TransverseMercator,\
    LambertAzimuthalEqualArea,\
    UTM,\
    OSGB,\
    EuroPP,\
    OSNI = tuple(
        type(name, (Base,), {})
        for name in ('AlbersEqualArea',
                     'AzimuthalEquidistant',
                     'LambertConformal',
                     'Orthographic',
                     'Stereographic',
                     'TransverseMercator',
                     'LambertAzimuthalEqualArea',
                     'UTM',
                     'OSGB',
                     'EuroPP',
                     'OSNI')
    )
