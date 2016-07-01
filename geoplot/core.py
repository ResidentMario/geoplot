# import folium
import numpy as np
# import pandas as pd
import functools
# import mplleaflet


# def geoplot(self):
#     pass

class FramePlotMethods():
    """
    Framing object for all geoplotting methods. Aliased to geoplot at runtime.
    """

    def __init__(self, kind='point', **kwargs):
        """
        This method is run when GeoDataFrame.geoplot() is executed directly. It wraps the call and sends it to the
        appropriate plotter method.

        :param kind:
        :param lat:
        :param long:
        :param kwargs:
        """
        caller = getattr(FramePlotMethods, kind)
        caller(**kwargs)


    @staticmethod
    def _initialize_folium_layer(gdf, **kwargs):
        """
        Generates the folium layer for a plot. Given a GeoDataFrame containing points that we expect to appear in
        the plot, computes the map center and map zoom level.

        Returns a folium.Map() object.
        """
        bounds = gdf['geometry'].bounds
        minx, miny = np.min(bounds['minx']), np.min(bounds['miny'])
        maxx, maxy = np.max(bounds['maxx']), np.max(bounds['maxy'])
        center = ((minx + miny) / 2, (maxx + maxy) / 2)
        # geoplot = folium.Map(location=center, zoom_start=13)
        # return geoplot
        return

    def point(self, **kwargs):
        print("Hello!")
        pass