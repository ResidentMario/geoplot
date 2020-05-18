"""
Choropleth of California districts with alternative binning schemes
===================================================================

This example demonstrates the continuous and categorical binning schemes available in ``geoplot``
on a sample dataset of California congressional districts. A binning scheme (or classifier) is a
methodology for splitting a sequence of observations into some number of bins (classes). It is also
possible to have no binning scheme, in which case the data is passed through to ``cmap`` as-is.

The options demonstrated are:

* scheme=None—A continuous colormap.
* scheme="Quantiles"—Bins the data such that the bins contain equal numbers of samples.
* scheme="EqualInterval"—Bins the data such that bins are of equal length.
* scheme="FisherJenks"—Bins the data using the Fisher natural breaks optimization
  procedure.

To learn more about colormaps in general, refer to the `Customizing Plots
<https://residentmario.github.io/geoplot/user_guide/Customizing_Plots.html#hue>`_ reference in the
documentation.

This demo showcases a small subset of the classifiers available in ``mapclassify``, the library
that ``geoplot`` relies on for this feature. To learn more about ``mapclassify``, including how
you can build your own custom ``UserDefined`` classifier, refer to `the mapclassify docs
<https://pysal.org/mapclassify/index.html>`_.
"""


import geopandas as gpd
import geoplot as gplt
import geoplot.crs as gcrs
import mapclassify as mc
import matplotlib.pyplot as plt

cali = gpd.read_file(gplt.datasets.get_path('california_congressional_districts'))
cali = cali.assign(area=cali.geometry.area)


proj=gcrs.AlbersEqualArea(central_latitude=37.16611, central_longitude=-119.44944)
fig, axarr = plt.subplots(2, 2, figsize=(12, 12), subplot_kw={'projection': proj})

gplt.choropleth(
    cali, hue='area', linewidth=0, scheme=None, ax=axarr[0][0]
)
axarr[0][0].set_title('scheme=None', fontsize=18)

import mapclassify as mc
scheme = mc.Quantiles(cali.area, k=5)
gplt.choropleth(
    cali, hue='area', linewidth=0, scheme=scheme, ax=axarr[0][1]
)
axarr[0][1].set_title('scheme="Quantiles"', fontsize=18)

scheme = mc.EqualInterval(cali.area, k=5)
gplt.choropleth(
    cali, hue='area', linewidth=0, scheme=scheme, ax=axarr[1][0]
)
axarr[1][0].set_title('scheme="EqualInterval"', fontsize=18)

scheme = mc.FisherJenks(cali.area, k=5)
gplt.choropleth(
    cali, hue='area', linewidth=0, scheme=scheme, ax=axarr[1][1]
)
axarr[1][1].set_title('scheme="FisherJenks"', fontsize=18)

plt.subplots_adjust(top=0.92)
plt.suptitle('California State Districts by Area, 2010', fontsize=18)

fig = plt.gcf()
plt.savefig("boston-airbnb-kde.png", bbox_inches='tight', pad_inches=0.1)
