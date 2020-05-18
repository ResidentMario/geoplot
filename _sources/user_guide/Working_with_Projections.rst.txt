Working with Projections
========================

This section of the tutorial discusses `map
projections <https://en.wikipedia.org/wiki/Map_projection>`__. If you
don’t know what a projection is, or are looking to learn more about how
they work in ``geoplot``, this page is for you!

I recommend following along with this tutorial interactively using
`Binder <https://mybinder.org/v2/gh/ResidentMario/geoplot/master?filepath=notebooks/tutorials/Working_with_Projections.ipynb>`__.

Projection and unprojection
---------------------------

.. code:: ipython3

    import geopandas as gpd
    import geoplot as gplt
    %matplotlib inline
    
    # load the example data
    contiguous_usa = gpd.read_file(gplt.datasets.get_path('contiguous_usa'))
    gplt.polyplot(contiguous_usa)




.. parsed-literal::

    <matplotlib.axes._subplots.AxesSubplot at 0x11a914208>




.. image:: Working_with_Projections_files/Working_with_Projections_1_1.png


This map is an example of an unprojected plot: it reproduces our
coordinates as if they were on a flat Cartesian plane. But remember, the
Earth is not a flat surface; it’s a sphere. This isn’t a map of the
United States that you’d seen in print anywhere because it badly
distorts both of the `two
criteria <http://www.geo.hunter.cuny.edu/~jochen/gtech201/lectures/lec6concepts/Map%20coordinate%20systems/How%20to%20choose%20a%20projection.htm>`__
most projections are evaluated on: *shape* and *area*.

For sufficiently small areas, the amount of distortion is very small.
This map of New York City, for example, is reasonably accurate:

.. code:: ipython3

    boroughs = gpd.read_file(gplt.datasets.get_path('nyc_boroughs'))
    gplt.polyplot(boroughs)




.. parsed-literal::

    <matplotlib.axes._subplots.AxesSubplot at 0x11d243898>




.. image:: Working_with_Projections_files/Working_with_Projections_3_1.png


But there is a better way: use a **projection**.

A projection is a way of mapping points on the surface of the Earth into
two dimensions (like a piece of paper or a computer screen). Because
moving from three dimensions to two is intrinsically lossy, no
projection is perfect, but some will definitely work better in certain
case than others.

The most common projection used for the contiguous United States is the
`Albers Equal Area
projection <https://en.wikipedia.org/wiki/Albers_projection>`__. This
projection works by wrapping the Earth around a cone, one that’s
particularly well optimized for locations near the middle of the
Northern Hemisphere (and particularly poorly for locations at the
poles).

To add a projection to a map in ``geoplot``, pass a ``geoplot.crs``
object to the ``projection`` parameter on the plot. For instance, here’s
what we get when we try ``Albers`` out on the contiguous United States:

.. code:: ipython3

    import geoplot.crs as gcrs
    gplt.polyplot(contiguous_usa, projection=gcrs.AlbersEqualArea())




.. parsed-literal::

    <cartopy.mpl.geoaxes.GeoAxesSubplot at 0x11dd02cc0>




.. image:: Working_with_Projections_files/Working_with_Projections_5_1.png


For a list of projections implemented in ``geoplot``, refer to `the
projections
reference <http://scitools.org.uk/cartopy/docs/latest/crs/projections.html>`__
in the ``cartopy`` documentation (``cartopy`` is the library ``geoplot``
relies on for its projections).

Stacking projected plots
------------------------

A key feature of ``geoplot`` is the ability to stack plots on top of one
another.

.. code:: ipython3

    cities = gpd.read_file(gplt.datasets.get_path('usa_cities'))
    ax = gplt.polyplot(
        contiguous_usa, 
        projection=gcrs.AlbersEqualArea()
    )
    gplt.pointplot(cities, ax=ax)




.. parsed-literal::

    <cartopy.mpl.geoaxes.GeoAxesSubplot at 0x11da21c50>




.. image:: Working_with_Projections_files/Working_with_Projections_8_1.png


By default, ``geoplot`` will set the
`extent <https://residentmario.github.io/geoplot/user_guide/Customizing_Plots.html#extent>`__
(the area covered by the plot) to the
`total_bounds <http://geopandas.org/reference.html#geopandas.GeoSeries.total_bounds>`__
of the last plot stacked onto the map.

However, suppose that even though we have data for One entire United
States (plus Puerto Rico) we actually want to display just data for the
contiguous United States. An easy way to get this is setting the
``extent`` parameter using ``total_bounds``.

.. code:: ipython3

    ax = gplt.polyplot(
        contiguous_usa, 
        projection=gcrs.AlbersEqualArea()
    )
    gplt.pointplot(cities, ax=ax, extent=contiguous_usa.total_bounds)




.. parsed-literal::

    <cartopy.mpl.geoaxes.GeoAxesSubplot at 0x11da947f0>




.. image:: Working_with_Projections_files/Working_with_Projections_10_1.png


The section of the tutorial on `Customizing
Plots <https://nbviewer.jupyter.org/github/ResidentMario/geoplot/blob/master/notebooks/tutorials/Customizing%20Plots.ipynb#Extent>`__
explains the ``extent`` parameter in more detail.

Projections on subplots
-----------------------

It is possible to compose multiple axes together into a single panel
figure in ``matplotlib`` using the ``subplots`` feature. This feature is
highly useful for creating side-by-side comparisons of your plots, or
for stacking your plots together into a single more informative display.

.. code:: ipython3

    import matplotlib.pyplot as plt
    import geoplot as gplt
    
    f, axarr = plt.subplots(1, 2, figsize=(12, 4))
    gplt.polyplot(contiguous_usa, ax=axarr[0])
    gplt.polyplot(contiguous_usa, ax=axarr[1])




.. parsed-literal::

    <matplotlib.axes._subplots.AxesSubplot at 0x11dc55438>




.. image:: Working_with_Projections_files/Working_with_Projections_13_1.png


``matplotlib`` supports subplotting projected maps using the
``projection`` argument to ``subplot_kw``.

.. code:: ipython3

    proj = gcrs.AlbersEqualArea(central_longitude=-98, central_latitude=39.5)
    f, axarr = plt.subplots(1, 2, figsize=(12, 4), subplot_kw={
        'projection': proj
    })
    gplt.polyplot(contiguous_usa, projection=proj, ax=axarr[0])
    gplt.polyplot(contiguous_usa, projection=proj, ax=axarr[1])




.. parsed-literal::

    <cartopy.mpl.geoaxes.GeoAxesSubplot at 0x11ded2b70>




.. image:: Working_with_Projections_files/Working_with_Projections_15_1.png


The
`Gallery <https://residentmario.github.io/geoplot/gallery/index.html>`__
includes several demos, like the `Pointplot Scale
Functions <https://residentmario.github.io/geoplot/gallery/plot_usa_city_elevations.html#sphx-glr-gallery-plot-usa-city-elevations-py>`__
demo, that use this feature to good effect.

Notice that in this code sample we specified some additional parameters
for our projection. The ``central_longitude=-98`` and
``central_latitude=39.5`` parameters set the “center point” around which
the points and shapes on the map are reprojected (in this case we use
the `geographic center of the contiguous United
States <https://en.wikipedia.org/wiki/Geographic_center_of_the_contiguous_United_States>`__).

When you pass a projection to a ``geoplot`` function, ``geoplot`` will
infer these values for you. But when passing the projection directly to
``matplotlib`` you must set them yourself.
