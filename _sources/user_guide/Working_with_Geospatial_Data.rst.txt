
Working with Geospatial Data
============================

This section of the tutorial discusses how to use ``geopandas`` and
``shapely`` to manipulate geospatial data in Python. If you’ve never
used these libraries before, or are looking for a refresher on how they
work, this page is for you!

I recommend following along with this tutorial interactively using
`Binder <https://mybinder.org/v2/gh/ResidentMario/geoplot/master?filepath=notebooks/tutorials/Working_with_Geospatial_Data.ipynb>`__.

Coordinate reference systems
----------------------------

The ``GeoDataFrame`` is an augmented version of a ``pandas``
``DataFrame`` with an attached geometry:

.. code:: ipython3

    import pandas as pd; pd.set_option('max_columns', 6)  # Unclutter display.
    import geopandas as gpd
    import geoplot as gplt
    
    # load the example data
    nyc_boroughs = gpd.read_file(gplt.datasets.get_path('nyc_boroughs'))
    nyc_boroughs




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>BoroCode</th>
          <th>BoroName</th>
          <th>Shape_Leng</th>
          <th>Shape_Area</th>
          <th>geometry</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>5</td>
          <td>Staten Island</td>
          <td>330385.03697</td>
          <td>1.623853e+09</td>
          <td>(POLYGON ((-74.05050806403247 40.5664220341608...</td>
        </tr>
        <tr>
          <th>1</th>
          <td>4</td>
          <td>Queens</td>
          <td>861038.47930</td>
          <td>3.049947e+09</td>
          <td>(POLYGON ((-73.83668274106708 40.5949466970158...</td>
        </tr>
        <tr>
          <th>2</th>
          <td>3</td>
          <td>Brooklyn</td>
          <td>726568.94634</td>
          <td>1.959432e+09</td>
          <td>(POLYGON ((-73.8670614947212 40.58208797679338...</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1</td>
          <td>Manhattan</td>
          <td>358532.95642</td>
          <td>6.364422e+08</td>
          <td>(POLYGON ((-74.01092841268033 40.6844914725429...</td>
        </tr>
        <tr>
          <th>4</th>
          <td>2</td>
          <td>Bronx</td>
          <td>464517.89055</td>
          <td>1.186804e+09</td>
          <td>(POLYGON ((-73.89680883223775 40.7958084451597...</td>
        </tr>
      </tbody>
    </table>
    </div>



.. raw:: html

   <div style="margin-top:2em">

Most operations that will work on a ``pandas`` ``DataFrame`` will work
on a ``GeoDataFrame``, but the latter adds a few additional methods and
features for dealing with geometry not present in the former. The most
obvious of these is the addition of a column for storing geometries,
accessible using the ``geometry`` attribute:

.. raw:: html

   </div>

.. code:: ipython3

    nyc_boroughs.geometry




.. parsed-literal::

    0    (POLYGON ((-74.05050806403247 40.5664220341608...
    1    (POLYGON ((-73.83668274106708 40.5949466970158...
    2    (POLYGON ((-73.8670614947212 40.58208797679338...
    3    (POLYGON ((-74.01092841268033 40.6844914725429...
    4    (POLYGON ((-73.89680883223775 40.7958084451597...
    Name: geometry, dtype: object



Whenever you work with novel geospatial data in a ``GeoDataFrame``, the
first thing you should do is check its **coordinate reference system**.

A `coordinate reference
system <https://en.wikipedia.org/wiki/Spatial_reference_system>`__, or
CRS, is a system for defining where points in space are. You can extract
what CRS your polygons are stored in using the ``crs`` attribute:

.. code:: ipython3

    nyc_boroughs.crs




.. parsed-literal::

    {'init': 'epsg:4326'}



In this case ``epsg:4326`` is the official identifier for what the rest
of us more commonly refer to as “longitude and latitude”. Most
coordinate reference systems have a well-defined EPSG number, which you
can look up using the handy
`spatialreference.org <http://spatialreference.org/ref/epsg/wgs-84/>`__
website.

Why do coordinate reference systems besides latitude-longitude even
exist? As an example, the United States Geolocial Service, which
maintains extremely high-accuracy maps of the United States, maintains
110 coordinate reference systems, refered to as “state plane coordinate
systems”, for various portions of the United States. Latitude-longitude
uses `spherical
coordinates <https://en.wikipedia.org/wiki/Spherical_coordinate_system>`__;
state plane coordinate systems use “flat-Earth” `Cartesian
coordinate <https://en.wikipedia.org/wiki/Cartesian_coordinate_system>`__.
State plane coordinates are therefore much simpler to work with
computationally, while remaining accurate enough (within their “zone”)
for most applications.

For this reason, state plane coordinate systems remain in use throughout
government. For example, here’s a sample of data taken from the MapPLUTO
dataset released by the City of New York:

.. code:: ipython3

    nyc_map_pluto_sample = gpd.read_file(gplt.datasets.get_path('nyc_map_pluto_sample'))
    nyc_map_pluto_sample




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>Borough</th>
          <th>Block</th>
          <th>Lot</th>
          <th>...</th>
          <th>Shape_Leng</th>
          <th>Shape_Area</th>
          <th>geometry</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>MN</td>
          <td>1</td>
          <td>10</td>
          <td>...</td>
          <td>12277.824113</td>
          <td>7.550340e+06</td>
          <td>POLYGON ((979561.8712409735 191884.2491553128,...</td>
        </tr>
        <tr>
          <th>1</th>
          <td>MN</td>
          <td>1</td>
          <td>101</td>
          <td>...</td>
          <td>3940.840373</td>
          <td>5.018974e+05</td>
          <td>POLYGON ((972382.8255597204 190647.2667211443,...</td>
        </tr>
        <tr>
          <th>2</th>
          <td>MN</td>
          <td>1</td>
          <td>101</td>
          <td>...</td>
          <td>3940.840373</td>
          <td>5.018974e+05</td>
          <td>POLYGON ((972428.8290766329 190679.1751218885,...</td>
        </tr>
        <tr>
          <th>3</th>
          <td>MN</td>
          <td>1</td>
          <td>101</td>
          <td>...</td>
          <td>3940.840373</td>
          <td>5.018974e+05</td>
          <td>POLYGON ((972058.3399882168 190689.2800885588,...</td>
        </tr>
        <tr>
          <th>4</th>
          <td>MN</td>
          <td>1</td>
          <td>201</td>
          <td>...</td>
          <td>6306.268341</td>
          <td>1.148539e+06</td>
          <td>POLYGON ((973154.7118112147 194614.3312935531,...</td>
        </tr>
        <tr>
          <th>5</th>
          <td>MN</td>
          <td>2</td>
          <td>1</td>
          <td>...</td>
          <td>2721.060649</td>
          <td>1.008250e+05</td>
          <td>POLYGON ((980915.0020648837 194319.1402828991,...</td>
        </tr>
        <tr>
          <th>6</th>
          <td>MN</td>
          <td>2</td>
          <td>2</td>
          <td>...</td>
          <td>2411.869687</td>
          <td>8.724423e+04</td>
          <td>POLYGON ((981169.004181549 194678.8213220537, ...</td>
        </tr>
      </tbody>
    </table>
    <p>7 rows × 90 columns</p>
    </div>



This data is stored in the Long Island State Plane coordinate reference
system (`EPSG
2263 <https://www.spatialreference.org/ref/epsg/2263/>`__).
Unfortunately the CRS on read is set incorrectly to ``epsg:4326`` and we
have to set it to the correct coordinate reference system ourselves.

.. code:: ipython3

    nyc_map_pluto_sample.crs = {'init': 'epsg:2263'}
    nyc_map_pluto_sample.crs




.. parsed-literal::

    {'init': 'epsg:2263'}



Depending on the dataset, ``crs`` may be set to either ``epsg:<INT>`` or
to a raw `proj4 <https://github.com/OSGeo/PROJ>`__ projection
dictionary. The bottom line is, after reading in a dataset, always
verify that the dataset coordinate reference system is set to what its
documentation it should be set to.

If you determine that your coordinates are not latitude-longitude,
usually the first thing you want to do is covert to it. ``to_crs`` does
this:

.. code:: ipython3

    nyc_map_pluto_sample = nyc_map_pluto_sample.to_crs(epsg=4326)
    nyc_map_pluto_sample




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>Borough</th>
          <th>Block</th>
          <th>Lot</th>
          <th>...</th>
          <th>Shape_Leng</th>
          <th>Shape_Area</th>
          <th>geometry</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>MN</td>
          <td>1</td>
          <td>10</td>
          <td>...</td>
          <td>12277.824113</td>
          <td>7.550340e+06</td>
          <td>POLYGON ((-74.0169058260488 40.69335342975063,...</td>
        </tr>
        <tr>
          <th>1</th>
          <td>MN</td>
          <td>1</td>
          <td>101</td>
          <td>...</td>
          <td>3940.840373</td>
          <td>5.018974e+05</td>
          <td>POLYGON ((-74.04279194703045 40.68995148413111...</td>
        </tr>
        <tr>
          <th>2</th>
          <td>MN</td>
          <td>1</td>
          <td>101</td>
          <td>...</td>
          <td>3940.840373</td>
          <td>5.018974e+05</td>
          <td>POLYGON ((-74.04262611856618 40.69003912689961...</td>
        </tr>
        <tr>
          <th>3</th>
          <td>MN</td>
          <td>1</td>
          <td>101</td>
          <td>...</td>
          <td>3940.840373</td>
          <td>5.018974e+05</td>
          <td>POLYGON ((-74.04396208819837 40.69006636010664...</td>
        </tr>
        <tr>
          <th>4</th>
          <td>MN</td>
          <td>1</td>
          <td>201</td>
          <td>...</td>
          <td>6306.268341</td>
          <td>1.148539e+06</td>
          <td>POLYGON ((-74.04001513069795 40.7008411559464,...</td>
        </tr>
        <tr>
          <th>5</th>
          <td>MN</td>
          <td>2</td>
          <td>1</td>
          <td>...</td>
          <td>2721.060649</td>
          <td>1.008250e+05</td>
          <td>POLYGON ((-74.01202751677701 40.70003725302833...</td>
        </tr>
        <tr>
          <th>6</th>
          <td>MN</td>
          <td>2</td>
          <td>2</td>
          <td>...</td>
          <td>2411.869687</td>
          <td>8.724423e+04</td>
          <td>POLYGON ((-74.01111163437271 40.70102458543801...</td>
        </tr>
      </tbody>
    </table>
    <p>7 rows × 90 columns</p>
    </div>



Coordinate order
----------------

``shapely``, the library ``geopandas`` uses to store its geometries,
uses “modern” longitude-latitude ``(x, y)`` coordinate order. This
differs from the “historical” latitude-longitude ``(y, x)`` coordinate
order. Datasets “in the wild” may be in either format.

There is no way for ``geopandas`` to know whether a dataset is in one
format or the other at load time. Once you have converted your dataset
to the right coordinate system, always always always make sure to next
check that the geometries are also in the right coordinate order.

This is an easy mistake to make and people are making it constantly!

The fastest way to ensure that coordinates are in the right order is to
know what the right x coordinates and y coordinates for your data should
be and eyeball it.

Types of geometries
-------------------

Every element of the ``geometry`` column in a ``GeoDataFrame`` is a
``shapely`` object. `Shapely <https://github.com/Toblerity/Shapely>`__
is a geometric operations library which is used for manipulating
geometries in space, and it’s the Python API of choice for working with
shape data.

``shapely`` defines just a handful of types of geometries:

-  ``Point``—a point.
-  ``MultiPoint``—a set of points.
-  ``LineString``—a line segment.
-  ``MultiLineString``—a collection of lines (e.g. a sequence of
   connected line segments).
-  ``LinearRing``—a closed collection of lines. Basically a polygon with
   zero-area.
-  ``Polygon``—an closed shape along a sequence of points.
-  ``MultiPolygon``—a collection of polygons.

You can check the ``type`` of a geometry using the ``type`` operator:

.. code:: ipython3

    type(nyc_boroughs.geometry.iloc[0])




.. parsed-literal::

    shapely.geometry.multipolygon.MultiPolygon



.. code:: ipython3

    type(nyc_map_pluto_sample.geometry.iloc[0])




.. parsed-literal::

    shapely.geometry.polygon.Polygon



Performing geometric operations
-------------------------------

The `shapely user
manual <https://shapely.readthedocs.io/en/latest/manual.html>`__
provides an extensive list of geometric operations that you can perform
using the library: from simple things like translations and
transformations to more complex operations like polygon buffering.

You can apply transformations to your geometries in an object-by-object
way by using the native ``pandas`` ``map`` function on the ``geometry``
column. For example, here is one way of deconstructing a set of
``Polygon`` or ``MultiPolygon`` objects into simplified `convex
hulls <https://en.wikipedia.org/wiki/Convex_hull>`__:

.. code:: ipython3

    %time gplt.polyplot(nyc_boroughs.geometry.map(lambda shp: shp.convex_hull))


.. parsed-literal::

    CPU times: user 62.7 ms, sys: 2.64 ms, total: 65.3 ms
    Wall time: 71.6 ms




.. parsed-literal::

    <matplotlib.axes._subplots.AxesSubplot at 0x11c61d7b8>




.. image:: Working_with_Geospatial_Data_files/Working_with_Geospatial_Data_18_2.png


You can perform arbitrarily complex geometric transformations on your
shapes this way. However, `most common
operations <http://geopandas.org/geometric_manipulations.html>`__ are
provided in optimized form as part of the ``geopandas`` API. Here’s a
faster way to create convex hulls, for example:

.. code:: ipython3

    %time nyc_boroughs.convex_hull


.. parsed-literal::

    CPU times: user 55.6 ms, sys: 6.45 ms, total: 62.1 ms
    Wall time: 39.9 ms




.. parsed-literal::

    0    POLYGON ((-74.24712436215984 40.49611539517034...
    1    POLYGON ((-73.94073681665428 40.54182008715522...
    2    POLYGON ((-73.98336058039274 40.56952999448672...
    3    POLYGON ((-74.02305574749596 40.68291694544512...
    4    POLYGON ((-73.87830680057651 40.78535662050845...
    dtype: object



It is beyond the scope of this short guide to dive too deeply into
geospatial data transformations. Suffice to say that there are many of
them, and that you can learn some more about them by consulting the
`geopandas <http://geopandas.org/>`__ and
`shapely <https://toblerity.org/shapely/manual.html>`__ documentation.

Defining your own geometries
----------------------------

In this section of the tutorial, we will focus on one particular aspect
of ``shapely`` which is likely to come up: defining your own geometries.

In the cases above we read a GeoDataFrame straight out of geospatial
files: our borough information was stored in the
`GeoJSON <http://geojson.org/>`__ format, while our building footprints
were a `Shapefile <https://en.wikipedia.org/wiki/Shapefile>`__. What if
we have geospatial data embedded in an ordinary ``CSV`` or ``JSON``
file, which read into an ordinary ``pandas`` ``DataFrame``?

.. code:: ipython3

    nyc_collisions_sample = pd.read_csv(gplt.datasets.get_path('nyc_collisions_sample'))
    nyc_collisions_sample




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>LATITUDE</th>
          <th>LONGITUDE</th>
          <th>DATE</th>
          <th>TIME</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>40.767373</td>
          <td>-73.950057</td>
          <td>04/16/2016</td>
          <td>4:13</td>
        </tr>
        <tr>
          <th>1</th>
          <td>40.862670</td>
          <td>-73.909039</td>
          <td>04/16/2016</td>
          <td>4:30</td>
        </tr>
        <tr>
          <th>2</th>
          <td>40.716507</td>
          <td>-73.961275</td>
          <td>04/16/2016</td>
          <td>4:30</td>
        </tr>
        <tr>
          <th>3</th>
          <td>40.749788</td>
          <td>-73.987768</td>
          <td>04/16/2016</td>
          <td>4:30</td>
        </tr>
        <tr>
          <th>4</th>
          <td>40.702401</td>
          <td>73.960496</td>
          <td>04/16/2016</td>
          <td>4:50</td>
        </tr>
      </tbody>
    </table>
    </div>



.. raw:: html

   <div style="margin-top:2em">

It is extremely common for datasets containing light geospatial data
(e.g. points, maybe line segments, but usually not whole polygons) to be
saved in a non-geospatial formats.

In this case can import ``shapely`` directly, use it to define our own
geometries, then initialize a ``GeoDataFrame``. The ``pandas`` ``apply``
function is the best to do this:

.. raw:: html

   </div>

.. code:: ipython3

    from shapely.geometry import Point
    
    collision_points = nyc_collisions_sample.apply(
        lambda srs: Point(float(srs['LONGITUDE']), float(srs['LATITUDE'])),
        axis='columns'
    )
    collision_points




.. parsed-literal::

    0           POINT (-73.950057 40.767373)
    1    POINT (-73.90903900000001 40.86267)
    2           POINT (-73.961275 40.716507)
    3           POINT (-73.987768 40.749788)
    4    POINT (73.96049599999999 40.702401)
    dtype: object



From there we pass this iterable of geometries to the ``geometry``
property of the ``GeoDataFrame`` initializer:

.. code:: ipython3

    import geopandas as gpd
    nyc_collisions_sample_geocoded = gpd.GeoDataFrame(nyc_collisions_sample, geometry=collision_points)
    nyc_collisions_sample_geocoded




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>LATITUDE</th>
          <th>LONGITUDE</th>
          <th>DATE</th>
          <th>TIME</th>
          <th>geometry</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>40.767373</td>
          <td>-73.950057</td>
          <td>04/16/2016</td>
          <td>4:13</td>
          <td>POINT (-73.950057 40.767373)</td>
        </tr>
        <tr>
          <th>1</th>
          <td>40.862670</td>
          <td>-73.909039</td>
          <td>04/16/2016</td>
          <td>4:30</td>
          <td>POINT (-73.90903900000001 40.86267)</td>
        </tr>
        <tr>
          <th>2</th>
          <td>40.716507</td>
          <td>-73.961275</td>
          <td>04/16/2016</td>
          <td>4:30</td>
          <td>POINT (-73.961275 40.716507)</td>
        </tr>
        <tr>
          <th>3</th>
          <td>40.749788</td>
          <td>-73.987768</td>
          <td>04/16/2016</td>
          <td>4:30</td>
          <td>POINT (-73.987768 40.749788)</td>
        </tr>
        <tr>
          <th>4</th>
          <td>40.702401</td>
          <td>73.960496</td>
          <td>04/16/2016</td>
          <td>4:50</td>
          <td>POINT (73.96049599999999 40.702401)</td>
        </tr>
      </tbody>
    </table>
    </div>



.. raw:: html

   <div style="margin-top:2em">

In most cases, data with geospatial information provided in a CSV will
be point data corresponding with individual coordinates. Sometimes,
however, one may wish to define more complex geometry: square areas, for
example, and *maybe* even complex polygons. While we won’t cover these
cases, they’re quite similar to the extremely simple point case we’ve
shown here. For further reference on such a task, refer to the
``shapely`` documentation.

.. raw:: html

   </div>

Joining on existing geometries
------------------------------

Sometimes the necessary geospatial data is elsewhere entirely.

Suppose now that we have information on obesity by state.

.. code:: ipython3

    obesity = pd.read_csv(gplt.datasets.get_path('obesity_by_state'), sep='\t')
    obesity.head()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>State</th>
          <th>Percent</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>Alabama</td>
          <td>32.4</td>
        </tr>
        <tr>
          <th>1</th>
          <td>Missouri</td>
          <td>30.4</td>
        </tr>
        <tr>
          <th>2</th>
          <td>Alaska</td>
          <td>28.4</td>
        </tr>
        <tr>
          <th>3</th>
          <td>Montana</td>
          <td>24.6</td>
        </tr>
        <tr>
          <th>4</th>
          <td>Arizona</td>
          <td>26.8</td>
        </tr>
      </tbody>
    </table>
    </div>



.. raw:: html

   <div style="margin-top:2em">

We’d like to put this information on a map. But we don’t have any
geometry!

We will once again have to define a geometry. Except that this time,
instead of writing our own, we will need to find data with state shapes,
and join that data against this data. In other cases there may be other
shapes: police precincts, survey zones, and so on. Here is just such a
dataset:

.. raw:: html

   </div>

.. code:: ipython3

    contiguous_usa = gpd.read_file(gplt.datasets.get_path('contiguous_usa'))
    contiguous_usa.head()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>state</th>
          <th>adm1_code</th>
          <th>population</th>
          <th>geometry</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>Minnesota</td>
          <td>USA-3514</td>
          <td>5303925</td>
          <td>POLYGON ((-89.59940899999999 48.010274, -89.48...</td>
        </tr>
        <tr>
          <th>1</th>
          <td>Montana</td>
          <td>USA-3515</td>
          <td>989415</td>
          <td>POLYGON ((-111.194189 44.561156, -111.291548 4...</td>
        </tr>
        <tr>
          <th>2</th>
          <td>North Dakota</td>
          <td>USA-3516</td>
          <td>672591</td>
          <td>POLYGON ((-96.601359 46.351357, -96.5389080000...</td>
        </tr>
        <tr>
          <th>3</th>
          <td>Idaho</td>
          <td>USA-3518</td>
          <td>1567582</td>
          <td>POLYGON ((-111.049728 44.488163, -111.050245 4...</td>
        </tr>
        <tr>
          <th>4</th>
          <td>Washington</td>
          <td>USA-3519</td>
          <td>6724540</td>
          <td>POLYGON ((-116.998073 46.33017, -116.906528 46...</td>
        </tr>
      </tbody>
    </table>
    </div>



.. raw:: html

   <div style="margin-top:2em">

A simple ``join`` solves the problem:

.. raw:: html

   </div>

.. code:: ipython3

    result = contiguous_usa.set_index('state').join(obesity.set_index('State'))
    result.head()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>adm1_code</th>
          <th>population</th>
          <th>geometry</th>
          <th>Percent</th>
        </tr>
        <tr>
          <th>state</th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>Minnesota</th>
          <td>USA-3514</td>
          <td>5303925</td>
          <td>POLYGON ((-89.59940899999999 48.010274, -89.48...</td>
          <td>25.5</td>
        </tr>
        <tr>
          <th>Montana</th>
          <td>USA-3515</td>
          <td>989415</td>
          <td>POLYGON ((-111.194189 44.561156, -111.291548 4...</td>
          <td>24.6</td>
        </tr>
        <tr>
          <th>North Dakota</th>
          <td>USA-3516</td>
          <td>672591</td>
          <td>POLYGON ((-96.601359 46.351357, -96.5389080000...</td>
          <td>31.0</td>
        </tr>
        <tr>
          <th>Idaho</th>
          <td>USA-3518</td>
          <td>1567582</td>
          <td>POLYGON ((-111.049728 44.488163, -111.050245 4...</td>
          <td>29.6</td>
        </tr>
        <tr>
          <th>Washington</th>
          <td>USA-3519</td>
          <td>6724540</td>
          <td>POLYGON ((-116.998073 46.33017, -116.906528 46...</td>
          <td>27.2</td>
        </tr>
      </tbody>
    </table>
    </div>



.. raw:: html

   <div style="margin-top:2em">

Now we can plot it:

.. raw:: html

   </div>

.. code:: ipython3

    import geoplot.crs as gcrs
    gplt.cartogram(result, scale='Percent', projection=gcrs.AlbersEqualArea())




.. parsed-literal::

    <cartopy.mpl.geoaxes.GeoAxesSubplot at 0x11c3bda20>




.. image:: Working_with_Geospatial_Data_files/Working_with_Geospatial_Data_36_1.png


Save formats
------------

You can read data out of a geospatial file format using
``GeoDataFrame.from_file``. You can write data to a geospatial file
format using ``GeoDataFrame.to_file``. By default, these methods will
infer the file format and save to a ``Shapefile``, respectively. To
specify an explicit file format, pass the name of that format to the
``driver`` argument. For example:

.. code:: python

   nyc_boroughs.to_file('boroughs.geojson', driver='GeoJSON')

The simplest and increasingly most common save format for geospatial
data is `GeoJSON <https://geojson.org/>`__. A geojson file may have a
``.geojson`` or ``.json`` extension, and stores data in a human-readable
format:

::

   {
     "type": "Feature",
     "geometry": {
       "type": "Point",
       "coordinates": [125.6, 10.1]
     },
     "properties": {
       "name": "Dinagat Islands"
     }
   }

Historically speaking, the most common geospatial data format is the
`Shapefile <https://en.wikipedia.org/wiki/Shapefile>`__. Shapefiles are
not actually really files, but instead groups of files in a folder or
``zip`` archive that together can encode very complex information about
your data. Shapefiles are a binary file format, so they are not
human-readable like GeoJSON files are, but can efficiently encode data
too complex for easy storage in a GeoJSON.

These are the two best-known file formats, but there are `many many
others <https://en.wikipedia.org/wiki/GIS_file_formats>`__. For a list
of geospatial file formats supported by ``geopandas`` refer to the
`fiona user
manual <https://fiona.readthedocs.io/en/latest/manual.html>`__.
