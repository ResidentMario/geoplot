
Working with Geospatial Data
----------------------------

In order to understand how to use ``geoplot``, we need to understand a
bit about the format it expects to recieves its data in: a ``geopandas``
``GeoDataFrame``.

The ``GeoDataFrame`` is an augmented version of a ``pandas``
``DataFrame`` with an attached geometry:

.. code:: python

    import pandas as pd; pd.set_option('max_columns', 6)  # Unclutter display.
    
    import geopandas as gpd
    boroughs = gpd.read_file("../../data/nyc_boroughs/boroughs.geojson", driver='GeoJSON')
    boroughs




.. raw:: html

    <div>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>BoroCode</th>
          <th>BoroName</th>
          <th>Shape_Area</th>
          <th>Shape_Leng</th>
          <th>geometry</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>5</td>
          <td>Staten Island</td>
          <td>1.623853e+09</td>
          <td>330385.03697</td>
          <td>(POLYGON ((-74.05050806403247 40.5664220341608...</td>
        </tr>
        <tr>
          <th>1</th>
          <td>4</td>
          <td>Queens</td>
          <td>3.049947e+09</td>
          <td>861038.47930</td>
          <td>(POLYGON ((-73.83668274106708 40.5949466970158...</td>
        </tr>
        <tr>
          <th>2</th>
          <td>3</td>
          <td>Brooklyn</td>
          <td>1.959432e+09</td>
          <td>726568.94634</td>
          <td>(POLYGON ((-73.8670614947212 40.58208797679338...</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1</td>
          <td>Manhattan</td>
          <td>6.364422e+08</td>
          <td>358532.95642</td>
          <td>(POLYGON ((-74.01092841268033 40.6844914725429...</td>
        </tr>
        <tr>
          <th>4</th>
          <td>2</td>
          <td>Bronx</td>
          <td>1.186804e+09</td>
          <td>464517.89055</td>
          <td>(POLYGON ((-73.89680883223775 40.7958084451597...</td>
        </tr>
      </tbody>
    </table>
    </div>



Any operation that will work on a ``DataFrame`` will work on a
``GeoDataFrame``, but the latter adds a few additional methods and
features for dealing with geometry not present in the former. The most
obvious of these is the addition of a column for storing geometries,
accessible using the ``geometry`` attribute:

.. code:: python

    boroughs.geometry




.. parsed-literal::

    0    (POLYGON ((-74.05050806403247 40.5664220341608...
    1    (POLYGON ((-73.83668274106708 40.5949466970158...
    2    (POLYGON ((-73.8670614947212 40.58208797679338...
    3    (POLYGON ((-74.01092841268033 40.6844914725429...
    4    (POLYGON ((-73.89680883223775 40.7958084451597...
    Name: geometry, dtype: object



That geometry is stored with reference to some kind of `**coordinate
reference
system** <https://en.wikipedia.org/wiki/Spatial_reference_system>`__, or
CRS. You can extract what CRS your polygons are stored in using the
``crs`` attribute:

.. code:: python

    boroughs.crs




.. parsed-literal::

    {'init': 'epsg:4326'}



In this case ``epsg:4326`` is an identifier for what the rest of us more
commonly refer to as "longitude and latitude". EPSG itself is a
standardized system for refering to coordinate reference systems;
`spatialreference.org <http://spatialreference.org/ref/epsg/wgs-84/>`__
is the best place to look these identifiers up.

Coordinate reference systems are, basically, different ways of
mathematically calculating locations. Due to the complexity of the
surface of the earth, different geographically sensitive systems of
measurement are more or less useful for different tasks. For example,
the United States Geolocial Service, which provides extremely
high-accuracy maps of United States localities, maintains individual
coordinate reference systems, refered to as "state plane systems", for
the various states of the union. These are used throughout government,
and look nothing like the latitude and longitude coordinates that we are
generally more used to.

For example, New York City approximately twice per year releases an
updated version of MapPLUTO, a geospatial dataset which provides
building footprint polygons for all buildings in New York City. This is
the dataset which powers some pretty amazing visualizations, like
`Bklynr's Brooklyn building age
map <http://bklynr.com/block-by-block-brooklyns-past-and-present/>`__.

.. code:: python

    manhattan_buildings = gpd.read_file('../../data/manhattan_mappluto/MN_Dcp_Mappinglot.shp')
    manhattan_buildings.head()




.. raw:: html

    <div>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>BLOCK</th>
          <th>BORO</th>
          <th>CREATED_BY</th>
          <th>...</th>
          <th>Shape_Area</th>
          <th>Shape_Leng</th>
          <th>geometry</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>20009</td>
          <td>1</td>
          <td>None</td>
          <td>...</td>
          <td>10289.237892</td>
          <td>836.495687</td>
          <td>POLYGON ((986519.6798000038 200244.1201999933,...</td>
        </tr>
        <tr>
          <th>1</th>
          <td>20031</td>
          <td>1</td>
          <td>None</td>
          <td>...</td>
          <td>8943.539985</td>
          <td>478.609196</td>
          <td>POLYGON ((992017.6599999964 216103.8700000048,...</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20027</td>
          <td>1</td>
          <td>None</td>
          <td>...</td>
          <td>10156.610383</td>
          <td>486.181920</td>
          <td>POLYGON ((991564.0900000036 215278.3798999935,...</td>
        </tr>
        <tr>
          <th>3</th>
          <td>20012</td>
          <td>1</td>
          <td>None</td>
          <td>...</td>
          <td>7657.969093</td>
          <td>357.345276</td>
          <td>POLYGON ((986364.6000999957 201496.4998999983,...</td>
        </tr>
        <tr>
          <th>4</th>
          <td>20067</td>
          <td>1</td>
          <td>None</td>
          <td>...</td>
          <td>9171.078777</td>
          <td>479.281556</td>
          <td>POLYGON ((995870.7099999934 223069.0699999928,...</td>
        </tr>
      </tbody>
    </table>
    <p>5 rows × 10 columns</p>
    </div>



But, unlike our easy coordinates above, this data is stored in the Long
Island State Plane coordinate reference system:

.. code:: python

    manhattan_buildings.geometry.head()




.. parsed-literal::

    0    POLYGON ((986519.6798000038 200244.1201999933,...
    1    POLYGON ((992017.6599999964 216103.8700000048,...
    2    POLYGON ((991564.0900000036 215278.3798999935,...
    3    POLYGON ((986364.6000999957 201496.4998999983,...
    4    POLYGON ((995870.7099999934 223069.0699999928,...
    Name: geometry, dtype: object



The file we just read in provided embedded information about its
coordinate reference system, which ``geopandas`` stores as a
```proj4`` <https://github.com/OSGeo/proj.4>`__ string:

.. code:: python

    manhattan_buildings.crs




.. parsed-literal::

    {'datum': 'NAD83',
     'lat_0': 40.16666666666666,
     'lat_1': 40.66666666666666,
     'lat_2': 41.03333333333333,
     'lon_0': -74,
     'no_defs': True,
     'proj': 'lcc',
     'units': 'us-ft',
     'x_0': 300000,
     'y_0': 0}



``geoplot`` expects its input to be in terms of latitude and longitude.
This is required because it's so easy to do: to convert your data from
one CRS to another, you can just use the ``geopandas`` ``to_crs``
method:

.. code:: python

    manhattan_buildings = manhattan_buildings.to_crs(epsg=4326)

Now all of our building footprints are in ordinary coordinates!

.. code:: python

    manhattan_buildings.geometry.head()




.. parsed-literal::

    0    POLYGON ((-73.99181250685882 40.71630025841903...
    1    POLYGON ((-73.97196114404649 40.75982822136702...
    2    POLYGON ((-73.97359928976277 40.75756284914222...
    3    POLYGON ((-73.99237153770106 40.71973777834428...
    4    POLYGON ((-73.95804078098135 40.77894165663843...
    Name: geometry, dtype: object



You should also know, at a minimum, that all of these geometries are
always ```shapely`` <http://toblerity.org/shapely/manual.html>`__
objects:

.. code:: python

    type(manhattan_buildings.geometry.iloc[0])




.. parsed-literal::

    shapely.geometry.polygon.Polygon



.. code:: python

    type(boroughs.geometry.iloc[0])




.. parsed-literal::

    shapely.geometry.multipolygon.MultiPolygon



``shapely`` provides a large API surface for any geometric
transformation or operations that you can think of, and ``geopandas``
wraps many of these even further, creating a convenient way of getting
"classical" GIS operations done on your data. Like ``geopandas``,
``shapely`` is very well-documented, so to dive into these further `read
the documentation <http://toblerity.org/shapely/manual.html>`__.

In this tutorial, we'll focus on one particular aspect of ``shapely``
which is likely to come up: defining your own geometries. A decision I
made early on in the design stages of ``geoplot`` was mandating input as
a ``GeoDataFrame``, as doing so (as opposed to, say, also supporting
``DataFrame`` input) greatly simplifies both internal and external
library design.

However, in the cases above we read a GeoDataFrame straight out of
geospatial files: our borough information was stored in the
`GeoJSON <http://geojson.org/>`__ format, while our building footprints
were a `Shapefile <https://en.wikipedia.org/wiki/Shapefile>`__. What if
we have geospatial data embedded in an ordinary ``CSV`` or ``JSON``
file, which read into an ordinary ``pandas`` ``DataFrame``?

.. code:: python

    collisions = pd.read_csv("../../data/nyc_collisions/NYPD_Motor_Vehicle_Collisions.csv", index_col=0).sample(5000)
    collisions = collisions[collisions['LOCATION'].notnull()]
    collisions.head()




.. raw:: html

    <div>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>TIME</th>
          <th>BOROUGH</th>
          <th>ZIP CODE</th>
          <th>...</th>
          <th>VEHICLE TYPE CODE 3</th>
          <th>VEHICLE TYPE CODE 4</th>
          <th>VEHICLE TYPE CODE 5</th>
        </tr>
        <tr>
          <th>DATE</th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>12/16/2014</th>
          <td>17:00</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>10/21/2015</th>
          <td>19:45</td>
          <td>QUEENS</td>
          <td>11691.0</td>
          <td>...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>08/12/2015</th>
          <td>8:50</td>
          <td>QUEENS</td>
          <td>11103.0</td>
          <td>...</td>
          <td>UNKNOWN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>08/04/2012</th>
          <td>4:40</td>
          <td>QUEENS</td>
          <td>11102.0</td>
          <td>...</td>
          <td>PASSENGER VEHICLE</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>07/15/2016</th>
          <td>10:50</td>
          <td>BRONX</td>
          <td>10456.0</td>
          <td>...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
      </tbody>
    </table>
    <p>5 rows × 28 columns</p>
    </div>



.. code:: python

    collisions[['LATITUDE', 'LONGITUDE']].head()




.. raw:: html

    <div>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>LATITUDE</th>
          <th>LONGITUDE</th>
        </tr>
        <tr>
          <th>DATE</th>
          <th></th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>12/16/2014</th>
          <td>40.677672</td>
          <td>-73.803327</td>
        </tr>
        <tr>
          <th>10/21/2015</th>
          <td>40.602834</td>
          <td>-73.765749</td>
        </tr>
        <tr>
          <th>08/12/2015</th>
          <td>40.764354</td>
          <td>-73.911304</td>
        </tr>
        <tr>
          <th>08/04/2012</th>
          <td>40.775731</td>
          <td>-73.926023</td>
        </tr>
        <tr>
          <th>07/15/2016</th>
          <td>40.835011</td>
          <td>-73.903520</td>
        </tr>
      </tbody>
    </table>
    </div>



In that case we can import ``shapely`` directly, use it to define our
own geometries, using the data from our ``DataFrame``, and use that to
initialize a ``GeoDataFrame``.

.. code:: python

    import shapely
    
    collision_points = collisions.apply(lambda srs: shapely.geometry.Point(srs['LONGITUDE'], srs['LATITUDE']),
                                        axis='columns')
    collision_points.head()




.. parsed-literal::

    DATE
    12/16/2014           POINT (-73.8033269 40.6776723)
    10/21/2015    POINT (-73.76574859999999 40.6028338)
    08/12/2015    POINT (-73.9113038 40.76435410000001)
    08/04/2012    POINT (-73.92602340000001 40.7757305)
    07/15/2016           POINT (-73.9035195 40.8350109)
    dtype: object



From there we pass this iterable of geometries to the ``geometry``
property of the ``GeoDataFrame`` initializer, and we're done!

.. code:: python

    collisions_geocoded = gpd.GeoDataFrame(collisions, geometry=collision_points)
    collisions_geocoded.head(5)




.. raw:: html

    <div>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>TIME</th>
          <th>BOROUGH</th>
          <th>ZIP CODE</th>
          <th>...</th>
          <th>VEHICLE TYPE CODE 4</th>
          <th>VEHICLE TYPE CODE 5</th>
          <th>geometry</th>
        </tr>
        <tr>
          <th>DATE</th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>12/16/2014</th>
          <td>17:00</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>POINT (-73.8033269 40.6776723)</td>
        </tr>
        <tr>
          <th>10/21/2015</th>
          <td>19:45</td>
          <td>QUEENS</td>
          <td>11691.0</td>
          <td>...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>POINT (-73.76574859999999 40.6028338)</td>
        </tr>
        <tr>
          <th>08/12/2015</th>
          <td>8:50</td>
          <td>QUEENS</td>
          <td>11103.0</td>
          <td>...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>POINT (-73.9113038 40.76435410000001)</td>
        </tr>
        <tr>
          <th>08/04/2012</th>
          <td>4:40</td>
          <td>QUEENS</td>
          <td>11102.0</td>
          <td>...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>POINT (-73.92602340000001 40.7757305)</td>
        </tr>
        <tr>
          <th>07/15/2016</th>
          <td>10:50</td>
          <td>BRONX</td>
          <td>10456.0</td>
          <td>...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>POINT (-73.9035195 40.8350109)</td>
        </tr>
      </tbody>
    </table>
    <p>5 rows × 29 columns</p>
    </div>



.. code:: python

    type(collisions_geocoded)




.. parsed-literal::

    geopandas.geodataframe.GeoDataFrame



In most cases, data with geospatial information provided in a CSV will
be point data corresponding with individual coordinates. Sometimes,
however, one may wish to define more complex geometry: square areas, for
example, and *maybe* even complex polygons. While we won't cover these
cases, they're quite similar to the extremely simple point case we've
shown here. For further reference on such a task, refer to the
``shapely`` documentation.

`Click here to continue to the next section of the tutorial:
"Projections" <./projections.html>`__.
