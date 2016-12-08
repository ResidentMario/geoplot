.. _tutorial:

Geoplot tutorial
================

Geospatial data today
---------------------

Geospatial data is any form of data which has a location to it. Most of
the data generated today has such a geospatial context, context which
is, in turn, oftentimes important to understanding the data itself.
Thanks to its ease-of-use and deep ecosystem, the Python programming
language has emerged as a leading choice in the performance of data
analytics, geospatial data analytics included. ``geoplot``, a tool for
generating easy-to-use geospatial plot types, builds on the fundamental
peices of this existing programming stack, which we will discuss first.

The core abstraction is the ``GeoDataFrame``, an augmented version of a
``pandas`` ``DataFrame`` with an attached geometry:

.. code:: python

    import geopandas as gpd
    boroughs = gpd.read_file("./data/boroughs.geojson", driver='GeoJSON')
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



Readers familiar with ``pandas`` operations should be pleased to know
that the library providing this geometric abstraction, ``geopandas``, is
just an extension of the former, and so has all the same fundamental
operations. That means that if you've ever worked with data before but
haven't yet touched geospatial, the transition should be completely
straightforward. If you haven't worked with data before in Python and
are thus unfamiliar with ``pandas``, you should refer to its
documentation and get good with using that library first—\ `10 Minutes
to Pandas <http://pandas.pydata.org/pandas-docs/stable/10min.html>`__ is
one place where you can get started.

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
