.. geoplot documentation master file, created by
   sphinx-quickstart on Mon Dec  5 22:22:01 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Geoplot: geospatial data visualization
======================================

.. raw:: html

    <div class="row">
    <a href=./examples/nyc-collision-factors.html>
    <img src="https://raw.githubusercontent.com/ResidentMario/geoplot/master/figures/demos/nyc-collision-factors
    -thumb.png" height="200" width="200">
    </a>

    <a href=./examples/los-angeles-flights.html>
    <img src="https://raw.githubusercontent.com/ResidentMario/geoplot/master/figures/demos/los-angeles-flights-thumb
    .png" height="200" width="200">
    </a>

    <!--
    <a href=.>
    <img src="https://raw.githubusercontent.com/ResidentMario/geoplot/master/figures/demos/aggplot-collisions-thumb
    .png" height="200" width="200">
    </a>
    -->

    <a href=./examples/usa-city-elevations.html>
    <img src="https://raw.githubusercontent.com/ResidentMario/geoplot/master/figures/demos/usa-city-elevations-thumb.png"
     height="200">
    </a>

    <a href=./examples/nyc-parking-tickets.html>
    <img src="https://raw.githubusercontent.com/ResidentMario/geoplot/master/figures/demos/nyc-parking-tickets-thumb
    .png" height="200" width="200">
    </a>

    <a href=./examples/dc-street-network.html>
    <img src="https://raw.githubusercontent.com/ResidentMario/geoplot/master/figures/demos/dc-street-network-thumb
    .png" height="200" width="200">
    </a>

    </div>

.. raw:: html

    <div style="float:left; width:66%;">

    <span style="font-weight:bold; font-size:28px; margin-bottom:1em">About</span>

``geoplot`` is a high-level Python geospatial plotting library. It's an extension to ``cartopy`` and ``matplotlib``
which makes mapping easy: like ``seaborn`` for geospatial. It comes with the following features:

* **High-level plotting API**: ``geoplot`` is cartographic plotting for the 90% of use cases. All of the standard-bearermaps that you’ve probably seen in your geography textbook are easily accessible.

* **Native projection support**: The most fundamental peculiarity of geospatial plotting is projection: how do you unroll a sphere onto a flat surface (a map) in an accurate way? The answer depends on what you’re trying to depict. ``geoplot`` provides these options.

* **Compatibility with matplotlib**: While ``matplotlib`` is not a good fit for working with geospatial data directly, it’s a format that’s well-incorporated by other tools.

* **Built with modern geospatial Python in mind**: Innovations in recent years have made working with geospatial data easier than ever, which geoplot leverages with an easy-to-use and widely compatible API.

Browse the `Gallery <./gallery.html>`_ to see what ``geoplot`` can do, then jump into the `Tutorial <./tutorial.html>`_
or the `API Reference <./api.html>`_ to see how to do it. For help installing, check out
`Installation <./installing.html>`_.

.. raw:: html

    </div>
    <div style="float:right; width:33%;">

    <span style="font-weight:bold; font-size:28px; margin-bottom:1em">Documentation</span>

.. toctree::
   :maxdepth: 1

   installing
   gallery
   api
   tutorial

.. raw:: html
    </div>

..  Indices and tables
    ==================
    * :ref:`genindex`
    * :ref:`modindex`
    * :ref:`search`
