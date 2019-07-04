geoplot: geospatial data visualization
======================================

.. raw:: html

    <div class="row">
    <a href=./examples/nyc-collision-factors.html>
    <img src="https://raw.githubusercontent.com/ResidentMario/geoplot/master/figures/nyc-collision-factors.png" height="200" width="200">
    </a>

    <a href=./examples/los-angeles-flights.html>
    <img src="https://raw.githubusercontent.com/ResidentMario/geoplot/master/figures/los-angeles-flights.png" height="200" width="200">
    </a>

    <a href=./examples/usa-city-elevations.html>
    <img src="https://raw.githubusercontent.com/ResidentMario/geoplot/master/figures/usa-city-elevations.png"
     height="200">
    </a>

    <a href=./examples/nyc-parking-tickets.html>
    <img src="https://raw.githubusercontent.com/ResidentMario/geoplot/master/figures/nyc-parking-tickets.png" height="200" width="200">
    </a>

    <a href=./examples/dc-street-network.html>
    <img src="https://raw.githubusercontent.com/ResidentMario/geoplot/master/figures/dc-street-network.png" height="200" width="200">
    </a>

    </div>


``geoplot`` is a high-level Python geospatial plotting library. It's an extension to ``cartopy`` and ``matplotlib``
which makes mapping easy: like ``seaborn`` for geospatial. It comes with the following features:

* **High-level plotting API**: ``geoplot`` is cartographic plotting for the 90% of use cases. All of the standard-bearermaps that you’ve probably seen in your geography textbook are easily accessible.

* **Native projection support**: The most fundamental peculiarity of geospatial plotting is projection: how do you unroll a sphere onto a flat surface (a map) in an accurate way? The answer depends on what you’re trying to depict. ``geoplot`` provides these options.

* **Compatibility with matplotlib**: While ``matplotlib`` is not a good fit for working with geospatial data directly, it’s a format that’s well-incorporated by other tools.

For a brief introduction refer to the `Quickstart`_.

.. _Quickstart: https://nbviewer.jupyter.org/github/ResidentMario/geoplot/blob/master/notebooks/tutorials/Quickstart.ipynb


.. toctree::
    :maxdepth: 1
    :caption: Getting Started

    installation.rst
    quickstart/quickstart.rst

.. toctree::
    :maxdepth: 1
    :caption: User Guide

    user_guide/Working_with_Geospatial_Data.rst
    user_guide/Working_with_Projections.rst
    user_guide/Customizing_Plots.rst
    plot_references/plot_reference.rst

.. toctree::
    :maxdepth: 1
    :caption: Gallery

    Gallery <gallery/index.rst>

.. toctree::
    :maxdepth: 1
    :caption: API Reference

    api_reference.rst
