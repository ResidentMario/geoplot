.. _installing:

Installation
============

Installation is currently more difficult than it should be. This should change in the future.

Linux
-----

``pip install geoplot`` should work out of the box.

Mac OSX
-------

Due to `this bug <https://groups.google.com/a/continuum.io/forum/#!msg/conda/kw2xC4wjI-Y/wDHMYeTZDeEJ)>`_ (see
further `here <https://github.com/Toblerity/Shapely/issues/177>`_,
`here <https://github.com/Toblerity/Shapely/issues/258>`_, `here <https://github.com/SciTools/cartopy/issues/481>`_,
`here <https://github.com/SciTools/cartopy/issues/738>`_), it is difficult to build both ``shapely`` and ``cartopy``
(two ``geoplot`` dependencies) working simultaneously, due to the way they each look for linked ``geos`` C libraries. A
`patch <https://github.com/conda-forge/shapely-feedstock/blob/master/recipe/geos_c.patch>`_ applied in ``conda``
addresses this issue, making `conda <http://conda.pydata.org/docs/>`_ a hard requirement for installing ``geoplot``
on Mac OSX - ``pip`` alone can't do it.

If you haven't already, `install conda <http://conda.pydata.org/docs/>`_. Then run the following block of code in the
console:

.. code:: bash

    # Initialize and enter an environment.
    conda create --name YOUR_ENVIRONMENT_NAME pandas python=3.5
    source activate YOUR_ENVIRONMENT_NAME
    # Add the conda-forge channel, install cartopy, (optionally) drop the channel.
    conda config --add channels conda-forge
    conda install cartopy
    conda config --remove channels conda-forge
    # Install geopandas via pip, *not* conda!
    pip install geopandas


Windows
-------

I have not yet succeeded in finding a way of installing ``geoplot`` on Windows at all. Sorry.