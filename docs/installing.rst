.. _installing:

Installation
============

`conda <http://conda.pydata.org/docs/>`_ is a hard requirement for installing ``geoplot``
on all platforms - ``pip`` alone can't do it. Installation is currently more difficult than it could be; this should
change in the future.


Linux
-----

If you haven't already, `install conda <http://conda.pydata.org/docs/>`_. Then run the following block of code in the
console:

.. code:: bash

    # Add the conda-forge channel.
    conda config --add channels conda-forge
    # Initialize and enter an environment.
    conda create --name YOUR_ENVIRONMENT_NAME seaborn cartopy geopandas python=3.5
    source activate YOUR_ENVIRONMENT_NAME
    # Install geoplot.
    pip install geoplot
    # (Optionally) drop the conda-forge channel
    conda config --remove channels conda-forge

Mac OSX
-------

If you haven't already, `install conda <http://conda.pydata.org/docs/>`_. Then run the following block of code in the
console:

.. code:: bash

    # Initialize and enter an environment.
    conda create --name YOUR_ENVIRONMENT_NAME seaborn python=3.5
    source activate YOUR_ENVIRONMENT_NAME
    # Add the conda-forge channel, install cartopy, (optionally) drop the channel.
    conda config --add channels conda-forge
    conda install cartopy
    conda config --remove channels conda-forge
    # Install geopandas and (finally) geoplot via pip, *not* conda!
    pip install geopandas
    pip install geoplot


Windows
-------

I have not yet succeeded in finding a way of installing ``geoplot`` on Windows at all. Sorry.