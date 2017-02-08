.. _installing:

Installation
============

Linux and Mac OSX
-----------------

With Conda
++++++++++

If you haven't already, `install conda <http://conda.pydata.org/docs/>`_.  You'll also need to have the
``conda-forge`` channel enabled: if you don't already, you can do so with ``conda config --add channels conda-forge``.

Then run ``conda install geoplot`` and you're done.

Without Conda
+++++++++++++

Things are much trickier if you don't have access to ``conda``, for whatever reason. You will need
``matplotlib``, ``seaborn``, the ``proj4`` (`docs <http://proj4.org/>`_) and ``GEOS`` (`docs <https://trac.osgeo
.org/geos/>`_) external libraries, and ``cartopy`` (which relies on ``proj4`` and ``GEOS``). Here's an approximate
code path:

.. code-block:: bash

    # Use of a virtualenv is strongly encouraged.
    mkdir my_project
    cd my_project
    virtualenv venv
    # Easy installs that shouldn't cause any problems.
    pip install matplotlib seaborn
    # ... install proj4 and GEOS using brew, apt-get, etc. ...
    # Install shapely with the no-binary option.
    # See https://github.com/Toblerity/Shapely/issues/435
    pip install --no-binary shapely
    # Install cartopy.
    pip install cartopy
    # Install geoplot.
    pip install geoplot

It's difficult to succeed installing this way, due to conflicts deep within the stack with reference to the C
dependencies. `This <https://github.com/SciTools/cartopy/issues/805>`_,
`this <https://github.com/Toblerity/Shapely/issues/435>`_, and `this <https://github.com/SciTools/cartopy/issues/823>`_
are reference issues.

Windows
-------

Unforunately ``geoplot`` is not available on Windows yet due to `an unresolved dependency issue <https://github.com/SciTools/cartopy/issues/805>`_.
Sorry.

