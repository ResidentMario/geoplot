
Geoplot tutorial
================

Introduction
------------

``geoplot`` is a tool which fills a gap which I have perceived, in my
own work, as a serious shortcoming in the geospatial Python ecosystem:
the need for a high-level geospatial plotting library.

Geospatial data is any form of data which has a location to it. Most of
the data generated today has such a geospatial context, context which
is, in turn, oftentimes important to understanding the data itself.
Thanks to its ease-of-use and deep ecosystem, the Python programming
language has emerged as a leading choice in the performance of data
analytics, geospatial data analytics included.

Though there are increasingly many options, the venerable ``matplotlib``
remains the core Python plotting tool. Nevertheless, we oftentimes don't
interact with ``matplotlib`` directly; we instead use extensions built
over it, like the ``pandas`` plotting facilities or ``seaborn``, to get
our work done. Knowing how ``matplotlib`` operates is certainly helpful,
but it's neither necessary to get started nor, usually, the fastest way
to get something done.

``geoplot`` aims to be ``seaborn`` for geospatial data. Hence it comes
with the following built-in features:

-  **High-level plotting API**: ``geoplot`` is cartographic plotting for
   the 90% of use cases. All of the standard-bearer maps that you've
   probably seen in your geography textbook are easily accessible, as
   are many more novel options.
-  **Native projection support**: The most fundamental peculiarity of
   geospatial plotting is projection: how do you unroll a sphere onto a
   flat surface (a map) in an accurate way? The answer depends on what
   you're trying to depict. ``geoplot`` provides these options.
-  **Compatibility with matplotlib**: While ``matplotlib`` is not a good
   fit for working with geospatial data directly, it's a format that's
   well-incorporated by other geospatial tools (``mplleaflet`` in
   particular). For compatibility, ``geoplot`` provides an option for
   emiting pure ``matplotlib`` plots.
-  **Built with modern geospatial Python in mind**: Geospatial data is a
   fast-moving target in the Python ecosystem. Innovations in recent
   years have made working with such data much easier than it was when
   e.g. matplotlib's lower-level ``basemap`` tool was being developed,
   which ``geoplot`` leverages with an easy-to-use and widely compatible
   API.

``geoplot`` does this by leveraging two pre-existing libraries in
particular: ``geopandas``, an extension to the mainstay ``pandas``
library with embedded geospatial data support, used for input; and
``cartopy``, a (``matplotlib``-based) low-level plotting library, used
for output.

The rest of this text is a tutorial on ``geoplot`` usage. It is written
for the perspective of someone who has some existing familiarity with
the existing data science ecosystem, but hasn't used the geospatial
tools just yet. If you haven't worked with data before in Python and are
thus unfamiliar with ``pandas``, you should refer to its documentation
first - `10 Minutes to
Pandas <http://pandas.pydata.org/pandas-docs/stable/10min.html>`__ is
one place where you can get started.

`Click here to continue to the next section of the tutorial: "Working
with Geospatial Data" <./data.html>`__.
