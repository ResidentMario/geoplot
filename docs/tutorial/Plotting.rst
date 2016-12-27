
Plotting
========

``geoplot`` is cartographic plotting for the 90% of use cases. What does
that mean in practice?

The dominant dataset format in the current Python ecosystem is the
``CSV`` (or something ``CSV``-like, like an ``HDF5`` store), and the
dominant geospatial datatype within such datasets is the simple
``(X, Y)`` coordinate. This is because at its deepest level, the data
that we create and consume tends to be point data: a police response at
an identifiable intersection; an ad click from a geolocated IP address;
a 311 complaint recorded from such-and-such address. Even when data
isn't *really* pointwise, oftentimes it can summarized succintly by
pretending that it is: take hurriances, for example, which the United
States NOAA releases (in its simplest form) as coordinate center
observations taken at one-hour intervals.

The few times when it's not one coordinate, it tends to be *two*
coordinates (``(X_1, Y_1)``; ``(X_2, Y_2)``) in a network of some kind.
An analysis of an urban bikeshare program, for example, might include
network data on station-to-station route popularity, for example.

Polygons are the other popular geospatial scheme. NYPD reports on New
York City crime trends, for example, are aggregated at the police
precinct level, and data collected in terms of points can easily be
aggregated into geographic areas (doubly so if a categorical variable
like "Census Tract" or "State" is in the fold).

That is not to say that all data comes in the form of points or
polygons, of course. There are all sorts of data types not adequetly
described by either, things like MRI scans or astronomical data or road
network utilization. "Points and polygons" are merely the simplest and
easiest-to-use geospatial formats; hence, the ones that the modern "data
scientist", in the troperific sense of the word, tends to run into; and
hence, the focus of this library.

And with that pontification out the way, the rest of this page will
explore the finer points of practical ``geoplot``. We'll introduce the
plot types as we go along, not dithering too long on anything in
particular: to get the full picture, read the API documentation! Along
the way we'll examine useful fundamentals that apply to the library as a
whole.

To be continued...
------------------

This tutorial is still a work in progress. For now I suggest reading the
documentation in the API Reference.
