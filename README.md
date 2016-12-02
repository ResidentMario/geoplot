### Quickstart

`geoplot` is an in-progress, very early-iteration library for high-level geographic data visualization, akin to
`seaborn`. It is based on `cartopy`, `geopandas`, and the rest of the modern geospatial Python stack.

Here are a few teasers of what it can do:

![a](./figures/example.png)

![a](./figures/example_2.png)

### Status

Frozen due to irresolvable differences between `shapely`, `fiona`, and `cartopy` `GEOS` C library linkages&mdash;both
 of my development rigs are suffering from dependency hell kernel deaths of different stripes. [See this ticket for
 reference](https://github.com/Toblerity/Shapely/issues/435).

Not sure where to go from here.

#### References

* http://geoffboeing.com/2016/11/osmnx-python-street-networks/
* http://seaborn.pydata.org/api.html
* http://scitools.org.uk/cartopy/docs/latest/gallery.html
* http://darribas.org/gds_scipy16/ipynb_md/02_geovisualization.html

### To-do

Tons.
