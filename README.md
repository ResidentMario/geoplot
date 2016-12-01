### Quickstart

`geoplot` is to geospatial analytics as `seaborn` is to regular data science. `geoplot` is to `cartopy` as `seaborn`
is to `matplotlib`.

This project is a very early "pre-alpha" work in progress.

![a](./figures/example.png)

![a](./figures/example_2.png)

### Potential features

* `geoplot()` &mdash; Simple extension of the `geopandas`-provided `plot()`.
* `dot()` &mdash; [Dot map](https://raw.githubusercontent.com/andrea-cuttone/geoplotlib/master/examples/screenshots/dotdensity.png).
A simple map type that places a dot wherever a coordinate occurs.
* `marker()` &mdash; An elaboration on `dot()`. Allows any marker type, as well as MultiMarkers.
* `heatmap()` &mdash; [Geo heatmap](https://raw.githubusercontent.com/andrea-cuttone/geoplotlib/master/examples/screenshots/kde1.png).
Fit a KDE or a distance-regular colormap to the map.
* `boundary()` &mdash; [Boundary map](https://raw.githubusercontent.com/andrea-cuttone/geoplotlib/master/examples/screenshots/shapefiles.png).
Draws the boundaries of polygons. Mostly useful in combination.
* `choropleth()` &mdash; [Choropleth](https://raw.githubusercontent.com/andrea-cuttone/geoplotlib/master/examples/screenshots/choropleth.png).
* `voronoi()` &mdash; [Voronoi tesselation](https://raw.githubusercontent.com/andrea-cuttone/geoplotlib/master/examples/screenshots/voronoi-filled.png).
Fits polygons around points that map by minimizing distance.
* `spatial()` &mdash; [Spatial map](https://raw.githubusercontent.com/andrea-cuttone/geoplotlib/master/examples/screenshots/graph-flights.png).
Colors polygons according to data on their number.
* `symbol()` &mdash; [Symbol map](symbol.png). Embed symbols like pies or growing circles into a map.
* `cartogram()` &mdash; Proportional sizing, no idea how that would be implemented.
    * Including [non-contiguous option](http://bl.ocks.org/mbostock/4055908)? Easier to implement.
* Enough to start. Plenty more [here](https://github.com/andrea-cuttone/geoplotlib/tree/master/examples/screenshots).

#### References

* http://geoffboeing.com/2016/11/osmnx-python-street-networks/
* http://seaborn.pydata.org/api.html
* http://scitools.org.uk/cartopy/docs/latest/gallery.html
* http://darribas.org/gds_scipy16/ipynb_md/02_geovisualization.html

### To-do

Tons.

### Development environment

This is not accurate anymore.

To set this library up for development in a local virtual environment:

#### Linux, Mac OSX

1. `git clone https://github.com/ResidentMario/geoplot.git`
2. `conda env create`
3. `source activate geoplot`

#### Windows

1. `git clone https://github.com/ResidentMario/geoplot.git`
2. `conda env create` (this will likely partially fail, keep going)
3. `runipy install.ipynb`
4. `activate geoplot`