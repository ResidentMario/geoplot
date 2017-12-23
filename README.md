## geoplot: geospatial data visualization  ![t](https://img.shields.io/badge/status-stable-green.svg) ![https://anaconda.org/conda-forge/geoplot](https://img.shields.io/conda/v/conda-forge/geoplot.svg)

<!-- [![PyPi version](https://img.shields.io/pypi/v/missingno.svg)]
(https://pypi.python.org/pypi/geoplot/)-->

<div class="row">
<a href=https://residentmario.github.io/geoplot/examples/nyc-collision-factors.html>
<img src="https://raw.githubusercontent.com/ResidentMario/geoplot/master/figures/nyc-collision-factors.png"
 height="150" width="150">
</a>

<a href=https://residentmario.github.io/geoplot/examples/los-angeles-flights.html>
<img src="https://raw.githubusercontent.com/ResidentMario/geoplot/master/figures/los-angeles-flights.png"
 height="150" width="150">
</a>

<!--
<a href=.>
<img src="https://raw.githubusercontent.com/ResidentMario/geoplot/master/figures/aggplot-collisions.png"
height="150" width="150">
</a>
-->

<a href=https://residentmario.github.io/geoplot/examples/usa-city-elevations.html>
<img src="https://raw.githubusercontent.com/ResidentMario/geoplot/master/figures/usa-city-elevations.png"
 height="150">
</a>

<a href=https://residentmario.github.io/geoplot/examples/nyc-parking-tickets.html>
<img src="https://raw.githubusercontent.com/ResidentMario/geoplot/master/figures/nyc-parking-tickets.png"
 height="150" width="150">
</a>

<a href=https://residentmario.github.io/geoplot/examples/dc-street-network.html>
<img src="https://raw.githubusercontent.com/ResidentMario/geoplot/master/figures/dc-street-network.png"
height="150" width="150">
</a>

</div>


``geoplot`` is a high-level Python geospatial plotting library. It's an extension to `cartopy` and `matplotlib` 
which makes mapping easy: like `seaborn` for geospatial. It comes with the following features:

* **High-level plotting API**: geoplot is cartographic plotting for the 90% of use cases. All of the standard-bearer
maps that you’ve probably seen in your geography textbook are easily accessible.
* **Native projection support**: The most fundamental peculiarity of geospatial plotting is projection: how do you
unroll a sphere onto a flat surface (a map) in an accurate way? The answer depends on what you’re trying to depict.
`geoplot` provides these options.
* **Compatibility with `matplotlib`**: While `matplotlib` is not a good fit for working with geospatial data directly,
it’s a format that’s well-incorporated by other tools.
* **Built with modern geospatial Python in mind**: Innovations in recent years have made working with geospatial data
 easier than ever, which geoplot leverages with an easy-to-use and widely compatible API.

Installation is simple with `conda install geoplot -c conda-forge`. [See the documentation for help getting started](https://residentmario.github.io/geoplot/index.html).
