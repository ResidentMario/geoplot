## Cloning

To work on `geoplot` locally, you will need to clone it.

```git
git clone https://github.com/ResidentMario/geoplot.git
```

You can then set up your own branch version of the code, and
 work on your changes for a pull request from there.

```bash
cd geoplot
git checkout -B new-branch-name
```



## Environment

`geoplot` depends on one set of requirements for installation and usage, and another (augmented) set of requirements 
for development. Both require [`conda`](https://conda.io/) to work. The easiest way to get a development environment 
set up is to run `conda install --file devenv.yml` out of the `envs` directory.

This should install all necessary dependencies. If it fails, you will need to localize a few yourself. Use the 
packages listed in `devenv.yml` as a guide.

Once you have the environment localized and activated, run `pip install -e .` out of the `geoplot` repository root 
folder, and all should be well.

## Testing

`geoplot` tests are located in the `tests` folder. Any PRs you submit should eventually pass all of the tests located
 in this folder, or fail them for very good reasons.

Tests are in a somewhat inconsistent state right now, as they are going through refactoring.

## Documentation

Documentation is provided via `sphinx`, plus a handful of plug-ins.

In general, to regenerate the documentation from the current source, navigate to the `docs` folder and run `make html`.

### Reference images
The API reference images are generated from the thusly named Jupyter notebook in the `notebooks` folder. You can 
regenerate these images by running all or part of this notebook. That means that editing these pages is a two-step 
process: edit and save the notebooks, then run `make html` to finally generate the HTML.

### Gallery images
The Gallery images are generated from Python scripts in the `docs/examples` subfolder. All of the examples are 
self-contained and you should be able to regenerate them by running them at the command line.

However, notably, two of the Gallery examples are webmaps. The code for these is hosted inside of separate Gists 
instead, and click-throughs are directed to [blocks](http://bl.ocks.org/) visualizing the inline JavaScript result.

### Tutorials

The tutorials are written as notebooks in the top-level `notebooks` repository. The documentation does not actually 
host these directly, linking click-throughs to an [NBViewer](http://nbviewer.jupyter.org/) projection instead. Hence, 
to update the tutorials you only need up to edit, save, and push to the repository!

### Example data

The tutorial, gallery, and reference images all make use of real-world example data that is packaged as an accessory to 
this library. This data had a very messy history during the early stages of the project, but was eventually cleaned 
up and refactored into a set of mostly `geojson` files. This now lives in the complimentary 
[`geoplot-data`](https://github.com/ResidentMario/geoplot-data) repository.

The data is zipped up and packaged online using [Quilt](https://quiltdata.com/package/ResidentMario/geoplot_data), a
data-as-a-package management service. You can view the package [on the web](https://quiltdata.com/package/ResidentMario/geoplot_data).
Data is imported from there throughout the `geoplot` documentation; the shebang looks something like:

```python
from quilt.data.ResidentMario import geoplot_data
import geopandas as gpd
boston_zip_codes = gpd.read_file(geoplot_data.boston_zip_codes())
```

The example data should need updating only very rarely, and then only to add new data, or *maybe* remove old data. 
But when it is necessary: submit a pull request to the `geoplot-data` repository. The data package owner 
(@ResidentMario) can then update the example data to go along with it (using the instructions [here](https://docs.quiltdata.com/)).

### Everything else
The remaining pages are all written as `rst` files accessible from the top level of the `docs` folder. For these, 
just `make html` will suffice.

### Serving 

The documentation is served at [residentmario.github.io](https://residentmario.github.io/geoplot/index.html) via 
GitHub's site export feature, served out of the `gh-pages` branch. The process for putting up a new version of the 
docs (once you have built them locally via `make html` and committed to the repository) basically follows the 
procedure outlined [here](http://www.willmcginnis.com/2016/02/29/automating-documentation-workflow-with-sphinx-and-github-pages/),
with a couple of small changes in the details: `git rm -rf *`, not `git rm -rf .`, and `_build`, not `build`.