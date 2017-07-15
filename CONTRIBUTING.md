## Cloning

To work on `geoplot` locally, you will need to clone it. You can then set up your own branch version of the code, and
 work on your changes for a pull request from there.

```git
git clone https://github.com/ResidentMario/geoplot.git
cd geoplot
git branch my-branch-name
git checkout my-branch-name
```

## Environment

`geoplot` depends on one set of requirements for installation and usage, and another (augmented) set of requirements 
for development. The easiest way to get either environment set up is by going into the `envs` directory and running 
`conda install --file devenv.yml`.

This should install all necessary dependencies. If it fails, you will need to localize a few yourself. Use the 
packages listed in `devenv.yml` as a guide.

Using `conda` is effecitively a hard requirement, as some of the packages `geoplot` depends on are very difficult to 
install otherwise. If you are unfamiliar with `conda`, check its docs for more information.

Due to the outstanding issue mentioned in the `README`, `geoplot` does not work on Windows. Sorry.

`test-env.py` in the `scripts` folders helps check that your environment is in good order.


## Testing

`geoplot` tests are located in the `tests` folder. Any PRs you submit must pass all of the tests located in this 
folder, or fail them for explainable reasons.

At the moment there are three sets of tests. All can be run via `python -m unittest [filename]`. The last of the 
three depends on the `hypothesis` property-based testing module.

## Documentation

Documentation is provided via `sphinx`, plus a handful of plug-ins.

To regenerate the documentation from the current source, navigate to the `docs` folder and run `make html`.

Much of the documentation source is provided in the form of Jupyter notebooks in the `notebooks` folder. To push 
these into HTML through `sphinx`, I use the `nbconvert` utility to generate the `rst` files `sphinx` needs.

To simplify the process, the two sets of documentation pages that depend on these notebooks&mdash;those for the 
tutorial, and those for the API reference&mdash;can be processed automatically by running the `export-tutorial.py` 
and `generate-api-reference-images.py` scripts in the `scripts` folder.

That means that editing these pages is a three-step process: edit and save the notebooks; run the scripts to 
generate `rst` files; and then run `make html` to finally generate the HTML.

The remaining pages are all written in `rst` format. They are located at the top level of the `docs` folder. To 
update these simply edit these files then run `make html` again.

I deploy and serve the HTML from a subfolder on my personal website. Transfer is via a manual copy-paste job.