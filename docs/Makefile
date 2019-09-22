# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

docs:
	GEOPLOT_HOME=$$(dirname $$PWD) && \
	jupyter nbconvert --to rst --output-dir $$GEOPLOT_HOME/docs/quickstart/ \
		$$GEOPLOT_HOME/notebooks/tutorials/Quickstart.ipynb && \
	jupyter nbconvert --to rst --output-dir $$GEOPLOT_HOME/docs/user_guide/ \
		$$GEOPLOT_HOME/notebooks/tutorials/Working_with_Geospatial_Data.ipynb && \
	jupyter nbconvert --to rst --output-dir $$GEOPLOT_HOME/docs/user_guide/ \
		$$GEOPLOT_HOME/notebooks/tutorials/Working_with_Projections.ipynb && \
	jupyter nbconvert --to rst --output-dir $$GEOPLOT_HOME/docs/user_guide/ \
		$$GEOPLOT_HOME/notebooks/tutorials/Customizing_Plots.ipynb && \
	make html
# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)