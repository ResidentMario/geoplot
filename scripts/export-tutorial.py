"""
Exports tutorial materials written in Jupyter notebooks in the ../notebooks/tutorial folder to RST pages and their
support files in the ../docs/tutorial folder.
"""
import subprocess
import os

# Get the list of tutorial notebooks.
tutorial_notebooks = [f for f in os.listdir("../notebooks/tutorial") if (".ipynb" in f) and ("checkpoints" not in f)]
# Run them in-place.
for notebook in tutorial_notebooks:
    print(" ".join(["jupyter", "nbconvert", "--to", "rst", "../notebooks/tutorial/{0}".format(notebook),
                    "--output", "../../scripts/{0}".format(notebook.replace(".ipynb", ".rst"))]))
    subprocess.run(["jupyter", "nbconvert", "--to", "rst", "../notebooks/tutorial/{0}".format(notebook),
                    "--output", "../../scripts/{0}".format(notebook.replace(".ipynb", ".rst"))])

# Get the list of generated files.
gened_files = [f for f in os.listdir(".") if (".py" not in f)]

# Move them to where they need to be.
for file in gened_files:
    os.rename(file, "../docs/tutorial/{0}".format(file))
