"""
Exports tutorial materials written in Jupyter notebooks in the ../notebooks/tutorial folder to RST pages and their
support files in the ../docs/tutorial folder.
"""
import subprocess
import os

# Get the list of tutorial notebooks.
tutorial_notebooks = [f for f in os.listdir("../notebooks/tutorial") if (".ipynb" in f) and ("checkpoints" not in f)]
# Run them in-place.
# import pdb; pdb.set_trace()

for notebook in tutorial_notebooks:
    print(" ".join(["jupyter", "nbconvert", "--to", "rst", "../notebooks/tutorial/{0}".format(notebook),
                    "--output", "../docs/tutorial/{0}".format(notebook.replace(".ipynb", ".rst"))]))
    subprocess.run(["jupyter", "nbconvert", "--to", "rst", "../notebooks/tutorial/{0}".format(notebook),
                    "--output", "../../docs/tutorial/{0}".format(notebook.replace(".ipynb", ".rst").lower())])

# # Get the list of generated files.
# gened_files = [f for f in os.listdir(".") if (".py" not in f)]
#
# # Move them to where they need to be. Lowercase the filenames along the way, otherwise it causes issues once the
# # files are hosted.
# for file in gened_files:
#     with open(file, "r") as f:
#         buffer = f.read()
#         title = file.title()[:-4]
#         # import pdb; pdb.set_trace()
#     with open(file, "w") as f:
#         f.write(buffer.replace("/scripts/{0}".format(title), "/docs/tutorial/{0}".format(title)))
#     os.rename(file, "../docs/tutorial/{0}".format(file.lower()))
