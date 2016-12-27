"""
Runs generate-api-reference-images.ipynb, which reloads all of the images in the API Reference in-place.
# cf. https://nbconvert.readthedocs.io/en/latest/usage.html#notebook-and-preprocessors
"""
import subprocess
subprocess.run(["jupyter", "nbconvert", "--to", "notebook", "../notebooks/generate-api-reference-images.ipynb"])