"""
Sankey of Napoleon's march on Moscow with custom colormap
=========================================================

This example reproduces a famous historical flow map: Charles Joseph Minard's map depicting
Napoleon's disastrously costly 1812 march on Russia during the Napoleonic Wars.

This plot demonstrates building and using a custom ``matplotlib`` colormap. To learn more refer to
`the matplotlib documentation
<https://matplotlib.org/gallery/images_contours_and_fields/custom_cmap.html>`_.

`Click here <https://bl.ocks.org/ResidentMario/ac2db57d1c6652ddbc4112a3d318c746>`_ to see an
interactive scrolly-panny version of this webmap built with ``mplleaflet``. To learn more about
``mplleaflet``, refer to `the mplleaflet GitHub repo <https://github.com/jwass/mplleaflet>`_.
"""

import geopandas as gpd
import geoplot as gplt
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

napoleon_troop_movements = gpd.read_file(gplt.datasets.get_path('napoleon_troop_movements'))

colors = [(215 / 255, 193 / 255, 126 / 255), (37 / 255, 37 / 255, 37 / 255)]
cm = LinearSegmentedColormap.from_list('minard', colors)

gplt.sankey(
    napoleon_troop_movements,
    scale='survivors', limits=(0.5, 45),
    hue='direction',
    cmap=cm
)
fig = plt.gcf()
plt.savefig("minard-napoelon-russia.png", bbox_inches='tight', pad_inches=0.1)

# Uncomment and run the following line of code to save as an interactive webmap.
# mplleaflet.save_html(fig, fileobj='minard-napoleon-russia.html')
