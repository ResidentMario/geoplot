"""
Sankey of Napoleon's disastarous march into Russia
==================================================

This example reproduces a famous historical flow map: Charles Joseph Minard's map depicting
Napoleon's disastrously costly 1812 march on Russia during the Napoleonic Wars.

This ``sankey`` demonstrates how to build and use a custom ``matplotlib`` colormap. It also
demonstrates using the ``mplleaflet`` library to quickly and easily transform the resulting plot
into an scrolly-panny webmap.

`Click here to see the interactive webmap version. 
<http://bl.ocks.org/ResidentMario/ac2db57d1c6652ddbc4112a3d318c746>`_
"""

import geopandas as gpd
import geoplot as gplt
import matplotlib.pyplot as plt
import mplleaflet
from matplotlib.colors import LinearSegmentedColormap

napoleon_troop_movements = gpd.read_file(gplt.datasets.get_path('napoleon_troop_movements'))

colors = [(215/255, 193/255, 126/255), (37/255, 37/255, 37/255)]
cm = LinearSegmentedColormap.from_list('minard', colors)

gplt.sankey(
    napoleon_troop_movements,
    scale='survivors', limits=(0.5, 45),
    hue='direction',
    cmap=cm
)
fig = plt.gcf()
plt.savefig("minard-napoelon-russia.png", bbox_inches='tight', pad_inches=0.1)

# Uncomment and run the following line of code to save an interactive webmap.
# mplleaflet.save_html(fig, fileobj='minard-napoleon-russia.html')
