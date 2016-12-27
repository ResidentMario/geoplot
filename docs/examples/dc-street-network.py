import sys; sys.path.insert(0, '../')
import geoplot as gplt
import geoplot.crs as gcrs
import geopandas as gpd
import matplotlib.pyplot as plt


# The data being used here is the DC portion of the Federal Highway Administration's roadway traffic volume
# shapefiles, retrieved from http://www.fhwa.dot.gov/policyinformation/hpms/shapefiles.cfm. The AADT column of
# interest here is the FHA's traffic volume estimates.


# Load the data.
dc = gpd.read_file("../../data/us_roads/District_Sections.shp")


# Plot the data.
ax = gplt.sankey(dc, path=dc.geometry, projection=gcrs.AlbersEqualArea(), scale='aadt',
                 limits=(0.1, 10))
plt.title("Streets in Washington DC by Average Daily Traffic, 2015")
plt.savefig("largest-cities-usa.png", bbox_inches='tight', pad_inches=0.1)