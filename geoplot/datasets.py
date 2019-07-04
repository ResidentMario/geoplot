"""
Example dataset fetching utility. Used in docs.
"""

src = 'https://raw.githubusercontent.com/ResidentMario/geoplot-data/master'


def get_path(dataset_name):
    """
    Returns the URL path to an example dataset suitable for reading into ``geopandas``.
    """
    if dataset_name == 'usa_cities':
        return f'{src}/usa-cities.geojson'
    elif dataset_name == 'contiguous_usa':
        return f'{src}/contiguous-usa.geojson'
    elif dataset_name == 'nyc_collision_factors':
        return f'{src}/nyc-collision-factors.geojson'
    elif dataset_name == 'nyc_boroughs':
        return f'{src}/nyc-boroughs.geojson'
    elif dataset_name == 'ny_census':
        return f'{src}/ny-census-partial.geojson'
    elif dataset_name == 'obesity_by_state':
        return f'{src}/obesity-by-state.tsv'
    elif dataset_name == 'la_flights':
        return f'{src}/la-flights.geojson'
    elif dataset_name == 'dc_roads':
        return f'{src}/dc-roads.geojson'
    elif dataset_name == 'nyc_map_pluto_sample':
        return f'{src}/nyc-map-pluto-sample.geojson'
    elif dataset_name == 'nyc_collisions_sample':
        return f'{src}/nyc-collisions-sample.csv'
    elif dataset_name == 'boston_zip_codes':
        return f'{src}/boston-zip-codes.geojson'
    elif dataset_name == 'boston_airbnb_listings':
        return f'{src}/boston-airbnb-listings.geojson'
    elif dataset_name == 'napoleon_troop_movements':
        return f'{src}/napoleon-troop-movements.geojson'
    elif dataset_name == 'nyc_fatal_collisions':
        return f'{src}/nyc-fatal-collisions.geojson'
    elif dataset_name == 'nyc_injurious_collisions':
        return f'{src}/nyc-injurious-collisions.geojson'
    elif dataset_name == 'nyc_police_precincts':
        return f'{src}/nyc-police-precincts.geojson'
    elif dataset_name == 'nyc_parking_tickets':
        return f'{src}/nyc-parking-tickets-sample.geojson'
    elif dataset_name == 'world':
        return f'{src}/world.geojson'
    elif dataset_name == 'melbourne':
        return f'{src}/melbourne.geojson'
    elif dataset_name == 'melbourne_schools':
        return f'{src}/melbourne-schools.geojson'
    else:
        raise ValueError(
            f'The dataset_name value {dataset_name!r} is not in the list of valid names.'
        )
