# Modeling parameters

LOOP_NUMNODE_BOUND = 30 # Maximum number of nodes in a loop
LOOP_LENGTH_BOUND = 40000 # Maximum total length of a loop in meters

CORRIDOR_RADIUS = 100  # Radius for link corridors in meters
ROAD_STRIP_RADIUS = 10 # Radius for link road strips in meters

EPSG = 25832

SEED = 70766

STUDY_AREA = "pilot"

EXCLUDED_TYPES = (13,) # 13 is UNCLASSIFIED (see src/splendor/landuse/types.py)

DIVERSITY_MEASURES = (
    # See src/splendor/diversity.py for supported functions
    "shannon",
    # "inverse_simpson", # Ranks almost identically to Shannon (see notebooks/exp/link_components.ipynb)
    "richness"
) 
RICHNESS_MIN_SHARE = .1

NORMALIZATION_METHODS = ("bounds", "percentile", "rank") # See src/splendor/composite/normalize.py
NORMALIZATION_PERCENTILE = 99

POI_BOUNDS = (0, 3) # Theoretical range of POI categories
GEOMETRIC_FLOOR = 0.25 # Components are lifted to floor + (1 - floor) * x in the geometric mean, so a 0 doesn't zero the score

AGGREGATION_METHODS = ("mean", "geometric") # See src/splendor/composite/aggregate.py

ATTRACTIVENESS = { # Land use type id to attractiveness, in 5 tiers (ids in src/splendor/landuse/types.py)
    12: 1.0, 11: 1.0, 10: 1.0,               # Sea, coast, lagoons/tidal flats
    9: .75, 8: .75, 7: .75, 5: .75, 6: .75,  # Lakes, rivers, mires, heathland, sparsely vegetated
    4: .5,                                   # Forest
    2: .25, 3: .25,                          # Cropland, grassland
    1: 0.0, 14: 0.0,                         # Settlement, road/rail
}

COMPONENT_WEIGHTS = ( # Weights of (land use diversity, POI, attractiveness) in the score
    (0.5, 0.25, 0.25),
    (0.6, 0.2, 0.2),
    (0.4, 0.3, 0.3),
    (1/3, 1/3, 1/3),
)
REFERENCE_WEIGHTS = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)) # Single-component rankings, kept out of the rank summary

MAIN_SPECIFICATION = {
    "measure": "shannon",
    "normalization": "percentile",
    "aggregation": "mean",
    "land_use_weight": 0.5,
    "poi_weight": 0.25,
    "attractiveness_weight": 0.25,
}
