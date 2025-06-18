# utils/gis/crs_codes.py

"""
Coordinate Reference System (CRS) codes for GIS transformations.
Uses EPSG identifiers for clarity.
"""

# WGS 84 geographic coordinate system (Longitude, Latitude in degrees)
CRS_WGS84_GEO = "EPSG:4326"  # World Geodetic System 1984 (lat/long) [oai_citation:3‡epsg.io](https://epsg.io/4326#:~:text=WGS%2084%20,System%201984%2C%20used%20in%20GPS)

# WGS 84 / UTM zone 36N projected coordinate system (Easting, Northing in meters)
CRS_WGS84_UTM_36N = "EPSG:32636"  # WGS84 in UTM Zone 36N (meters) [oai_citation:4‡epsg.io](https://epsg.io/32636#:~:text=EPSG%3A32636)