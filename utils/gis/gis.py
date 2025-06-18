from pyproj import Transformer
from pyproj import Geod
from utils.gis import crs_codes


def project_points(points, crs_from, crs_to):
    """Transform a set of points from one coordinate system to another.

    Args:
        points: list of points: [(x1, y1), (x2, y2), (x3, y3)] or 1 point: (x, y)
        crs_from: The given points coordinate system code. Use the codes under
            utils.gis.crs_codes package.
        crs_to: The target coordinate system code. Use the codes under
            utils.gis.crs_codes package.

    Returns:
        list of points transformed to the target coordinate system.
    """
    transformer = Transformer.from_crs(crs_from, crs_to, always_xy=True)
    try:
        trans_list = list(transformer.itransform(points))
    except:
        trans_list = list(transformer.itransform([points]))
    return trans_list


def longlat2utm(points):
    """Transform a set of points from WGS84 GEO coordinate system to UTM 36N.

    Args:
        points: list of points: [(lon1, lat1), (lon2, lat2)] or 1 point: (lon,
            lat)

    Returns:
        list of points in UTM_36N coordinate system (UTM_X-Easting,
        UTM_Y-Northing).
    """
    UTM_points = project_points(points, crs_codes.CRS_WGS84_GEO, crs_codes.
                                CRS_WGS84_UTM_36N)
    return UTM_points


def utm2longlat(points):
    """Transform a set of points from UTM_36 coordinate system to WGS84_GEO.

    Args:
        points: list of points: ((x1, y1), (x2, y2) or 1 point: (x, y)

    Returns:
        list of points in WGS84_GEO coordinate system (lon, lat).
    """
    longlat_points = project_points(points, crs_codes.CRS_WGS84_UTM_36N,
                                    crs_codes.CRS_WGS84_GEO)
    return longlat_points


def distance_between_points(p1, p2):
    """Calculate the distance between 2 points.

    Args:
        p1: Point 1 In WGS84 GEO
        p2: Point 2 In WGS84 GEO

    Returns:
        The distance between point 1 to point 2.
    """
    lono, lato = p1
    lon1, lat1 = p2
    geod = Geod(ellps='WGS84')
    _, _, distance = geod.inv(lono, lato, lon1, lat1)
    return distance