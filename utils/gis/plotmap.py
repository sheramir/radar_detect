"""plot data on map object. Data can be entered in WGS84 or UTM-36 coordinates.
Created on Thu Jun 25 13:12:54 2020
@author: Amir Sher
"""
from os import path
import json
import random
from pandas import DataFrame
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Ellipse, Circle, Wedge
import cartopy.crs as ccrs
from sensors_metadata.sensors import Sensors
from utils.gis import gis
from utils.string_utils import fix_hebrew

#Lat - North/South
# Long - East/West
#Constants
CCRS_GEOD = ccrs.Geodetic()
CCRS_UTM = ccrs.UTM(zone='36')
CCRS_PLATECARREE = ccrs.PlateCarree()
UTM_X_MIN = 30000
UTM_X_MAX = 800000
UTM_Y_MIN = 3000000
UTM_Y_MAX = 4000000
LAT_MIN = 27
LAT_MAX = 37
LON_MIN = 25
LON_MAX = 43
TEXT_MARGIN_PERCENT = 0.01
# Current map ID's are: ISRAEL, OTEF_GAZA, ISRAEL_CENTER_NORTH, ISRAEL_NORTH,
# ISRAEL_SOUTH


class mymap:
    """Map object."""
    def __init__(self, map_ID=''):
        """Create mymap map object.

        Parameters
        ----------
        map_ID : string, optional
            ID of map object. The default is ''. Map IDs are created from json
            file.

        Returns
        -------
        None.

        """
        self.map_title = ""
        self.map_name = ""
        self.map_path = ""
        self.map_bounds = ()
        self.map_points = []
        self.map_lines = []
        self.map_shapes = []
        self.layers = set([])
        self.basecolors = ('b', 'g', 'r', 'c', 'm', 'y', 'k')
        self.basemarkers = ('.', 'o', 'v', '^', '<', '>', '8', 's', 'p', 'P',
                            '*', 'h', 'H', '+', 'x', 'X', 'D', 'd', '|', '_')
        self.line_styles = ('solid', 'dotted', 'dashed', 'dashdot')
        self.map_colors = list(self.basecolors)
        self.points_markers = list(self.basemarkers)
        self.maps = self.get_maps_from_files()
        if map_ID != '':
            self.choose_map(map_ID)

    def get_maps_from_files(self, filepath=""):
        """Read available maps from json file and PNG files.

        Parameters
        ----------
        filepath : string, optional
            Path of the map files. The default is "" (current path).

        Returns
        -------
        dict
            dict containing the metedata of the maps from the json file.
        """
        if filepath == "":
            filepath = path.join(path.dirname(path.abspath(__file__)), 'maps')
        maps_json_filepath = path.join(filepath, 'maps.json')
        with open(maps_json_filepath, 'r', encoding='utf-8') as maps_json_file:
            maps_data = json.load(maps_json_file)
        for thismap in maps_data['maps']:
            thismap['filename'] = path.join(filepath, thismap['filename'])
            lonlat0 = (thismap['bounds_west'], thismap['bounds_south'])
            lonlat1 = (thismap['bounds_east'], thismap['bounds_north'])
            [utm0, utm1] = gis.longlat2utm([lonlat0, lonlat1])
            thismap['bounds_x_min'], thismap['bounds_y_min'] = utm0
            thismap['bounds_x_max'], thismap['bounds_y_max'] = utm1
        return maps_data['maps']

    def add_map(self, ID, name, filename, bounds):
        """Add new map to the maps collection.

        Parameters
        ----------
        ID : string
            New map ID.
        name : string
            The name of the map (for displaying map title).
        filename : string
            Full path and file name of the map PNG file.
        bounds : list of float
            Map boundries in WGS84 [south, north, west, east).

        Returns
        -------
        bool
            True if success, False if error (in filename or bounds).
        """
        if path.exists(filename):
            if len(bounds) != 4:
                print('ERROR in bounds!')
                return False
            else:
                bounds.sort() # only valid in Israel region
                thismap = {'ID': ID,
                           'name': name,
                           'filename': filename,
                           'bounds_west': bounds[2],
                           'bounds_east': bounds[3],
                           'bounds_south': bounds[0],
                           'bounds_north': bounds[1]}
                self.maps.append(thismap)
                return True
        else:
            print(f'ERROR: file {filename} not found!')
            return False

    def choose_map(self, ID):
        """Choose map from the available maps collection by map ID.

        Parameters
        ----------
        ID : string
            Map ID.

        Returns
        -------
        boolean
            False if error, True if success.
        """
        found = False
        for thismap in self.maps:
            if thismap['ID'] == ID:
                map_path = thismap['filename']
                if not path.exists(map_path):
                    print(f'ERROR: Could not find map file: {self.map_path}')
                    found = False
                else:
                    self.map_path = map_path
                    self.map_name = fix_hebrew(thismap['name'])
                    x_min = thismap['bounds_x_min']
                    x_max = thismap['bounds_x_max']
                    y_min = thismap['bounds_y_min']
                    y_max = thismap['bounds_y_max']
                    s = thismap['bounds_south']
                    n = thismap['bounds_north']
                    w = thismap['bounds_west']
                    e = thismap['bounds_east']
                    self.map_bounds = (x_min, x_max, y_min, y_max)
                    self.map_bounds_latlon = (w, e, s, n)
                    self.map_width_utm = x_max - x_min
                    self.map_width_latlon = e - w
                    found = True
        if not found:
            print(f'ERROR: Could not find map ID: {ID}')
        return found

    def print_maps(self):
        """Print metadata about the maps collection.

        Returns
        -------
        None.

        """
        df = DataFrame.from_dict(self.maps)
        print(df.to_string())

    def get_maps_id(self):
        """Return a list of the avialable map IDs/

        Returns
        -------
        maps_id : string
            List of map IDs from the collection of maps.
        """
        maps_id = [amap['ID'] for amap in self.maps]
        return maps_id

    def add_point(self, lon, lat, name="", layer="", color='random', marker=
                  'random'):
        """Add one point to the map.

        Parameters
        ----------
        lon : float
            Longitude in WGS84 or X coordinate (easting) in UTM-36.
        lat : float
            Latitude in WGS84 or Y coordinate (northing) in UTM-36.
        name : string, optional
            Name of the point to display on map. The default is "".
        layer : string, optional
            Name of the layer for that point. The default is "".
        color : string, optional
            Color code. The default is 'random'.
        marker : string, optional
            Marker style code. The default is 'random'.

        Returns
        -------
        boolean
            True if success, False if error.
        """
        colorcode, self.map_colors = self.get_unige_color(color, self.map_colors)
        markercode, self.points_markers = self.get_marker(marker, self.
                                                          points_markers)
        marker = colorcode + markercode
        data_type = self.get_point_type((lon, lat))
        if data_type == 'UTM':
            transform = CCRS_UTM
        elif data_type == 'LATLON':
            transform = CCRS_GEOD
        else:
            print('ERROR in point data!')
            return False
        point = {'Lon': lon, 'Lat': lat, 'Name': fix_hebrew(name),
                 'Layer': layer, 'Marker': marker, 'Transform': transform}
        self.map_points.append(point)
        self.layers.add(layer)
        return True

    def add_points(self, points, name="", layer="", color='random', marker=
                   'random'):
        """Add a list of points to the map.

        Parameters
        ----------
        points : list of float tuple pairs.
            List of points ((x1,y1), (x2,y2)].
        name : string, optional
            Name of the points to display on map. The default is "".
        layer : string, optional
            Name of the layer for that points. The default is "".
        color : string, optional
            Color code. The default is 'random'.
        marker : string, optional
            Marker style code. The default is 'random'.

        Returns
        -------
        boolean
            True if success, False if error.
        """
        points, error = self.fix_points_list(points, 1)
        if not error:
            for point in points:
                self.add_point(*point, name=name, layer=layer, color=color, marker
                               = marker)
            return True
        else:
            print('ERROR in line data!')
            return False

    def add_line(self, line, name="", layer="", color='random', width=2, style=
                 'solid'):
        """Add line to the map by connecting at least 2 points.

        Parameters
        ----------
        line : list of float tuple pairs.
            List of the line points [(x1,y1), (x2,y2)].
        name : string, optional
            Name of the line to display on map. The default is "".
        layer : string, optional
            Name of the layer for that line. The default is "".
        color : string, optional
            Color code. The default is 'random'.
        width : integer, optional
            Width of the line. The default is 2.
        style : string, optional
            Style code (solid, dashed, dashdot, dotted). The default is 'solid'.

        Returns
        -------
        boolean
            True if success, False if error.
        """
        colorcode, self.map_colors = self.get_unige_color(color, self.map_colors)
        line, error = self.fix_points_list(line, 2)
        if not error:
            data_type = self.get_point_type(line)
            if data_type == 'UTM':
                transform = CCRS_UTM
            elif data_type == 'LATLON':
                transform = CCRS_GEOD
            else:
                print('ERROR in line data!')
                return False
            line = {'Line': line, 'Name': fix_hebrew(name), 'Layer': layer,
                    'Color': colorcode, 'Width': width, 'Style': style, 'Transform':
                        transform}
            self.map_lines.append(line)
            self.layers.add(layer)
            return True
        else:
            print('ERROR in line data!')
            return False

    def add_polygon(self, poly, name="", layer="", facecolor='random', edgecolor=
                    'random', alpha=0.8):
        """Add a closed polygon to the map.

        Parameters
        ----------
        poly : list of float tuple pairs.
            List of polygon points, at least 3 [(x1,y1), (x2,y2), (x3,y3)].
        name : string, optional
            Name of the polygon to display on map. The default is "".
        layer : string, optional
            Name of the layer for that polygon. The default is "".
        facecolor : string, optional
            Color code for the polygon face. The default is 'random'.
        edgecolor : string, optional
            Color code for the polygon edge. The default is 'random'.
        alpha : float, optional
            Transperancy level of the face color. The default is 0.8.

        Returns
        -------
        boolean
            True if success, False if error.
        """
        colorcode, self.map_colors = self.get_unige_color(facecolor, self.
                                                          map_colors)
        edgecolor = self.get_color(edgecolor)
        poly, error = self.fix_points_list(poly, 3)
        if not error:
            x, y = list(zip(*poly))
            pol_center = (self.get_mean(x), self.get_mean(y))
            data_type = self.get_point_type(poly)
            if data_type == 'UTM':
                transform = CCRS_UTM
            elif data_type == 'LATLON':
                transform = CCRS_GEOD
            else:
                print('ERROR in polygon data!')
                return False
            pol = {'Shape': 'Polygon', 'xy': poly, 'Name': fix_hebrew(name), 'Layer'
                   : layer,
                   'FaceColor': colorcode, 'EdgeColor': edgecolor, 'Alpha': alpha,
                   'Center': pol_center, 'Transform': transform}
            self.map_shapes.append(pol)
            self.layers.add(layer)
            return True
        else:
            print('Error in polygon data!')
            return False

    def add_circle(self, xy, rad, name="", layer="", facecolor='random',
                   edgecolor='random', alpha=0.8):
        """Add a circle to the map.

        Parameters
        ----------
        xy : tuple of float
            Coordinates of the circle center.
        rad : float
            Radius of the circle in meters.
        name : string, optional
            Name of the circle to display on map. The default is ''.
        layer : string, optional
            Name of the layer for that circle. The default is ''.
        facecolor : string, optional
            Color code for the circle face. The default is 'random'.
        edgecolor : string, optional
            Color code for the circle edge. The default is 'random'.
        alpha : float, optional
            Transperancy level of the face color. The default is 0.8.

        Returns
        -------
        boolean
            True if success, False if error.
        """
        colorcode, self.map_colors = self.get_unige_color(facecolor, self.
                                                          map_colors)
        edgecolor = self.get_color(edgecolor)
        data_type = self.get_point_type(xy)
        if data_type == 'LATLON':
            xy = gis.longlat2utm(xy)[0]
        elif data_type == 'UNKNOWN':
            print('ERROR in point data!')
            return False
        transform = CCRS_UTM
        circle = {'Shape': 'Circle', 'xy': xy, 'Radius': rad,
                  'Name': fix_hebrew(name), 'Layer': layer,
                  'FaceColor': colorcode, 'EdgeColor': edgecolor, 'Alpha': alpha,
                  'Transform': transform}
        self.map_shapes.append(circle)
        self.layers.add(layer)
        return True

    def add_ellipse(self, xy, width, height, angle, name="", layer="", facecolor=
                    'random', edgecolor='random', alpha=0.8):
        """Add an ellipse to the map.

        Parameters
        ----------
        xy : tuple of float
            Coordinates of the ellipse center.
        width : float
            Width of the ellipse in meters.
        height : float
            Height of the ellipse in meters.
        angle : float
            Angle of the ellipse.
        name : string, optional
            Name of the ellipse to display on map. The default is ''.
        layer : string, optional
            Name of the layer for that ellipse. The default is ''.
        facecolor : string, optional
            Color code for the ellipse face. The default is 'random'.
        edgecolor : string, optional
            Color code for the ellipse edge. The default is 'random'.
        alpha : float, optional
            Transperancy level of the face color. The default is 0.8.

        Returns
        -------
        boolean
            True if success, False if error.
        """
        colorcode, self.map_colors = self.get_unige_color(facecolor, self.
                                                          map_colors)
        edgecolor = self.get_color(edgecolor)
        data_type = self.get_point_type(xy)
        if data_type == 'LATLON':
            xy = gis.longlat2utm(xy)[0]
        elif data_type == 'UNKNOWN':
            print('ERROR in point data!')
            return False
        transform = CCRS_UTM
        ellipse = {'Shape': 'Ellipse', 'xy': xy,
                   'Width': width, 'Height': height, 'Angle': angle,
                   'Name': fix_hebrew(name), 'Layer': layer,
                   'FaceColor': colorcode, 'EdgeColor': edgecolor, 'Alpha': alpha,
                   'Transform': transform}
        self.map_shapes.append(ellipse)
        self.layers.add(layer)
        return True

    def add_wedge(self, xy, rad, theta1, theta2, width=None, name="", layer="",
                  facecolor='random', edgecolor='random', alpha=0.8):
        """Add a wedge to the map.

        Parameters
        ----------
        xy : tuple of float
            Coordinates of the wedge center.
        rad : float
            Radius of the wedge in meters.
        theta1 : float
            Angle of the wedge start.
        theta2 : float
            Angle of the wedge end.
        width : float, optional
            Width of the wedge in meters. The default is None.
        name : string, optional
            Name of the wedge to display on map. The default is "".
        layer : string, optional
            Name of the layer for that wedge. The default is "".
        facecolor : string, optional
            Color code for the wedge face. The default is 'random'.
        edgecolor : string, optional
            Color code for the wedge edge. The default is 'random'.
        alpha : float, optional
            Transperancy level of the face color. The default is 0.8.

        Returns
        -------
        boolean
            True if success, False if error.
        """
        colorcode, self.map_colors = self.get_unige_color(facecolor, self.
                                                          map_colors)
        edgecolor = self.get_color(edgecolor)
        data_type = self.get_point_type(xy)
        if data_type == 'LATLON':
            xy = gis.longlat2utm(xy)[0]
        elif data_type == 'UNKNOWN':
            print('ERROR in point data!')
            return False
        transform = CCRS_UTM
        wedge = {'Shape': 'Wedge', 'xy': xy,
                 'Radius': rad, 'Theta1': theta1, 'Theta2': theta2, 'Width': width,
                 'Name': fix_hebrew(name), 'Layer': layer,
                 'FaceColor': colorcode, 'EdgeColor': edgecolor, 'Alpha': alpha,
                 'Transform': transform}
        self.map_shapes.append(wedge)
        self.layers.add(layer)
        return True

    def points_to_polygon(self, layer, name="", facecolor='random', edgecolor=
                          'random', alpha=0.8, remove=False):
        """Convert a list of points in a specific layer to a closed polygon.

        Parameters
        ----------
        layer : string
            Layer name.
        name : string, optional
            Name of the polygon to display in map. The default is "".
        facecolor : string, optional
            Color code for the polygon face. The default is 'random'.
        edgecolor : string, optional
            Color code for the polygon edge. The default is 'random'.
        alpha : float, optional
            Transperancy level of the face color. The default is 0.8.
        remove : boolean, optional
            True Remove points from the map after converting to polygon. The
            default is False.

        Returns
        -------
        None.

        """
        poly = []
        i = -1
        for point in list(self.map_points):
            i += 1
            if point['Layer'] == layer:
                poly.append((point['Lon'], point['Lat']))
                if remove:
                    self.map_points.pop(i)
                    i -= 1
        if len(poly) > 0:
            self.add_polygon(poly, name=name, layer=layer, facecolor=facecolor,
                             edgecolor=edgecolor, alpha=alpha)

    def points_to_line(self, layer, name="", color='random', width=2, style=
                       'solid', remove=False):
        """Convert a list of points in a specific layer to a line.

        Parameters
        ----------
        layer : string
            Layer name.
        name : string, optional
            Name of the line to display in map. The default is "".
        color : string, optional
            Color code for the line. The default is 'random'.
        width : integer, optional
            Width of the line. The default is 2.
        style : string, optional
            Style of the line. The default is 'solid'.
        remove : boolean, optional
            True Remove points from the map after converting to line. The
            default is False.

        Returns
        -------
        None.

        """
        line = []
        i = -1
        for point in list(self.map_points):
            i += 1
            if point['Layer'] == layer:
                line.append((point['Lon'], point['Lat']))
                if remove:
                    self.map_points.pop(i)
                    i -= 1
        if len(line) > 0:
            self.add_line(line, name=name, layer=layer, color=color, width=width,
                          style=style)

    def fix_points_list(self, points, min_points=1):
        """Make sure a list of points is in the right format and check if it
        meets a minmum points requirement.

        Parameters
        ----------
        points : list of tuples or a tuple
            List of points, ot 1 point without a list.
        min_points : integer, optional
            A minimum requirement for the number of points in the list. The
            default is 1.

        Returns
        -------
        points_list : list of tuples
            A fixed list of tuple pairs (points).
        error : boolean
            True if number of points equals or exceeds the minimum requirement..
        """
        try:
            x, y = list(zip(*points))
            points_list = list(zip(x, y))
        except:
            points_list = [points]
        error = False if len(points_list) >= min_points else True
        return points_list, error

    def add_title(self, title):
        """Add a title to the map plot.

        Parameters
        ----------
        title : string
            Map title (otherwise the title would be the map name).

        Returns
        -------
        string
            The map title. If in hebrew the text would be reveresed (fix for the
            plot).
        """
        if title != '':
            self.map_title = fix_hebrew(title)
        return self.map_title

    def get_marker(self, marker, marker_list):
        """Return a marker from a collection of markers. Make sure they don't
        repeat.

        Parameters
        ----------
        marker : string
            A marker code, or 'random' for a randomly generated marker.
            Available codes: '.', 'o', 'v', '^', '8', 's', 'p', 'P', '*', 'h', 'H',
            '+', 'x', 'X', 'D', 'd', '|', '_'
        marker_list : list of string
            List of available marker codes.

        Returns
        -------
        string
            A marker code.
        marker_list : list of string
            A reduced marker list without the returned marker.
        """
        if len(marker_list) == 0:
            marker_list = list(self.basemarkers)
        if marker == 'random':
            marker_index = random.randint(0, len(marker_list) - 1)
            return_marker = marker_list.pop(marker_index)
        else:
            if marker in marker_list:
                marker_index = marker_list.index(marker)
                return_marker = marker_list.pop(marker_index)
            else:
                return_marker = marker
        return return_marker, marker_list

    def get_unige_color(self, color, color_list):
        """Return a color from a collection of colors. Make sure they don't
        repeat.

        Parameters
        ----------
        color : string
            A color code or 'random' for a randomly generated color.
        color_list : list of string
            List of available color codes.
            Available colors: 'b', 'g', 'r', 'c', 'm', 'y', 'k'

        Returns
        -------
        return_color : string
            A color code.
        color_list : list of string
            List of available color codes, without the returned color.
        """
        if len(color_list) == 0:
            color_list = list(self.basecolors)
        if color == 'random':
            color_index = random.randint(0, len(color_list) - 1)
            return_color = color_list.pop(color_index)
        else:
            if color in color_list:
                color_index = color_list.index(color)
                return_color = color_list.pop(color_index)
            else:
                return_color = color
        return return_color, color_list

    def get_color(self, color):
        """Return a color from a collection of colors. Random colors could
        repeat.

        Parameters
        ----------
        color : string
            A color code. or 'random' for a randomly generated color.

        Returns
        -------
        return_color : string
            A color code.
        """
        if color == 'random':
            color_index = random.randint(0, len(self.basecolors) - 1)
            return_color = self.basecolors[color_index]
        else:
            return_color = color
        return return_color

    def get_sensors(self, color='b', marker='o', layer='Sensors'):
        """Get all sensors locations and put them in a 'Sensors' layer.

        Parameters
        ----------
        color : string, optional
            Color of the sensor points. The default is 'b' (blue).
        marker : string, optional
            The Marker type of the sensors. The default is 'o'.
        layer : string, optional
            Layer name of the sensors. The default is 'Sensors'.

        Returns
        -------
        None.
        """
        mysensors = Sensors().sensors
        for sensor in mysensors:
            if sensor['port'] > 0:
                self.add_point(lat=sensor['latitude'], lon=sensor['longitude'],
                               name=sensor['name'], layer=layer, color=color,
                               marker=marker)

    def get_point_type(self, points):
        """Check if all points in a list are in UTM or WGS84 ('LATLON')

        Parameters
        ----------
        points : list of tuples
            List of coordinate points.

        Returns
        -------
        point_type : string
            UTM for UTM-36, 'LATLON' for WGS84. Otherwise 'UNKNOWN'.
        """
        is_utm_count = 0
        is_latlon_count = 0
        point_type = 'UNKOWN'
        points, error = self.fix_points_list(points, 1)

        for point in points:
            is_utm = (UTM_X_MIN <= point[0] <= UTM_X_MAX) and (UTM_Y_MIN <= point
                                                               [1] <= UTM_Y_MAX)
            is_latlon = (LON_MIN <= point[0] <= LON_MAX) and (LAT_MIN <= point[1]
                                                              <= LAT_MAX)
            if is_utm:
                is_utm_count += 1
            if is_latlon:
                is_latlon_count += 1

        if is_utm_count == len(points):
            point_type = 'UTM'
        elif is_latlon_count == len(points):
            point_type = 'LATLON'
        return point_type

    def point_in_map(self, x, y):
        """Check if a point is within the chosen map boundries.

        Parameters
        ----------
        x : float
            Longitude in WGS84 or X coordinate (easting) in UTM-36.
        y : float
            Latitude in WGS84 or Y coordinate (northing) in UTM-36.

        Returns
        -------
        boolean
            True if the (x,y) is within the map boundries.
        """
        if self.map_path != '':
            point_type = self.get_point_type((x, y))
            if point_type == 'UTM':
                x_min, x_max, y_min, y_max = self.map_bounds
            elif point_type == 'LATLON':
                x_min, x_max, y_min, y_max = self.map_bounds_latlon
            else:
                x_min, x_max, y_min, y_max = [0, 0, 0, 0]
            in_map = (x_min <= x <= x_max) and (y_min <= y <= y_max)
            return in_map
        else:
            print('ERROR: Valid map was not chosen yet!')
            return False

    def get_mean(self, x):
        """Return mean of array (saves importing numpy just for mean calculation).

        Parameters
        ----------
        x : list
            A list of numbers.

        Returns
        -------
        float
            The mean of the list.
        """
        return sum(x) / len(x)

    def get_text_margin(self, crs, margin_percent=TEXT_MARGIN_PERCENT):
        """Return margin for printing text label on a map.

        Parameters
        ----------
        crs : cartopy crs type
            The crs system. (ccrs.Geodetic or ccrs.UTM)
        margin_percent : float, optional
            The margin percentage from the map size. The default is
            TEXT_MARGIN_PERCENT = 0.01.

        Returns
        -------
        text_margin : float
            Calculated text margin according to the map size.
        """
        text_margin_utm = margin_percent * self.map_width_utm # margin for utm
        # labels
        text_margin_latlon = margin_percent * self.map_width_latlon # margin for
        # latlon labels
        text_margin = text_margin_utm if crs == CCRS_UTM else text_margin_latlon
        return text_margin

    def plotmap(self, map_ID="", title="", layers='all', dpi=80):
        """Plot the map of the chosen map ID, with all its entities.

        Parameters
        ----------
        map_ID : string, optional
            Map ID of the chosen map. The default is "", assuming a map was
            chosen in init or by choose_map.
        title : string, optional
            Title of the map. The default is the name of the map or a title
            chosen by add_title.
        layers : list of strings, optional
            List of layer names. The map will show only entities in this layers.
            The default is 'all'.
        dpi : integer, optional
            Dots per inche. the size of the image. The default is 80.

        Returns
        -------
        None.
        """
        if map_ID != "":
            self.choose_map(map_ID)
        if self.map_path != '':
            if title == "":
                title = self.map_name
            else:
                title = fix_hebrew(title)

            #Draw map image according to projection
            img = plt.imread(self.map_path)
            height, width, depth = img.shape
            figsize = (width / float(dpi), height / float(dpi))
            fig = plt.figure(figsize=figsize)
            ax = plt.axes(projection=CCRS_UTM)
            ax.set_xlim(left=self.map_bounds[0], right=self.map_bounds[1])
            ax.set_ylim(bottom=self.map_bounds[2], top=self.map_bounds[3])
            plt.title(title)
            ax.imshow(img, origin='upper', extent=self.map_bounds, transform=
                      CCRS_UTM, aspect='equal')

            #Draw geometries on the map
            if layers == 'all' or layers == '':
                layers = self.layers

            #Draw points
            for point in self.map_points:
                lon = point['Lon']
                lat = point['Lat']
                if self.point_in_map(lon, lat) and point['Layer'] in layers:
                    transform = point['Transform']
                    ax.plot(lon, lat, point['Marker'], markersize=7, transform=
                            transform)
                    text_margin = self.get_text_margin(transform)
                    ax.text(lon + text_margin, lat, point['Name'], transform=
                            transform)

            #Draw lines
            for line in self.map_lines:
                line_points = line['Line']
                x, y = list(zip(*line_points))
                if line['Layer'] in layers:
                    transform = line['Transform']
                    ax.plot(x, y, c=line['Color'], ls=line['Style'], lw=line[
                        'Width'], transform=transform)

                    text_margin = self.get_text_margin(transform)
                    text_x, text_y = (x[0] + text_margin, y[0])
                    if self.point_in_map(text_x, text_y):
                        ax.text(text_x, text_y, line['Name'], transform=transform)

            #Draw shapes
            for shape in self.map_shapes:
                if shape['Layer'] in layers:
                    xy = shape['xy']
                    text_xy = xy
                    fc = shape['FaceColor']
                    ec = shape['EdgeColor']
                    alpha = shape['Alpha']
                    name = shape['Name']
                    transform = shape['Transform']
                    shape_type = shape['Shape']

                    if shape_type == 'Polygon':
                        obj = Polygon(xy, closed=True, fc=fc, ec=ec, alpha=alpha,
                                      transform=transform)
                        text_xy = shape['Center']

                    if shape_type == 'Circle':
                        rad = shape['Radius']
                        obj = Circle(xy, radius=rad, fc=fc, ec=ec, alpha=alpha,
                                     transform=transform)

                    if shape_type == 'Ellipse':
                        width = shape['Width']
                        height = shape['Height']
                        angle = shape['Angle']
                        obj = Ellipse(xy, width=width, height=height, angle=angle,
                                      fc=fc, ec=ec, alpha=alpha, transform=transform)

                    if shape_type == 'Wedge':
                        rad = shape['Radius']
                        theta1 = shape['Theta1']
                        theta2 = shape['Theta2']
                        width = shape['Width']
                        obj = Wedge(xy, rad, theta1, theta2, width, fc=fc, ec=ec,
                                    alpha=alpha, transform=transform)
                    ax.add_patch(obj)
                    if self.point_in_map(*text_xy):
                        ax.text(*text_xy, name, horizontalalignment='center',
                                verticalalignment='center', transform=transform)

            plt.show()
        else:
            print("ERROR: None map was chosen")


#########################
def main():
    newmap = mymap()
    newmap.get_sensors()
    newmap.add_line(((35.235857, 31.840576), (34.838555, 31.969417)), color='random',
                   width=2)
    newmap.add_line([(35.235857, 31.840576), (34.5120099, 31.595165)], color=
                   'random', style='dotted')
    #newmap.get_sensors(layer='line')
    #newmap.points_to_line(layer='line', remove=True)
    newmap.add_circle((34.501469, 31.479329), 5000)
    newmap.add_point(34.301, 31.479, name='lonlat_point', color='k', marker='o')
    newmap.add_point(655624.823, 3403701.605, name='utm_point', color='b', marker=
                     'o')
    newmap.add_circle((650624.823, 3473701.605), 8000, name='מעגל')
    newmap.add_point(35.568, 32.688, layer='poly1')
    newmap.add_point(35.302, 32.619, layer='poly1')
    newmap.add_point(35.500, 32.500, layer='poly1')
    newmap.add_point(35.785, 32.544, layer='poly1')
    newmap.points_to_polygon(layer='poly1', name='polygon1', remove=True)
    newmap.add_points([(35.822, 32.811), (35.842, 32.911)], name='new_points')
    newmap.print_maps() # Print info of all available maps
    mapsID = newmap.get_maps_id() # Get a list of all available maps
    print(mapsID)
    newmap.plotmap(map_ID='OTEF_GAZA', title='סנסורים בעוטף')
    newmap.plotmap(map_ID='ISRAEL_CENTER_NORTH')
    newmap.plotmap(map_ID='ISRAEL_NORTH', layers='all')
    newmap.plotmap(map_ID='ISRAEL SOUTH')
    newmap.plotmap(map_ID='ISRAEL_CENTER')
    newmap.plotmap(map_ID='ISRAEL')


if __name__ == '__main__':
    main()