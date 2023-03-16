from shapely.geometry import Point, Polygon
from shapely.geometry import LineString
import numpy as np

__all__ = [
    "convert_to_shapely",
    "dist_2_points",
    "x_perc_width",
]


def convert_to_shapely(waypoints):
    """Convert to shapely object."""

    waypoints_copy = waypoints.copy()

    s_dict = {
        "center_line": waypoints_copy[:, 0:2],
        "inner_border": waypoints_copy[:, 2:4],
        "outer_border": waypoints_copy[:, 4:6],
        'l_center_line': LineString(waypoints_copy[:, 0:2]),
        'l_inner_border': LineString(waypoints_copy[:, 2:4]),
        'l_outer_border': LineString(waypoints_copy[:, 4:6]),
        'road_poly': Polygon()
    }

    s_dict['road_poly'] = Polygon(np.vstack((s_dict['l_outer_border'], np.flipud(s_dict['l_inner_border']))))

    return s_dict


def dist_2_points(x1, x2, y1, y2):
    """Calculate the distance between two points."""

    return abs(abs(x1-x2)**2 + abs(y1-y2)**2)**0.5


def x_perc_width(waypoint, perc_width):
    """Calculate the x coordinate of the waypoint at a given percentage of the width of the road."""

    center_x, center_y, inner_x, inner_y, outer_x, outer_y = waypoint

    # TODO: this is not the width, but the distance between the two borders. Dead code?
    width = dist_2_points(inner_x, outer_x, inner_y, outer_y)

    delta_x = outer_x-inner_x
    delta_y = outer_y-inner_y

    inner_x_new = inner_x + delta_x/2 * (1-perc_width)
    outer_x_new = outer_x - delta_x/2 * (1-perc_width)
    inner_y_new = inner_y + delta_y/2 * (1-perc_width)
    outer_y_new = outer_y - delta_y/2 * (1-perc_width)

    return [center_x, center_y, inner_x_new, inner_y_new, outer_x_new, outer_y_new]