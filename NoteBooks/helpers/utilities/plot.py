from shapely.geometry import LineString


__all__ = [
    "print_border"
]


def plot_coords(ax, ob):
    """Plot the coordinates of a LineString or Polygon geometry"""

    x, y = ob.xy
    ax.plot(x, y, '.', color='#999999', zorder=1)


def plot_bounds(ax, ob):
    """Plot the bounds of a LineString or Polygon geometry"""

    x, y = zip(*list((p.x, p.y) for p in ob.boundary))
    ax.plot(x, y, '.', color='#000000', zorder=1)


def plot_line(ax, ob):
    """Plot a LineString geometry"""

    x, y = ob.xy
    ax.plot(x, y, color='cyan', alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2)


def print_border(ax, waypoints, inner_border_waypoints, outer_border_waypoints):
    """Print the borders of a track"""

    line = LineString(waypoints)
    plot_coords(ax, line)
    plot_line(ax, line)

    line = LineString(inner_border_waypoints)
    plot_coords(ax, line)
    plot_line(ax, line)

    line = LineString(outer_border_waypoints)
    plot_coords(ax, line)
    plot_line(ax, line)
