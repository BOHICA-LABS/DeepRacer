import numpy as np
from shapely.geometry import LineString, Polygon
from .convert import convert_to_shapely


def test_convert_to_shapely():
    # Test case 1: empty input
    waypoints = np.empty((0, 6))
    result = convert_to_shapely(waypoints)
    assert result is None

    # Test case 2: input with only one row
    waypoints = np.array([[0, 0, 1, 1, 2, 2]])
    result = convert_to_shapely(waypoints)
    assert result is None

    # Test case 3: input with two rows
    waypoints = np.array([[0, 0, 1, 1, 2, 2],
                          [1, 1, 2, 2, 3, 3]])
    result = convert_to_shapely(waypoints)
    assert len(result) == 7
    assert result['center_line'].shape == (2, 2)
    assert isinstance(result['l_center_line'], LineString)
    assert isinstance(result['l_inner_border'], LineString)
    assert isinstance(result['l_outer_border'], LineString)
    assert isinstance(result['road_poly'], Polygon)

    # Test case 4: input with more than two rows
    waypoints = np.array([[0, 0, 1, 1, 2, 2],
                          [1, 1, 2, 2, 3, 3],
                          [2, 2, 3, 3, 4, 4]])
    result = convert_to_shapely(waypoints)
    assert len(result) == 7
    assert result['center_line'].shape == (3, 2)
    assert isinstance(result['l_center_line'], LineString)
    assert isinstance(result['l_inner_border'], LineString)
    assert isinstance(result['l_outer_border'], LineString)
    assert isinstance(result['road_poly'], Polygon)


