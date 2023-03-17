from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import seaborn as sns
import pandas as pd
import math

__all__ = [
    "circle_radius",
    "circle_indexes",
    "optimal_velocity",
    "is_left_curve",
    "dist_2_points",
    "create_velocity_profiles",
    "optimal_velocity_runner",
]


def get_x_y(racing_track):
    """Get x and y."""

    x = [i[0] for i in racing_track]
    y = [i[1] for i in racing_track]

    return x, y


def distance_between_points(points):
    """Calculate the distance between two points."""

    distance_to_prev = []
    for i in range(len(points)):
        indexes = circle_indexes(points, i, add_index_1=-1, add_index_2=0)[0:2]
        coords = [points[indexes[0]], points[indexes[1]]]
        dist_to_prev = dist_2_points(x1=coords[0][0], x2=coords[1][0], y1=coords[0][1], y2=coords[1][1])
        distance_to_prev.append(dist_to_prev)

    return distance_to_prev


def time_between_points(points, distance_to_prev, velocity):
    """Calculate the time between two points."""

    time_to_prev = [(distance_to_prev[i] / velocity[i]) for i in range(len(points))]

    total_time = sum(time_to_prev)

    return time_to_prev, total_time


def create_velocity_profiles(optimal_speed_vars, size=(10, 7)):
    """Create velocity profiles."""

    velocity_figs = {}

    count = 0
    for velocity in optimal_speed_vars["OPTIMAL_SPEED_VELOCITY"].keys():
        velocity_figs[f"{velocity}"] = plt.figure(
            count + 1,
            figsize=size
        )

        total_time = optimal_speed_vars["OPTIMAL_SPEED_VELOCITY"][velocity]['TOTAL_TIME']

        ax1 = velocity_figs[f"{velocity}"].add_subplot(2, 3, 1)
        sns.scatterplot(
            x=optimal_speed_vars["OPTIMAL_SPEED_X_Y"][0],
            y=optimal_speed_vars["OPTIMAL_SPEED_X_Y"][1],
            hue=optimal_speed_vars["OPTIMAL_SPEED_VELOCITY"][velocity]['VELOCITY'],
            palette="vlag",
            ax=ax1
        ).set_title(f"{velocity} - Total Time: {total_time:.2f} Seconds")

        ax2 = velocity_figs[f"{velocity}"].add_subplot(2, 3, 2)
        sns.scatterplot(
            data=optimal_speed_vars["OPTIMAL_SPEED_VELOCITY"][velocity]['ACTION_PACKAGE'],
            x="steering",
            y="velocity",
            ax=ax2
        )
        ax2.invert_xaxis()
        ax2.set_title(f"{velocity} - Steering and Velocity")

        ax3 = velocity_figs[f"{velocity}"].add_subplot(2, 3, 3)
        sns.lineplot(
            data=optimal_speed_vars["OPTIMAL_SPEED_VELOCITY"][velocity]['ACTION_PACKAGE']["velocity"],
            color="r",
            ax=ax3
        )

        sns.lineplot(
            data=optimal_speed_vars["OPTIMAL_SPEED_VELOCITY"][velocity]['ACTION_PACKAGE']["steering"],
            color="g",
            ax=ax3
        )

        ax3.axhline(0, ls='--', color="g")
        ax3.set_title(f"{velocity} - Speed (red), Steering (green; positive=left)")

        ax4 = velocity_figs[f"{velocity}"].add_subplot(2, 3, 4)
        sns.kdeplot(
            data=optimal_speed_vars["OPTIMAL_SPEED_VELOCITY"][velocity]['NORMALIZED_ACTION_PACKAGE_LESS'],
            x="steering",
            y="velocity",
            ax=ax4,
        )
        ax4.invert_xaxis()
        ax4.set_title(f"{velocity} - Steering and Velocity KDE")

        ax5 = velocity_figs[f"{velocity}"].add_subplot(2, 3, 5)
        sns.scatterplot(
            data=optimal_speed_vars["OPTIMAL_SPEED_VELOCITY"][velocity]['ACTION_PACKAGE'],
            x="steering",
            y="velocity",
            alpha=.1,
            ax=ax5
        )

        sns.scatterplot(
            data=optimal_speed_vars["OPTIMAL_SPEED_VELOCITY"][velocity]['ACTION_SPACE_E'],
            x="steering",
            y="velocity",
            ax=ax5
        )
        ax5.invert_xaxis()
        ax5.set_title(f"{velocity} - KMEANS Steering and Velocity")

        count += 1

    return velocity_figs


# Uses previous and next coords to calculate the radius of the curve
# so you need to pass a list with form [[x1,y1],[x2,y2],[x3,y3]]
# Input 3 coords [[x1,y1],[x2,y2],[x3,y3]]
def circle_radius(coords):
    """Calculate the radius of a circle given three points on the circle"""

    # Flatten the list and assign to variables (makes code easier to read later)
    x1, y1, x2, y2, x3, y3 = [i for sub in coords for i in sub]

    a = x1 * (y2 - y3) - y1 * (x2 - x3) + x2 * y3 - x3 * y2
    b = (x1 ** 2 + y1 ** 2) * (y3 - y2) + (x2 ** 2 + y2 ** 2) * (y1 - y3) + (x3 ** 2 + y3 ** 2) * (y2 - y1)
    c = (x1 ** 2 + y1 ** 2) * (x2 - x3) + (x2 ** 2 + y2 ** 2) * (x3 - x1) + (x3 ** 2 + y3 ** 2) * (x1 - x2)
    d = (x1 ** 2 + y1 ** 2) * (x3 * y2 - x2 * y3) + (x2 ** 2 + y2 ** 2) * \
        (x1 * y3 - x3 * y1) + (x3 ** 2 + y3 ** 2) * (x2 * y1 - x1 * y2)

    # In case a is zero (so radius is infinity)
    try:
        r = abs((b ** 2 + c ** 2 - 4 * a * d) / abs(4 * a ** 2)) ** 0.5
    except:
        r = 999

    return r


# Returns indexes of next index and index+lookfront
# We need this to calculate the radius for next track section.
def circle_indexes(mylist, index_car, add_index_1=0, add_index_2=0):
    """Returns indexes of next index and index+lookfront"""

    list_len = len(mylist)

    # if index >= list_len:
    #     raise ValueError("Index out of range in circle_indexes()")

    # Use modulo to consider that track is cyclical
    index_1 = (index_car + add_index_1) % list_len
    index_2 = (index_car + add_index_2) % list_len

    return [index_car, index_1, index_2]


def optimal_velocity(track, min_speed, max_speed, look_ahead_points):
    """Calculate the optimal velocity for every point on the track"""

    # Calculate the radius for every point of the track
    radius = []
    for i in range(len(track)):
        indexes = circle_indexes(track, i, add_index_1=-1, add_index_2=1)
        coords = [track[indexes[0]],
                  track[indexes[1]], track[indexes[2]]]
        radius.append(circle_radius(coords))

    # Get the max_velocity for the smallest radius
    # That value should multiplied by a constant multiple
    v_min_r = min(radius) ** 0.5
    constant_multiple = min_speed / v_min_r
    print(f"Constant multiple for optimal speed: {constant_multiple}")

    if look_ahead_points == 0:
        # Get the maximal velocity from radius
        max_velocity = [(constant_multiple * i ** 0.5) for i in radius]
        # Get velocity from max_velocity (cap at MAX_SPEED)
        velocity = [min(v, max_speed) for v in max_velocity]
        return velocity, constant_multiple

    else:
        # Looks at the next n radii of points and takes the minimum
        # goal: reduce lookahead until car crashes bc no time to break
        LOOK_AHEAD_POINTS = look_ahead_points
        radius_lookahead = []
        for i in range(len(radius)):
            next_n_radius = []
            for j in range(LOOK_AHEAD_POINTS + 1):
                index = circle_indexes(
                    mylist=radius, index_car=i, add_index_1=j)[1]
                next_n_radius.append(radius[index])
            radius_lookahead.append(min(next_n_radius))
        max_velocity_lookahead = [(constant_multiple * i ** 0.5)
                                  for i in radius_lookahead]
        velocity_lookahead = [min(v, max_speed)
                              for v in max_velocity_lookahead]
        return velocity_lookahead, constant_multiple


def optimal_velocity_runner(optimal_speed_vars, race_line_vars):
    """Calculate the optimal velocity for every point on the track"""

    results = {}

    for look_ahead in optimal_speed_vars["OPTIMAL_SPEED_LOOK_AHEAD_POINTS_LIST"]:
        vel, con = optimal_velocity(
            track=race_line_vars['RACE_LINE_IMPROVED_LOOP_RACE_LINE_CLEAN'],
            min_speed=optimal_speed_vars["OPTIMAL_SPEED_MIN_SPEED"],
            max_speed=optimal_speed_vars["OPTIMAL_SPEED_MAX_SPEED"],
            look_ahead_points=look_ahead
        )

        results[f"LOOK_AHEAD_{look_ahead}"] = {}

        results[f"LOOK_AHEAD_{look_ahead}"]['VELOCITY'] = vel
        results[f"LOOK_AHEAD_{look_ahead}"]['CONSTANT_MULTIPLE'] = con
        results[f"LOOK_AHEAD_{look_ahead}"]['LOOK_AHEAD_POINTS'] = look_ahead
        results[f"LOOK_AHEAD_{look_ahead}"]['DISTANCE_BETWEEN_POINTS'] = distance_between_points(
            race_line_vars['RACE_LINE_IMPROVED_LOOP_RACE_LINE_CLEAN']
        )

        time_between, total_time = time_between_points(
            race_line_vars['RACE_LINE_IMPROVED_LOOP_RACE_LINE_CLEAN'],
            results[f"LOOK_AHEAD_{look_ahead}"]['DISTANCE_BETWEEN_POINTS'],
            results[f"LOOK_AHEAD_{look_ahead}"]['VELOCITY']
        )
        results[f"LOOK_AHEAD_{look_ahead}"]['TIME_BETWEEN_POINTS'] = time_between
        results[f"LOOK_AHEAD_{look_ahead}"]['TOTAL_TIME'] = total_time

        racing_track_everything = []
        for i in range(len(race_line_vars['RACE_LINE_IMPROVED_LOOP_RACE_LINE_CLEAN'])):
            racing_track_everything.append(
                [
                    race_line_vars['RACE_LINE_IMPROVED_LOOP_RACE_LINE_CLEAN'][i][0],
                    race_line_vars['RACE_LINE_IMPROVED_LOOP_RACE_LINE_CLEAN'][i][1],
                    results[f"LOOK_AHEAD_{look_ahead}"]['VELOCITY'][i],
                    results[f"LOOK_AHEAD_{look_ahead}"]['TIME_BETWEEN_POINTS'][i],
                ]
            )
        racing_track_everything = np.around(racing_track_everything, 5).tolist()
        results[f"LOOK_AHEAD_{look_ahead}"]['RACE_PACKAGE'] = racing_track_everything
        results[f"LOOK_AHEAD_{look_ahead}"]['ACTION_PACKAGE'] = calculate_action_space(
            race_line_vars['RACE_LINE_IMPROVED_LOOP_RACE_LINE_CLEAN'],
            results[f"LOOK_AHEAD_{look_ahead}"]['VELOCITY']
        )

        results[f"LOOK_AHEAD_{look_ahead}"]['NORMALIZED_ACTION_PACKAGE'] = normalize_action_space(
            results[f"LOOK_AHEAD_{look_ahead}"]['ACTION_PACKAGE'],
            resample_size=optimal_speed_vars["OPTIMAL_SPEED_RESAMPLE_SIZE"],
            velocity_sd=optimal_speed_vars["OPTIMAL_SPEED_STD_VELOCITY"],
            steering_sd=optimal_speed_vars["OPTIMAL_SPEED_STD_STEERING"],
            max_speed=optimal_speed_vars["OPTIMAL_SPEED_MAX_SPEED"]
        )

        results[f"LOOK_AHEAD_{look_ahead}"]['NORMALIZED_ACTION_PACKAGE_LESS'] = results[f"LOOK_AHEAD_{look_ahead}"]['NORMALIZED_ACTION_PACKAGE'].sample(frac=0.01).reset_index(drop=True) # sample bc less compute time

        results[f"LOOK_AHEAD_{look_ahead}"]['ACTION_SPACE_E'] = get_kmeans_action_space(
            results[f"LOOK_AHEAD_{look_ahead}"]['NORMALIZED_ACTION_PACKAGE'],
            seed=optimal_speed_vars["OPTIMAL_SPEED_SEED"],
            n_clusters=optimal_speed_vars["OPTIMAL_SPEED_N_CLUSTERS"],
            min_speed=optimal_speed_vars["OPTIMAL_SPEED_MIN_SPEED"],
            max_speed=optimal_speed_vars["OPTIMAL_SPEED_MAX_SPEED"]
        )

    return results


# For each point in racing track, check if left curve (returns boolean)
def is_left_curve(coords):
    """Check if the curve is left or right"""

    # Flatten the list and assign to variables (makes code easier to read later)
    x1, y1, x2, y2, x3, y3 = [i for sub in coords for i in sub]

    return ((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)) > 0


# Calculate the distance between 2 points
def dist_2_points(x1, x2, y1, y2):
    """Calculate the distance between 2 points"""

    return abs(abs(x1 - x2) ** 2 + abs(y1 - y2) ** 2) ** 0.5


def calculate_action_space(racing_track, velocity):
    """Calculate the action space for every point on the track"""

    # Calculate the radius for every point of the racing_track
    radius = []
    for i in range(len(racing_track)):
        indexes = circle_indexes(racing_track, i, add_index_1=-1, add_index_2=1)  # CHANGE BACK? 1;2
        coords = [racing_track[indexes[0]],
                  racing_track[indexes[1]], racing_track[indexes[2]]]
        radius.append(circle_radius(coords))

    # Calculate curve direction
    left_curve = []
    for i in range(len(racing_track)):
        indexes = circle_indexes(racing_track, i, add_index_1=-1, add_index_2=1)
        coords = [racing_track[indexes[1]],
                  racing_track[indexes[0]], racing_track[indexes[2]]]
        left_curve.append(is_left_curve(coords))

    # Calculate radius with + and - for direction (+ is left, - is right)
    radius_direction = []
    for i in range(len(racing_track)):
        radius_with_direction = radius[i]
        if not left_curve[i]:
            radius_with_direction *= -1
        radius_direction.append(radius_with_direction)

    # Calculate steering with + and -
    dist_wheels_front_back = 0.165  # meters
    steering = []
    for i in range(len(racing_track)):
        steer = math.degrees(math.asin(dist_wheels_front_back / radius_direction[i]))
        steering.append(steer)

    # Merge relevant lists into dataframe
    all_actions = pd.DataFrame({"velocity": velocity,
                                "steering": steering})

    return all_actions


def normalize_action_space(all_actions, resample_size=1000, velocity_sd=0.1, steering_sd=0.1, max_speed=4.0):
    """Normalize the action space"""

    all_actions_norm = all_actions.copy()

    all_actions_norm_len = len(all_actions_norm)

    # Add gaussian noise to action space
    for i in range(all_actions_norm_len):
        v_true = all_actions_norm.iloc[i]["velocity"]
        s_true = all_actions_norm.iloc[i]["steering"]
        v_norm = np.random.normal(loc=v_true, scale=velocity_sd, size=resample_size)
        s_norm = np.random.normal(loc=s_true, scale=steering_sd, size=resample_size)
        vs_norm = pd.DataFrame(np.column_stack([v_norm, s_norm]), columns=["velocity", "steering"])
        all_actions_norm = pd.concat([all_actions_norm, vs_norm], axis=0, ignore_index=True)

    # Take out actions with max speed, so that they are not affected by gaussian noise
    # We do this because there are disproportionally many points with max speed, so
    # K-Means will focus too much on these
    all_actions_norm = all_actions_norm[all_actions_norm["velocity"] < max_speed]

    # Take out actions with max speed, so that they are not affected by gaussian noise
    # We do this because there are disproportionally many points with max speed, so
    # K-Means will focus too much on these
    all_actions_norm = all_actions_norm[all_actions_norm["velocity"] < max_speed]

    # Add initial actions to action space (to make clustering more focused on initial actions)
    add_n_initial_actions = int(resample_size / 8)
    add_initial_actions = pd.DataFrame()
    for i in range(add_n_initial_actions):
        add_initial_actions = pd.concat([add_initial_actions, all_actions], axis=0, ignore_index=True)
    all_actions_norm = pd.concat([all_actions_norm, add_initial_actions], axis=0, ignore_index=True)

    return all_actions_norm


def get_kmeans_action_space(all_actions_norm, seed=1, n_init=3, n_clusters=100, min_speed=0.1, max_speed=4.0):
    """Get the K-Means action space"""
    x = all_actions_norm

    # Rescale data with minmax
    minmax_scaler = MinMaxScaler()
    x_minmax = pd.DataFrame(minmax_scaler.fit_transform(x),
                            columns=["velocity", "steering"])

    # KMeans
    model = MiniBatchKMeans(
        n_clusters=n_clusters,
        n_init=n_init,
        random_state=seed
    ).fit(x_minmax)

    # Centroids (interpretable)
    minmax_scaler = MinMaxScaler()
    x_minmax_fit = minmax_scaler.fit(x)
    x_centroids = pd.DataFrame(x_minmax_fit.inverse_transform(model.cluster_centers_),
                               columns=["velocity", "steering"])

    # Add 2 manual actions
    # Reason: When car starts new episode, it does not start on or direction of racing line, so
    # it cannot steer enough to get on racing line
    manual_actions = pd.DataFrame({"velocity": [min_speed, max_speed], "steering": [30, -30]})
    x_centroids = pd.concat([x_centroids, manual_actions], ignore_index=True)

    action_space_e = x_centroids.copy()

    return action_space_e
