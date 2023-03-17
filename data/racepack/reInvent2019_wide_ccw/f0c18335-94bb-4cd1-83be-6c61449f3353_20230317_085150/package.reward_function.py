import math


class Reward:
    def __init__(self, verbose=False):
        self.first_racingpoint_index = None
        self.verbose = verbose

    def reward_function(self, params):

        ################## HELPER FUNCTIONS ###################

        def dist_2_points(x1, x2, y1, y2):
            return abs(abs(x1 - x2) ** 2 + abs(y1 - y2) ** 2) ** 0.5

        def closest_2_racing_points_index(racing_coords, car_coords):

            # Calculate all distances to racing points
            distances = []
            for i in range(len(racing_coords)):
                distance = dist_2_points(x1=racing_coords[i][0], x2=car_coords[0],
                                         y1=racing_coords[i][1], y2=car_coords[1])
                distances.append(distance)

            # Get index of the closest racing point
            closest_index = distances.index(min(distances))

            # Get index of the second closest racing point
            distances_no_closest = distances.copy()
            distances_no_closest[closest_index] = 999
            second_closest_index = distances_no_closest.index(
                min(distances_no_closest))

            return [closest_index, second_closest_index]

        def dist_to_racing_line(closest_coords, second_closest_coords, car_coords):

            # Calculate the distances between 2 closest racing points
            a = abs(dist_2_points(x1=closest_coords[0],
                                  x2=second_closest_coords[0],
                                  y1=closest_coords[1],
                                  y2=second_closest_coords[1]))

            # Distances between car and closest and second closest racing point
            b = abs(dist_2_points(x1=car_coords[0],
                                  x2=closest_coords[0],
                                  y1=car_coords[1],
                                  y2=closest_coords[1]))
            c = abs(dist_2_points(x1=car_coords[0],
                                  x2=second_closest_coords[0],
                                  y1=car_coords[1],
                                  y2=second_closest_coords[1]))

            # Calculate distance between car and racing line (goes through 2 closest racing points)
            # try-except in case a=0 (rare bug in DeepRacer)
            try:
                distance = abs(-(a ** 4) + 2 * (a ** 2) * (b ** 2) + 2 * (a ** 2) * (c ** 2) -
                               (b ** 4) + 2 * (b ** 2) * (c ** 2) - (c ** 4)) ** 0.5 / (2 * a)
            except:
                distance = b

            return distance

        # Calculate which one of the closest racing points is the next one and which one the previous one
        def next_prev_racing_point(closest_coords, second_closest_coords, car_coords, heading):

            # Virtually set the car more into the heading direction
            heading_vector = [math.cos(math.radians(
                heading)), math.sin(math.radians(heading))]
            new_car_coords = [car_coords[0] + heading_vector[0],
                              car_coords[1] + heading_vector[1]]

            # Calculate distance from new car coords to 2 closest racing points
            distance_closest_coords_new = dist_2_points(x1=new_car_coords[0],
                                                        x2=closest_coords[0],
                                                        y1=new_car_coords[1],
                                                        y2=closest_coords[1])
            distance_second_closest_coords_new = dist_2_points(x1=new_car_coords[0],
                                                               x2=second_closest_coords[0],
                                                               y1=new_car_coords[1],
                                                               y2=second_closest_coords[1])

            if distance_closest_coords_new <= distance_second_closest_coords_new:
                next_point_coords = closest_coords
                prev_point_coords = second_closest_coords
            else:
                next_point_coords = second_closest_coords
                prev_point_coords = closest_coords

            return [next_point_coords, prev_point_coords]

        def racing_direction_diff(closest_coords, second_closest_coords, car_coords, heading):

            # Calculate the direction of the center line based on the closest waypoints
            next_point, prev_point = next_prev_racing_point(closest_coords,
                                                            second_closest_coords,
                                                            car_coords,
                                                            heading)

            # Calculate the direction in radius, arctan2(dy, dx), the result is (-pi, pi) in radians
            track_direction = math.atan2(
                next_point[1] - prev_point[1], next_point[0] - prev_point[0])

            # Convert to degree
            track_direction = math.degrees(track_direction)

            # Calculate the difference between the track direction and the heading direction of the car
            direction_diff = abs(track_direction - heading)
            if direction_diff > 180:
                direction_diff = 360 - direction_diff

            return direction_diff

        # Gives back indexes that lie between start and end index of a cyclical list 
        # (start index is included, end index is not)
        def indexes_cyclical(start: int, end: int, array_len):
            if start == None:
                start = 0

            if end < start:
                end += array_len

            return [index % array_len for index in range(start, end)]

        # Calculate how long car would take for entire lap, if it continued like it did until now
        def projected_time(first_index, closest_index, step_count, times_list):

            # Calculate how much time has passed since start
            current_actual_time = (step_count - 1) / 15

            # Calculate which indexes were already passed
            indexes_traveled = indexes_cyclical(first_index, closest_index, len(times_list))

            # Calculate how much time should have passed if car would have followed optimals
            current_expected_time = sum([times_list[i] for i in indexes_traveled])

            # Calculate how long one entire lap takes if car follows optimals
            total_expected_time = sum(times_list)

            # Calculate how long car would take for entire lap, if it continued like it did until now
            try:
                projected_time = (current_actual_time / current_expected_time) * total_expected_time
            except:
                projected_time = 9999

            return projected_time

        #################### RACING LINE ######################

        # Optimal racing line for the Spain track
        # Each row: [x,y,speed,timeFromPreviousPoint]
        racing_track = [[-1.43802, -1.59867, 3.00957, 0.09045],
                        [-1.16354, -1.65757, 3.33494, 0.08418],
                        [-0.87681, -1.69688, 3.79065, 0.07635],
                        [-0.58044, -1.71958, 4.0, 0.07431],
                        [-0.27747, -1.72958, 4.0, 0.07578],
                        [0.02895, -1.73169, 4.0, 0.07661],
                        [0.33674, -1.73122, 4.0, 0.07695],
                        [0.64453, -1.73121, 4.0, 0.07695],
                        [0.95231, -1.7315, 4.0, 0.07695],
                        [1.25748, -1.72888, 3.70647, 0.08234],
                        [1.55017, -1.70765, 2.72048, 0.10787],
                        [1.82157, -1.65679, 2.2502, 0.12271],
                        [2.06223, -1.5719, 1.95982, 0.13021],
                        [2.26495, -1.45417, 1.75679, 0.13344],
                        [2.42487, -1.30794, 1.60465, 0.13504],
                        [2.53816, -1.13889, 1.48435, 0.1371],
                        [2.60049, -0.95342, 1.3847, 0.1413],
                        [2.6054, -0.75946, 1.3, 0.14925],
                        [2.54295, -0.5696, 1.38254, 0.14456],
                        [2.41703, -0.39872, 1.64746, 0.12884],
                        [2.24283, -0.25262, 1.88566, 0.12057],
                        [2.02823, -0.13455, 2.26061, 0.10835],
                        [1.78298, -0.04292, 2.99145, 0.08752],
                        [1.51908, 0.03007, 3.40838, 0.08033],
                        [1.24917, 0.12571, 3.68945, 0.07762],
                        [0.98695, 0.23719, 4.0, 0.07123],
                        [0.7314, 0.36152, 4.0, 0.07105],
                        [0.48137, 0.49594, 4.0, 0.07097],
                        [0.23557, 0.63785, 4.0, 0.07096],
                        [-0.00736, 0.78472, 4.0, 0.07097],
                        [-0.24885, 0.93405, 4.0, 0.07098],
                        [-0.47007, 1.07045, 4.0, 0.06497],
                        [-0.69068, 1.19896, 3.92877, 0.06499],
                        [-0.911, 1.31357, 3.21803, 0.07717],
                        [-1.13212, 1.40978, 2.78981, 0.08644],
                        [-1.35518, 1.48388, 2.49562, 0.09418],
                        [-1.58062, 1.5322, 2.27427, 0.10138],
                        [-1.80769, 1.55068, 2.10161, 0.1084],
                        [-2.03396, 1.53488, 1.95994, 0.11573],
                        [-2.25463, 1.48013, 1.84058, 0.12353],
                        [-2.46161, 1.38249, 1.7376, 0.13171],
                        [-2.64327, 1.24026, 1.64483, 0.14027],
                        [-2.78566, 1.05458, 1.98812, 0.11769],
                        [-2.89586, 0.84214, 2.0568, 0.11636],
                        [-2.97156, 0.60528, 2.12451, 0.11704],
                        [-3.00903, 0.3455, 2.1973, 0.11945],
                        [-3.00287, 0.06437, 2.24977, 0.12499],
                        [-2.94438, -0.23625, 2.28444, 0.13406],
                        [-2.84417, -0.50367, 2.31713, 0.12324],
                        [-2.7119, -0.73989, 2.34499, 0.11545],
                        [-2.55296, -0.94789, 2.37063, 0.11042],
                        [-2.37056, -1.12939, 2.39326, 0.10752],
                        [-2.16651, -1.28495, 2.40408, 0.10673],
                        [-1.94152, -1.41392, 2.56749, 0.10101],
                        [-1.69801, -1.518, 2.76339, 0.09583]]

        ################## INPUT PARAMETERS ###################

        # Read all input parameters
        all_wheels_on_track = params['all_wheels_on_track']
        x = params['x']
        y = params['y']
        distance_from_center = params['distance_from_center']
        is_left_of_center = params['is_left_of_center']
        heading = params['heading']
        progress = params['progress']
        steps = params['steps']
        speed = params['speed']
        steering_angle = params['steering_angle']
        track_width = params['track_width']
        waypoints = params['waypoints']
        closest_waypoints = params['closest_waypoints']
        is_offtrack = params['is_offtrack']

        ############### OPTIMAL X,Y,SPEED,TIME ################

        # Get closest indexes for racing line (and distances to all points on racing line)
        closest_index, second_closest_index = closest_2_racing_points_index(
            racing_track, [x, y])

        # Get optimal [x, y, speed, time] for closest and second closest index
        optimals = racing_track[closest_index]
        optimals_second = racing_track[second_closest_index]

        # Save first racingpoint of episode for later
        if self.verbose == True:
            self.first_racingpoint_index = 0  # this is just for testing purposes
        if steps == 1:
            self.first_racingpoint_index = closest_index

        ################ REWARD AND PUNISHMENT ################

        ## Define the default reward ##
        reward = 1

        ## Reward if car goes close to optimal racing line ##
        DISTANCE_MULTIPLE = 1
        dist = dist_to_racing_line(optimals[0:2], optimals_second[0:2], [x, y])
        distance_reward = max(1e-3, 1 - (dist / (track_width * 0.5)))
        reward += distance_reward * DISTANCE_MULTIPLE

        ## Reward if speed is close to optimal speed ##
        SPEED_DIFF_NO_REWARD = 1
        SPEED_MULTIPLE = 2
        speed_diff = abs(optimals[2] - speed)
        if speed_diff <= SPEED_DIFF_NO_REWARD:
            # we use quadratic punishment (not linear) bc we're not as confident with the optimal speed
            # so, we do not punish small deviations from optimal speed
            speed_reward = (1 - (speed_diff / (SPEED_DIFF_NO_REWARD)) ** 2) ** 2
        else:
            speed_reward = 0
        reward += speed_reward * SPEED_MULTIPLE

        # Reward if less steps
        REWARD_PER_STEP_FOR_FASTEST_TIME = 1
        STANDARD_TIME = 10
        FASTEST_TIME = 9
        times_list = [row[3] for row in racing_track]
        projected_time = projected_time(self.first_racingpoint_index, closest_index, steps, times_list)
        try:
            steps_prediction = projected_time * 15 + 1
            reward_prediction = max(1e-3, (-REWARD_PER_STEP_FOR_FASTEST_TIME * (FASTEST_TIME) /
                                           (STANDARD_TIME - FASTEST_TIME)) * (
                                            steps_prediction - (STANDARD_TIME * 15 + 1)))
            steps_reward = min(REWARD_PER_STEP_FOR_FASTEST_TIME, reward_prediction / steps_prediction)
        except:
            steps_reward = 0
        reward += steps_reward

        # Zero reward if obviously wrong direction (e.g. spin)
        direction_diff = racing_direction_diff(
            optimals[0:2], optimals_second[0:2], [x, y], heading)
        if direction_diff > 30:
            reward = 1e-3

        # Zero reward of obviously too slow
        speed_diff_zero = optimals[2] - speed
        if speed_diff_zero > 0.5:
            reward = 1e-3

        ## Incentive for finishing the lap in less steps ##
        REWARD_FOR_FASTEST_TIME = 1500  # should be adapted to track length and other rewards
        STANDARD_TIME = 20  # seconds (time that is easily done by model)
        FASTEST_TIME = 11  # seconds (best time of 1st place on the track)
        if progress == 100:
            finish_reward = max(1e-3, (-REWARD_FOR_FASTEST_TIME /
                                       (15 * (STANDARD_TIME - FASTEST_TIME))) * (steps - STANDARD_TIME * 15))
        else:
            finish_reward = 0
        reward += finish_reward

        ## Zero reward if off track ##
        if all_wheels_on_track == False:
            reward = 1e-3

        ####################### VERBOSE #######################

        if self.verbose == True:
            print("Closest index: %i" % closest_index)
            print("Distance to racing line: %f" % dist)
            print("=== Distance reward (w/out multiple): %f ===" % (distance_reward))
            print("Optimal speed: %f" % optimals[2])
            print("Speed difference: %f" % speed_diff)
            print("=== Speed reward (w/out multiple): %f ===" % speed_reward)
            print("Direction difference: %f" % direction_diff)
            print("Predicted time: %f" % projected_time)
            print("=== Steps reward: %f ===" % steps_reward)
            print("=== Finish reward: %f ===" % finish_reward)

        #################### RETURN REWARD ####################

        # Always return a float value
        return float(reward)


reward_object = Reward()  # add parameter verbose=True to get noisy output for testing


def reward_function(params):
    return reward_object.reward_function(params)
