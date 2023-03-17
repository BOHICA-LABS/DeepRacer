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
        racing_track = [[2.89403, 0.70184, 3.79253, 0.08175],
                        [3.16466, 0.693, 4.0, 0.06769],
                        [3.43314, 0.68823, 4.0, 0.06713],
                        [3.73805, 0.68548, 3.46458, 0.08801],
                        [4.10749, 0.68438, 2.72738, 0.13546],
                        [4.41121, 0.68403, 2.31766, 0.13105],
                        [4.70859, 0.68388, 2.04959, 0.1451],
                        [5.32, 0.68405, 1.85389, 0.3298],
                        [5.47294, 0.68837, 1.70328, 0.08982],
                        [5.73669, 0.70621, 1.58272, 0.16703],
                        [5.99188, 0.74397, 1.48305, 0.17394],
                        [6.2125, 0.80009, 1.39574, 0.16309],
                        [6.40159, 0.8723, 1.39574, 0.14502],
                        [6.56417, 0.95947, 1.36126, 0.13552],
                        [6.70337, 1.06122, 1.31665, 0.13096],
                        [6.8203, 1.17773, 1.31665, 0.12536],
                        [6.91413, 1.30941, 1.3, 0.12438],
                        [6.98181, 1.45656, 1.3, 0.12459],
                        [7.02175, 1.61695, 1.3, 0.12714],
                        [7.02831, 1.78832, 1.3, 0.13192],
                        [6.99394, 1.9655, 1.3, 0.13883],
                        [6.91406, 2.13948, 1.3, 0.14726],
                        [6.78253, 2.29682, 1.57511, 0.1302],
                        [6.6155, 2.43265, 1.75637, 0.12257],
                        [6.42189, 2.544, 2.01683, 0.11074],
                        [6.21112, 2.6322, 2.44261, 0.09354],
                        [5.99094, 2.70254, 3.0994, 0.07458],
                        [5.76663, 2.76273, 2.35659, 0.09855],
                        [5.56291, 2.81599, 2.35659, 0.08936],
                        [5.36026, 2.87264, 2.35659, 0.08929],
                        [5.15931, 2.93486, 2.35659, 0.08927],
                        [4.96058, 3.00487, 2.35659, 0.08941],
                        [4.76448, 3.08511, 2.35659, 0.08991],
                        [4.57237, 3.18404, 2.63608, 0.08197],
                        [4.38341, 3.29902, 3.04342, 0.07268],
                        [4.19707, 3.42683, 3.13703, 0.07203],
                        [4.01268, 3.56362, 2.83389, 0.08102],
                        [3.82932, 3.70508, 2.60388, 0.08894],
                        [3.67897, 3.81731, 2.4207, 0.07751],
                        [3.52789, 3.92462, 2.26916, 0.08166],
                        [3.37566, 4.02541, 2.13912, 0.08535],
                        [3.22171, 4.11836, 2.02641, 0.08874],
                        [3.06532, 4.20236, 1.92826, 0.09207],
                        [2.90547, 4.27644, 1.84221, 0.09564],
                        [2.74071, 4.33958, 1.75963, 0.10027],
                        [2.56887, 4.39046, 1.68438, 0.1064],
                        [2.38649, 4.42699, 1.61252, 0.11535],
                        [2.18768, 4.44523, 1.49978, 0.13311],
                        [1.96237, 4.43671, 1.49978, 0.15034],
                        [1.69985, 4.38288, 1.49978, 0.17868],
                        [1.42039, 4.26083, 1.49978, 0.20333],
                        [1.16503, 4.06146, 1.49978, 0.21602],
                        [0.96753, 3.78363, 1.49978, 0.22728],
                        [0.87363, 3.43687, 1.95029, 0.1842],
                        [0.85453, 3.09651, 2.30451, 0.14793],
                        [0.8766, 2.81168, 2.23781, 0.12766],
                        [0.91229, 2.57756, 1.99439, 0.11875],
                        [0.96294, 2.31103, 1.81435, 0.14952],
                        [1.00825, 2.10289, 1.6741, 0.12725],
                        [1.0623, 1.90085, 1.5598, 0.13408],
                        [1.12998, 1.70432, 1.5598, 0.13326],
                        [1.21209, 1.52228, 1.5598, 0.12803],
                        [1.30759, 1.3607, 1.5598, 0.12033],
                        [1.41609, 1.22064, 1.5598, 0.11358],
                        [1.53931, 1.10095, 1.5598, 0.11013],
                        [1.68365, 1.00024, 1.80992, 0.09724],
                        [1.85113, 0.91238, 1.97289, 0.09586],
                        [2.04923, 0.83633, 2.18684, 0.09703],
                        [2.28992, 0.77293, 2.48683, 0.10009],
                        [2.58494, 0.72608, 2.94742, 0.10135]]

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
        STANDARD_TIME = 20
        FASTEST_TIME = 11
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
