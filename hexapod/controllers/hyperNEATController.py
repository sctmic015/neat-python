import math
import neat

import numpy as np

# radius, offset, step_height, phase, duty_factor
from hexapod.controllers.anglequee import anglequee

tripod_gait = [0.15, 0, 0.05, 0.5, 0.5,  # leg 1
               0.15, 0, 0.05, 0.0, 0.5,  # leg 2
               0.15, 0, 0.05, 0.5, 0.5,  # leg 3
               0.15, 0, 0.05, 0.0, 0.5,  # leg 4
               0.15, 0, 0.05, 0.5, 0.5,  # leg 5
               0.15, 0, 0.05, 0.0, 0.5]  # leg 6

wave_gait = [0.15, 0, 0.05, 0 / 6, 5 / 6,  # leg 1
             0.15, 0, 0.05, 1 / 6, 5 / 6,  # leg 2
             0.15, 0, 0.05, 2 / 6, 5 / 6,  # leg 3
             0.15, 0, 0.05, 3 / 6, 5 / 6,  # leg 4
             0.15, 0, 0.05, 4 / 6, 5 / 6,  # leg 5
             0.15, 0, 0.05, 5 / 6, 5 / 6]  # leg 6

quadruped_gait = [0.15, 0, 0.05, 0 / 3, 2 / 3,  # leg 1
                  0.15, 0, 0.05, 1 / 3, 2 / 3,  # leg 2
                  0.15, 0, 0.05, 2 / 3, 2 / 3,  # leg 3
                  0.15, 0, 0.05, 0 / 3, 2 / 3,  # leg 4
                  0.15, 0, 0.05, 1 / 3, 2 / 3,  # leg 5
                  0.15, 0, 0.05, 2 / 3, 2 / 3]  # leg 6

stationary = [0.18, 0, 0, 0, 0] * 6


class Controller:

    def __init__(self, params=tripod_gait, crab_angle=0.0, body_height=0.14, period=1.0, velocity=0.1, dt=1 / 240, ann = None, printangles = False, activations = 2):
        # link lengths
        self.count = 0
        self.activations = activations
        self.ann = ann
        self.l_1 = 0.05317
        self.l_2 = 0.10188
        self.l_3 = 0.14735

        self.dt = dt
        self.period = period
        self.velocity = velocity
        self.crab_angle = crab_angle
        self.body_height = body_height
        self.printangles = printangles

        self.array_dim = int(np.around(period / dt))
        # print("Hi")
        # print(self.array_dim)

        self.positions = np.empty((0, self.array_dim))
        self.velocities = np.empty((0, self.array_dim))

        self.angles = np.empty((0, self.array_dim))
        self.speeds = np.empty((0, self.array_dim))

        params = np.array(params).reshape(6, 5)

        # Generate trajectories for each leg
        for leg_index in range(6):
            foot_positions, foot_velocities = self.__leg_traj(leg_index, params[
                leg_index])  ## Gets positions and velocities per leg
            joint_angles, joint_speeds = self.__inverse_kinematics(foot_positions, foot_velocities)
            # check that the position is achieveble
            achieved_positions = self.forward_kinematics(joint_angles)
            valid = np.all(np.isclose(foot_positions, achieved_positions))
            # if not valid:
            #	raise RuntimeError('Desired foot trajectory not achieveable')

            self.positions = np.append(self.positions, foot_positions, axis=0)
            self.velocities = np.append(self.velocities, foot_velocities, axis=0)
            self.angles = np.append(self.angles, joint_angles, axis=0)
            self.speeds = np.append(self.speeds, joint_speeds, axis=0)

        initial_angle = self.angles[:, 0]
        self.current_angle = initial_angle
        self.aq = anglequee()
        self.aq.add(self.current_angle)


    def joint_angles(self, t):

        self.ann.reset()
        #sinewave = math.sin(self.count * 2 * math.pi/3) * math.pi
        self.count += 1
        sinewave = math.sin(t * 2 * math.pi / 3 * 16) * 2*math.pi
        coswave = math.cos(t * 2 * math.pi / 3 * 16) * 2*math.pi
        input_angles = np.append(self.current_angle, sinewave)
        input_angles = np.append(input_angles, coswave)
        # for i in range(self.activations):
        for i in range(self.activations):
            current_angles = self.ann.activate(input_angles)
        # Current theory is that the for loop makes the program to slow to do NEAT
        for i in range(len(current_angles)):
            if i % 3 == 0:
                current_angles[i] = (current_angles[i] * 0.91 * 2) - 0.91
            elif i % 3 == 1:
                current_angles[i] = (((current_angles[i] -0)*(0.64+0.2))/(1-0))-0.2
            else:
                current_angles[i] = (((current_angles[i] -0)*(-1.4+2.11))/(1-0))-2.11

        # for i in range(len(current_angles)):
        #     if i % 3 == 0:
        #         current_angles[i] = (current_angles[i] * 2.0944) - (2.0944 / 2)
        #     elif i % 3 == 1:
        #         current_angles[i] = current_angles[i] * 0.63
        #     else:
        #         current_angles[i] = current_angles[i] * 0.7 - 2.095

        self.current_angle = current_angles

        return current_angles

        # sinewave = math.sin(t*2*math.pi)
        # coswave = math.cos(t*2*math.pi)
        # input_angles = self.aq.get_moving_average()
        # input_angles = np.append(self.current_angle, sinewave)
        # input_angles = np.append(input_angles, coswave)
        # #for i in range(self.activations):
        # current_angles = self.ann.activate(input_angles)
        #
        # # Current theory is that the for loop makes the program to slow to do NEAT
        # for i in range(len(current_angles)):
        #     if i % 3 == 0:
        #         current_angles[i] = (current_angles[i] * 2) - (2 / 2)
        #     elif i % 3 == 1:
        #         current_angles[i] = current_angles[i] * 0.63
        #     else:
        #         current_angles[i] = current_angles[i] * 0.7 - 2.095
        #
        # self.aq.add(current_angles)
        # current_angles = self.aq.get_moving_average()
        # self.current_angle = current_angles
        #
        # return current_angles

    def joint_speeds(self, t):
        k = int(((t % self.period) / self.period) * self.array_dim)
        return self.speeds[:, k]

    # Returns x, y, z trajectory path points for a leg in the leg coordinate space
    def __leg_traj(self, leg_index, leg_params):
        leg_angle = (np.pi / 3.0) * (leg_index)
        radius, offset, step_height, phase, duty_factor = leg_params
        stride = self.velocity * duty_factor * self.period

        # key points in path
        mid = np.zeros(3)
        mid[0] = radius * np.cos(offset)
        mid[1] = radius * np.sin(offset)
        mid[2] = -self.body_height + 0.014 + step_height

        start = np.zeros(3)
        start[0] = mid[0] + (stride / 2) * np.cos(-leg_angle + self.crab_angle)
        start[1] = mid[1] + (stride / 2) * np.sin(-leg_angle + self.crab_angle)
        start[2] = -self.body_height + 0.014

        end = np.zeros(3)
        end[0] = mid[0] - (stride / 2) * np.cos(-leg_angle + self.crab_angle)
        end[1] = mid[1] - (stride / 2) * np.sin(-leg_angle + self.crab_angle)
        end[2] = -self.body_height + 0.014

        # compute support path
        support_dim = int(np.around(self.array_dim * duty_factor))
        support_positions, support_velocities = self.__support_traj(start, end, support_dim)

        # compute swing path
        swing_dim = int(np.around(self.array_dim * (1.0 - duty_factor)))
        swing_positions, swing_velocities = self.__swing_traj(end, mid, start, swing_dim)

        # combine support and swing paths
        positions = np.append(support_positions, swing_positions, axis=1)
        velocities = np.append(support_velocities, swing_velocities, axis=1)

        # shift points according to phase
        phase_shift = int(np.around(phase * self.array_dim))
        positions = np.roll(positions, phase_shift, axis=1)
        velocities = np.roll(velocities, phase_shift, axis=1)

        return positions, velocities

    ##def __leg_trag_ann(self, leg_index, leg_params):

    def __support_traj(self, start, end, num):
        # Straight line from start to end point
        positions = np.linspace(start, end, num, axis=1)
        # the servo velocity is the same for each position
        duration = num * self.dt
        # duration can sometimes be 0
        with np.errstate(divide='ignore', invalid='ignore'):
            velocity = ((end - start) / duration).reshape(3, 1)
        # velocities = np.nan_to_num(velocities, nan=0.0, posinf=0.0, neginf=0.0)
        velocities = np.tile(velocity, num)

        return positions, velocities

    # Generates a smooth trajectory from start point to end point through the via point
    # The start point must be ignored as this is included in the support trajectory
    def __swing_traj(self, start, via, end, num):
        t = np.ones((7, num))
        tf = num * self.dt
        time = np.linspace(0, tf, num)

        for i in range(7):
            t[i, :] = np.power(time, i)

        a_0 = start
        a_1 = np.zeros(3)
        a_2 = np.zeros(3)
        a_3 = (2 / (tf ** 3)) * (32 * (via - start) - 11 * (end - start))
        a_4 = -(3 / (tf ** 4)) * (64 * (via - start) - 27 * (end - start))
        a_5 = (3 / (tf ** 5)) * (64 * (via - start) - 30 * (end - start))
        a_6 = -(32 / (tf ** 6)) * (2 * (via - start) - (end - start))

        positions = np.stack([a_0, a_1, a_2, a_3, a_4, a_5, a_6], axis=-1).dot(t)
        velocities = np.stack([a_1, 2 * a_2, 3 * a_3, 4 * a_4, 5 * a_5, 6 * a_6, np.zeros(3)], axis=-1).dot(t)

        return positions, velocities

    # Inverse kinematics to calculate joint angles to reach provided points
    def __inverse_kinematics(self, foot_position, foot_speed):
        l_1, l_2, l_3 = self.l_1, self.l_2, self.l_3

        x, y, z = foot_position
        dx, dy, dz = foot_speed

        theta_1 = np.arctan2(y, x)

        c_1, s_1 = np.cos(theta_1), np.sin(theta_1)
        c_3 = ((x - l_1 * c_1) ** 2 + (y - l_1 * s_1) ** 2 + z ** 2 - l_2 ** 2 - l_3 ** 2) / (2 * l_2 * l_3)
        s_3 = -np.sqrt(np.maximum(1 - c_3 ** 2, 0))  # maximum ensures not negative

        theta_2 = np.arctan2(z, (np.sqrt((x - l_1 * c_1) ** 2 + (y - l_1 * s_1) ** 2))) - np.arctan2((l_3 * s_3),
                                                                                                     (l_2 + l_3 * c_3))
        theta_3 = np.arctan2(s_3, c_3)

        c_2, s_2 = np.cos(theta_2), np.sin(theta_2)
        c_23 = np.cos(theta_2 + theta_3)

        with np.errstate(all='ignore'):
            theta_dot_1 = (dy * c_1 - dx * s_1) / (l_1 + l_3 * c_23 + l_2 * c_2)
            theta_dot_2 = (1 / l_2) * (dz * c_2 - dx * c_1 * s_2 - dy * s_1 * s_2 + (c_3 / s_3) * (
                        dz * s_2 + dx * c_1 * c_2 + dy * c_2 * s_1))
            theta_dot_3 = -(1 / l_2) * (
                        dz * c_2 - dx * c_1 * s_2 - dy * s_1 * s_2 + ((l_2 + l_3 * c_3) / (l_3 * s_3)) * (
                            dz * s_2 + dx * c_1 * c_2 + dy * c_2 * s_1))

        theta_dot_1 = np.nan_to_num(theta_dot_1, nan=0.0, posinf=0.0, neginf=0.0)
        theta_dot_2 = np.nan_to_num(theta_dot_2, nan=0.0, posinf=0.0, neginf=0.0)
        theta_dot_3 = np.nan_to_num(theta_dot_3, nan=0.0, posinf=0.0, neginf=0.0)

        joint_angles = np.array([theta_1, theta_2, theta_3])
        joint_speeds = np.array([theta_dot_1, theta_dot_2, theta_dot_3])

        return joint_angles, joint_speeds

    # def __joint_velocities(self, foot_speed, joint_angles):
    # 	l_1, l_2, l_3 = self.l_1, self.l_2, self.l_3

    # 	dx, dy, dz = foot_speed
    # 	theta_1, theta_2, theta_3 = joint_angles

    # 	c_1, s_1 = np.cos(theta_1), np.sin(theta_1)
    # 	c_2, s_2 = np.cos(theta_2), np.sin(theta_2)
    # 	c_3, s_3 = np.cos(theta_3), np.sin(theta_3)
    # 	c_23 = np.cos(theta_2 + theta_3)

    # 	theta_dot_1 = (dy*c_1 - dx*s_1) / (l_1 + l_3*c_23 + l_2*c_2)
    # 	theta_dot_2 = (1/l_2)*(dz*c_2 - dx*c_1*s_2 - dy*s_1*s_2 + (c_3 / s_3)*(dz*s_2 + dx*c_1*c_2 + dy*c_2*s_1))
    # 	theta_dot_3 = -(1/l_2)*(dz*c_2 - dx*c_1*s_2 - dy*s_1*s_2 + ((l_2 + l_3*c_3)/(l_3*s_3))*(dz*s_2 + dx*c_1*c_2 + dy*c_2*s_1))

    # 	joint_speeds = np.array([theta_dot_1, theta_dot_2, theta_dot_3])

    # 	return joint_speeds

    def forward_kinematics(self, joint_angles):
        l_1, l_2, l_3 = self.l_1, self.l_2, self.l_3
        theta_1, theta_2, theta_3 = joint_angles

        # Compute point from joint angles
        x = np.cos(theta_1) * (l_1 + l_3 * np.cos(theta_2 + theta_3) + l_2 * np.cos(theta_2))
        y = np.sin(theta_1) * (l_1 + l_3 * np.cos(theta_2 + theta_3) + l_2 * np.cos(theta_2))
        z = l_3 * np.sin(theta_2 + theta_3) + l_2 * np.sin(theta_2)

        return np.array([x, y, z])

    def IMU_feedback(self, measured_attitude):
        return


# reshapes a 32 length array of floats range 0.0 - 1.0 into the range expected by the controller
def reshape(x):
    x = np.array(x)
    # get body height and velocity
    height = x[0] * 0.2
    velocity = x[1] * 0.5
    leg_params = x[2:].reshape((6, 5))
    # radius, offset, step_height, phase, duty_cycle
    param_min = np.array([0.0, -1.745, 0.01, 0.0, 0.0])
    param_max = np.array([0.3, 1.745, 0.2, 1.0, 1.0])
    # scale and shifted params into the ranges expected by controller
    leg_params = leg_params * (param_max - param_min) + param_min

    return height, velocity, leg_params


if __name__ == '__main__':
    import time

    # Radius, offset, step, phase, duty_cycle
    leg_params = [
        0.1, 0, 0.1, 0.0, 0.5,  # leg 0
        0.1, 0, 0.1, 0.5, 0.5,  # leg 1
        0.1, 0, 0.1, 0.0, 0.5,  # leg 2
        0.1, 0, 0.1, 0.5, 0.5,  # leg 3
        0.1, 0, 0.1, 0.0, 0.5,  # leg 4
        0.1, 0, 0.1, 0.5, 0.5]  # leg 5

    start = time.perf_counter()
    ctrl = Controller(leg_params)
    end = time.perf_counter()

    print((end - start) * 1000)

