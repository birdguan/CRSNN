import sys

sys.path.append('/usr/lib/python2.7/dist-packages')  # weil ROS nicht mit Anaconda installiert
import rospy

import math
import time
import numpy as np

from std_msgs.msg import Int8MultiArray, Float32, Bool
from geometry_msgs.msg import Transform

from decison_making_unit.parameters import *


class VrepEnvironment():
    def __init__(self):
        self.dvs_sub = rospy.Subscriber('dvsData', Int8MultiArray, self.dvs_callback)
        self.pos_sub = rospy.Subscriber('transformData', Transform, self.pos_callback)
        self.left_pub = rospy.Publisher('leftMotorSpeed', Float32, queue_size=1)
        self.right_pub = rospy.Publisher('rightMotorSpeed', Float32, queue_size=1)
        self.reset_pub = rospy.Publisher('resetRobot', Bool, queue_size=None)
        self.dvs_data = np.array([0, 0])
        self.pos_data = []
        self.distance = 0
        self.steps = 0
        self.v_pre = v_pre
        self.turn_pre = turn_pre
        self.resize_factor = [dvs_resolution[0] // resolution[0],
                              (dvs_resolution[1] - crop_bottom - crop_top) // resolution[1]]
        self.outer = False
        rospy.init_node('dvs_controller')
        self.rate = rospy.Rate(rate)

        # Some values for calculating distance to center of lane
        self.v1 = 1.0
        self.v2 = 5.0
        self.scale = 1.0
        self.c1 = np.array([self.scale * 2.0, self.scale * 4.0])
        self.c2 = np.array([self.scale * 2.0, self.scale * 7.5])
        self.c3 = np.array([self.scale * 5.0, self.scale * 8.0])
        self.c4 = np.array([self.scale * 6.5, self.scale * 8.75])
        self.c5 = np.array([self.scale * 7.0, self.scale * 5.0])

    def get_coordinate(self):
        return self.pos_data

    def dvs_callback(self, msg):
        """
        Store incoming DVS data
        :param msg:
        :return:
        """
        self.dvs_data = msg.data
        return

    def pos_callback(self, msg):
        # Store incoming position data
        self.pos_data = np.array([msg.translation.x, msg.translation.y, time.time()])
        return

    def reset(self):
        """
        Reset sterring wheel model
        :return:
        """
        self.left_pub.publish(0.)
        self.right_pub.publish(0.)
        self.v_pre = v_min
        self.turn_pre = 0.
        self.reset_pub.publish(Bool(False))
        time.sleep(1)
        return np.zeros((resolution[0], resolution[1]), dtype=int), 0.

    def step(self, n_l, n_r):

        self.steps += 1
        t = False  # terminal state

        # Steering wheel model
        m_l = n_l / n_max
        m_r = n_r / n_max
        a = m_l - m_r
        v_cur = - abs(a) * (v_max - v_min) + v_max
        if 6.5 <= self.pos_data[0] < 7.5 and self.pos_data[1] > 8:
            turn_factor1 = 3.0
        elif 3.5 <= self.pos_data[0] <= 5.0 and self.pos_data[1] > 8:
            turn_factor1 = 1.5
        else:
            turn_factor1 = 0.9
        turn_cur = turn_factor1 * a
        if turn_cur > turn_cur_max:
            turn_cur = turn_cur_max
        c = math.sqrt((m_l ** 2 + m_r ** 2) / 2.0)
        self.v_pre = c * v_cur + (1 - c) * self.v_pre
        self.turn_pre = c * turn_cur + (1 - c) * self.turn_pre

        # Publish motor speeds
        self.left_pub.publish(self.v_pre + self.turn_pre)
        self.right_pub.publish(self.v_pre - self.turn_pre)
        self.rate.sleep()

        # Get position and distance
        d, p = self.getDistance(self.pos_data)
        # Set reward signal
        r = d
        self.distance = d
        s = self.getState()
        n = self.steps
        lane = self.outer
        total_length = 4.0 + 0.5 * math.pi * 1.75 + 0.5 + 0.25 * math.pi * 1.25 + 1.5 + 0.25 * math.pi * 0.5 + 3.5\
                       + 0.25 * math.pi * 0.25 + 3.0
        # Terminate episode of robot reaches start position again
        # or reset distance
        if 6.5 <= self.pos_data[0] <= 7.5 and 5.0 < self.pos_data[1] < 9.5:
            if abs(d) > 0.8 or p > total_length:
                positions.append(p)
                p = 0
                self.steps = 0
                t = True
                self.reset()
        else:
            if abs(d) > reset_distance or p > total_length:
                positions.append(p)
                p = 0
                self.steps = 0
                t = True
                self.reset()

        # Return state, distance, position, reward, termination, steps, lane
        return s, d, p, r, t, n, lane

    def getDistance(self, p):
        """
        Get distance and position of the vehicle
        position: total distance
        distance: value between target route and current route
        p[0]: x axis value of the vehicle
        p[1]: y axis value of the vehicle
        np.linalg: linear function model
        np.linalg.norm: get normal distance of point a and b
        :param p:
        :return:
        """
        # segment 1: 0 <= x < 0.5 and y < 4
        if 0 <= p[0] < 0.5 and p[1] < 4.0:
            position = p[1]
            distance = p[0] - 0.25
            return distance, position
        # segment 2: 0 <= x < 2 and 4 <= y <= 6
        elif 0 <= p[0] < 2 and (4 <= p[1] <= 6):
            length = np.linalg.norm(p[:2] - self.c1)
            theta = math.asin((p[1] - 4.0) / length)
            position = 4.0 + theta * 1.75
            distance = 1.75 - length
            return distance, position
        # segment 3: 2 <= x < 4 and 5.5 <= y <7.5
        elif 2 <= p[0] < 4 and 5.5 <= p[1] < 7.5:
            length = np.linalg.norm(p[:2] - self.c2)
            theta = math.asin((p[0] - 2.0) / length)
            position = 4.0 + 0.25 * math.pi * 1.75 + theta * 1.75
            distance = length - 1.75
            return distance, position
        # segment 4: 3.5 <= x <= 4 and 7.5 <= y <= 8
        elif 3.5 <= p[0] <= 4 and 7.5 <= p[1] <= 8:
            position = 4.0 + 0.5 * math.pi * 1.75 + p[0]
            distance = p[0] - 3.75
            return distance, position
        # segment 5: 3.5 <= x <= 5 and 8 <= y <= 9.5
        elif 3.5 <= p[0] <= 5 and 8 <= p[1] <= 9.5:
            length = np.linalg.norm(p[:2] - self.c3)
            theta = math.asin((5.0 - p[0]) / length)
            position = 4.0 + 0.5 * math.pi * 1.75 + 0.5 + theta * 1.25
            distance = 1.25 - length
            return distance, position
        # segment 6: 5 < x <= 6.5 and 9 <= y <= 9.5
        elif 5 <= p[0] <= 6.5 and 9 <= p[1] <= 9.5:
            position = 4.0 + 0.5 * math.pi * 1.75 + 0.5 + 0.25 * math.pi * 1.25 + p[0]
            distance = 9.25 - p[1]
            return distance, position
        # segment 7: 6.5 <= x <= 7 and 8.75 <= y <= 9.5
        elif 6.5 <= p[0] <= 7 and 8.75 <= p[1] <= 9.5:
            length = np.linalg.norm(p[:2] - self.c4)
            theta = math.asin((p[0] - 6.5) / length)
            position = 4.0 + 0.5 * math.pi * 1.75 + 0.5 + 0.25 * math.pi * 1.25 + 1.5 + theta * 0.5
            distance = 0.5 - length - 0.5
            return distance, position
        # segment 8: 6.5 <= x <= 7.5 and 5.25 <= y <= 8.75
        elif 6.5 <= p[0] <= 7.5 and 5.25 <= p[1] <= 8.75:
            position = 4.0 + 0.5 * math.pi * 1.75 + 0.5 + 0.25 * math.pi * 1.25 + 1.5 + 0.25 * math.pi * 0.5 + (
                    8.75 - p[1])
            distance = 7 - p[0]
            return distance, position
        # segment 9: 6.5 <=x <= 7.5 and 5 <= y <= 5.25
        elif 6.5 <= p[0] <= 7.5 and 5 <= p[1] <= 5.25:
            length = np.linalg.norm(p[:2] - self.c5)
            theta = math.asin((7.0 - p[0]) / length)
            position = 4.0 + 0.5 * math.pi * 1.75 + 0.5 + 0.25 * math.pi * 1.25 + 1.5 + 0.25 * math.pi * 0.5 + 3.5 + theta * 0.25
            distance = length - 0.25
            return distance, position
        # segment 10: 6.5 <= x <= 7 and 2 <= y <= 5
        elif 6.5 <= p[0] <= 7 and 2.0 <= p[1] <= 5:
            position = 4.0 + 0.5 * math.pi * 1.75 + 0.5 + 0.25 * math.pi * 1.25 + 1.5 + 0.25 * math.pi * 0.5 + 3.5 \
                       + 0.25 * math.pi * 0.25 + (5.0 - p[1])
            distance = 6.75 - p[0]
            return distance, position
        else:
            distance = 10
            position = 0
            return distance, position

    def getState(self):
        new_state = np.zeros((resolution[0], resolution[1]), dtype=int)
        for i in range(len(self.dvs_data) // 2):
            try:
                if crop_bottom <= self.dvs_data[i * 2 + 1] < (dvs_resolution[1] - crop_top):
                    idx = ((self.dvs_data[i * 2]) // self.resize_factor[0],
                           (self.dvs_data[i * 2 + 1] - crop_bottom) // self.resize_factor[1])
                    new_state[idx] += 1
            except:
                pass
        return new_state
