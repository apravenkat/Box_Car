import gymnasium as gym
import cv2
import numpy as np
import time
from scipy.interpolate import interp1d

class Controller:
    def __init__(self, steering, throttle):
        self.steering = steering
        self.throttle = throttle

    def getSteeringAngle(self, car, next_point, yaw):
        dx = next_point[0] - car[0]
        dy = next_point[1] - car[1]
        ld = 10
        L = 2.5
        alpha = np.arctan2(dy,dx) + np.pi/2 
        print(alpha)
        alpha = np.arctan2(np.sin(alpha), np.cos(alpha))
        delta = np.arctan2(2*L*np.sin(alpha), ld) 
        max_steering_angle = 0.4  # radians
        steering_action = np.clip(delta / max_steering_angle, -1.0, 1.0)
        return steering_action
    def getThrottleBrake(self, car, next_point, yaw):

        return 0.01, 0
    