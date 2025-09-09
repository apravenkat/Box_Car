import gymnasium as gym
import cv2
import numpy as np
import time
from scipy.interpolate import interp1d

class Controller:
    def __init__(self, steering, throttle):
        self.steering = steering
        self.throttle = throttle
        self.dt = 1/50
        self.prev_speed = [0,0]
        self.errors = 0
        self.F_max = 400
        self.mass = 400
        self.prev_error = 0
        self.car_pos = [0,0]
    def getSteeringAngle(self, car, next_point, heading, speed):
        
        c = np.cos(-heading)
        s = np.sin(-heading)
        v_body_x = c * speed[0] - s * speed[1]
        v_body_y = s * speed[0] + c * speed[1]
        heading = np.arctan2(v_body_y, v_body_x)  - np.pi/2 # Adjust heading to be perpendicular to velocity vector
        if self.steering == 'PurePursuit':
            vel = np.sqrt(speed[0]**2 + speed[1]**2)
            dx = next_point[0] - car[0]
            dy = next_point[1] - car[1]
            ld = 20  # look-ahead distance
            L = 2.5  # wheelbase length  
            target_angle = np.arctan2(dy, dx) + np.pi/2
            alpha = target_angle - heading
            alpha = np.arctan2(np.sin(alpha), np.cos(alpha))  # normalize [-pi, pi]
            delta = np.arctan2(2*L*np.sin(alpha), ld) 
            max_steering_angle = 0.4  # radians
            steering_action = np.clip(delta / max_steering_angle, -1.0, 1.0)
            
        elif self.steering == 'Stanley':
            steering_action = 0
        self.car_pos = car
        return steering_action  
    def getThrottleBrake(self, car, next_point, yaw, max_curvature, speed):
        brake = 0
        throttle = 0
        vel = np.sqrt(speed[0]**2 + speed[1]**2)
        vel_prev = (np.sqrt(self.prev_speed[0]**2 + self.prev_speed[1]**2))
        if self.throttle == 'PID':
            kp = 0.05
            kd = 0.00001  
            ki = 0.0001
            target_speed = 50
            err = target_speed - vel
            self.errors += err * self.dt
            actuation = kp * (target_speed - vel) + kd * (err - self.prev_error)/self.dt  + ki * self.errors
            self.prev_error = err   
            if actuation > 0:
                throttle = np.clip(actuation, 0, 1)
                brake = 0
            else:
                throttle = 0
                brake = np.clip(actuation, 0, 1) 
        
        if self.throttle == 'New': 
            acceleration_x = (np.abs(speed[0]) -np.abs(self.prev_speed[0])) / self.dt
            acceleration_y = (np.abs(speed[1]) - np.abs(self.prev_speed[1])) / self.dt
            
        
        self.prev_speed = speed

        return throttle, brake
    