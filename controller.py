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
        self.counter = 0
    def getSteeringAngle(self, car, closest_point, heading, speed, centerline):
        
        c = np.cos(-heading)
        s = np.sin(-heading)
        v_body_x = c * speed[0] - s * speed[1]
        v_body_y = s * speed[0] + c * speed[1]
        vel = np.sqrt(speed[0]**2 + speed[1]**2)
        ld = 1*vel  # look-ahead distance
        distances = np.linalg.norm(centerline - car, axis=1)
        closest = np.argmin(np.abs(distances-ld))
        next_point = np.array(centerline[closest])
        dx = next_point[0] - car[0]
        dy = next_point[1] - car[1]
        L = 2.5  # wheelbase length 
        target_angle = np.arctan2(dy, dx) + np.pi/2
        heading = np.arctan2(v_body_y, v_body_x)  - np.pi/2 # Adjust heading to be perpendicular to velocity vector
        max_steering_angle = 0.4

        if self.steering == 'PurePursuit':
            alpha = target_angle - heading
            alpha = np.arctan2(np.sin(alpha), np.cos(alpha))  # normalize [-pi, pi]
            delta = np.arctan2(2*L*np.sin(alpha), ld) 
            
            steering_action = np.clip(delta / max_steering_angle, -1.0, 1.0)
            
        elif self.steering == 'Stanley':
            heading_error = target_angle - heading
            heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))  # normalize [-pi, pi]
            xf = car[0] + L * np.cos(heading) * 0.5
            yf = car[1] + L * np.sin(heading) * 0.5
            path_vec = next_point - closest_point
            front_vec = np.array([xf, yf]) - closest_point
            cte = np.cross(path_vec, front_vec) / (np.linalg.norm(path_vec)+0.1)
            vel = max(vel, 0.1)
            delta = 0.3*heading_error + np.arctan2(0.1* cte, vel)
            delta = np.arctan2(np.sin(delta), np.cos(delta))
            if self.counter == 0:
                steering_action = 0
                self.counter = 1    
            else:

                steering_action = np.clip(delta / max_steering_angle, -1.0, 1.0)
                
        self.car_pos = car
        return steering_action  
    def getThrottleBrake(self, car, heading, centerline, speed):
        brake = 0
        throttle = 0
        c = np.cos(-heading)
        s = np.sin(-heading)
        v_body_x = c * speed[0] - s * speed[1]
        v_body_y = s * speed[0] + c * speed[1]
        acceleration = (speed - self.prev_speed) / self.dt
        acc_x = acceleration[0]*c - acceleration[1]*s
        acc_y = acceleration[0]*s + acceleration[1]*c
        vel = np.sqrt(speed[0]**2 + speed[1]**2)
        if self.throttle == 'PID':
            kp = 0.05
            kd = 0.00001  
            ki = 0.0001
            target_speed = 35
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
        
        if self.throttle == 'MPC': 
            return 0
        self.prev_speed = speed

        return throttle, brake
    