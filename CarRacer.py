import gymnasium as gym
import cv2
import numpy as np
import time
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from controller import Controller
class CarRacer:
    def __init__(self, render_mode="human"):

        self.env = gym.make("CarRacing-v3", render_mode=render_mode)
        self.observation = None
    
    def reset_env(self):
        self.observation, _ = self.env.reset()
        return self.observation
    
    def step(self, action):
        self.observation, reward, terminated, truncated, _ = self.env.step(action)
        if terminated or truncated:
            print("Episode finished. Resetting environment.")
            self.reset_env()

    def extract_features(self):
        image = cv2.cvtColor(self.observation, cv2.COLOR_BGR2HSV)
        image = image[:82, :,:]
        image = cv2.resize(image, (96*3,96*3))
        lower_green = np.array([35, 40,40])
        upper_green = np.array([85,255,255])
        
        #mask_inv = cv2.resize(mask_inv,(96*3,96*3))
        #contours,_ = cv2.findContours(mask_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        #contour_img = image.copy()
        
        #largest = max(contours, key=cv2.contourArea)
        cx, cy = self.getCarLocation(image)
        car = []
        closest_point = np.array([])
        next_point = np.array([])
        car = np.array([cx,cy])
        raw_env = self.env.unwrapped
        linear_velocity = raw_env.car.hull.linearVelocity
        angle = raw_env.car.hull.angle
        speed = np.sqrt(linear_velocity[0]**2 + linear_velocity[1]**2)
        avg_curvature = None
        centerline = np.array([])
        if cx !=-1 and cy !=-1:
            look_ahead = 25   # how far ahead to look (pixels)
            band_height = 60   # vertical size of the cropped region
            band_width = 150  # horizontal size of the cropped region
            y1 = max(0, cy - look_ahead - band_height)
            y2 = max(0, cy - look_ahead)
            x1 = max(0, cx - band_width // 2)
            x2 = min(image.shape[1], cx + band_width // 2)
            cropped = image[y1:y2, x1:x2]
            
            mask = cv2.inRange(cropped, lower_green, upper_green)
            mask_inv = cv2.bitwise_not(mask)
            contours, _ = cv2.findContours(mask_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                largest = largest + np.array([x1, y1])
                lane_points = largest.squeeze()
                diffs = np.diff(lane_points, axis = 0)
                dists = np.hypot(diffs[:,0], diffs[:,1])
                arc_lengths = np.insert(np.cumsum(dists),0,0)
                fx = interp1d(arc_lengths, lane_points[:,0])
                fy = interp1d(arc_lengths, lane_points[:,1])
                num_samples = 50
                uniform_dists = np.linspace(0, arc_lengths[-1], num_samples)
                sampled = np.vstack((fx(uniform_dists), fy(uniform_dists))).T
                mid = len(sampled) // 2
                left_side = sampled[:mid]
                right_side = sampled[mid:][::-1]
                centerline = (left_side + right_side) / 2.0
                #centerline = centerline[:]
                centerline[:,0] = savgol_filter(centerline[:,0], 9, 3)  # window=9, poly=3
                centerline[:,1] = savgol_filter(centerline[:,1], 9, 3)
                curvature = self.findCurvature(centerline)
                avg_curvature = np.mean(curvature)
                distances = np.linalg.norm(centerline - car, axis=1) + look_ahead
                ld = 1*speed  # look-ahead distance
                closest = np.argmin(np.abs(distances-ld))
                next_point = np.array(centerline[closest])
                closest_point = centerline[np.argmin(np.linalg.norm(centerline - car, axis=1))]
                contour_img = image.copy()
                for i in range(1, len(centerline)):
                    pt1 = (int(centerline[i-1][0]), int(centerline[i-1][1]))
                    pt2 = (int(centerline[i][0]), int(centerline[i][1]))
                    cv2.line(contour_img, pt1, pt2, (0, 255, 0), 2)

                cv2.circle(contour_img, (int(closest_point[0]),int(closest_point[1])), 4, (255,0,0), -1)
                contour_img = cv2.resize(contour_img, (96*3, 96*3))
                cv2.imshow("Next Point ", contour_img)
                cv2.waitKey(1)
        return car, closest_point, angle, centerline, linear_velocity
    def findCurvature(self, points):
        if len(points) < 3:
            return 0.0
        x = points[:, 0]
        y = points[:, 1]
        dx = np.gradient(x)
        dy = np.gradient(y)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        epsilon = 1e-6
        curvature = np.abs(dx * ddy - dy * ddx) / ((dx**2 + dy**2)**1.5 + epsilon)
        curvature = np.nan_to_num(curvature)
        return curvature
    def getCarLocation(self, image):
        #image = cv2.resize(image, (96*3,96*3))
        # Define the lower and upper bounds for red color in RGB
        lower_blue = np.array([100, 150, 50])
        upper_blue = np.array([140, 255, 255])
        # Create the mask for off-white areas
        
        mask = cv2.inRange(image, lower_blue, upper_blue)
        
        cx = 0
        cy = 0
        
        #cv2.imshow("Masked", mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        if contours:
            # Find the largest contour based on area
            largest_contour = max(contours, key=cv2.contourArea)

            # Compute the centroid using image moments
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])  # X-coordinate of centroid
                cy = int(M["m01"] / M["m00"])  # Y-coordinate of centroid

                # Draw the largest contour and its centroid
                cv2.drawContours(image, [largest_contour], -1, (0, 255, 0), 3)
                cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)  # Draw red dot at centroid

                #print(f"Centroid of largest contour: ({cx}, {cy})")
            else:
                #print("Centroid calculation failed (Zero area).")
                cx = -1
                cy = -1
        else:
            #print("No blue contours detected.")
            cx = -1
            cy = -1
        
        cv2.imshow("Location of Car", image)
        cv2.waitKey(1)
        return cx,cy
    def get_actuation(self, c, car, closest_point, heading, centerline, speed):
        if car[0]!=-1 and car[1]!=-1:
            steering_angle = c.getSteeringAngle(car, closest_point, heading, speed, centerline)
            throttle, brake = c.getThrottleBrake(car, heading, centerline, speed) 
        else:
            steering_angle = 0
            throttle = 0
            brake = 0
        return steering_angle, throttle, brake
    def random_play(self, steps=1000):
        self.reset_env()
        lateral_controller = ['Stanley', 'PurePursuit']
        longitudinal_controller = ['PID', 'MPC'] 
        c = Controller(steering=lateral_controller[0], throttle=longitudinal_controller[0])
        #time.sleep(3)
        for _ in range(steps):
            car, closest_point, heading, centerline, speed = self.extract_features()
            steering, throttle, brake = self.get_actuation(c, car, closest_point, heading, centerline, speed)
            action = np.array([steering, throttle, brake])
            self.step(action)
        self.close()
    def close(self):
        self.env.close()


if __name__=="__main__":
    agent = CarRacer()
    agent.random_play(steps=5000000)