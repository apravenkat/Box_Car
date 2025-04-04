import gymnasium as gym
import cv2
import numpy as np
import time
from scipy.interpolate import splprep, splev

class CarRacer:
    def __init__(self, render_mode="human"):

        self.env = gym.make("CarRacing-v3", render_mode=render_mode)
        self.observation = None
    
    def reset_env(self):
        self.observation, _ = self.env.reset()
        return self.observation
    
    def step(self, action):

        self.observation, reward, terminated, truncated, _ = self.env.step(action)
        self.extract_features()
        if terminated or truncated:
            print("Episode finished. Resetting environment.")
            self.reset_env()

    def extract_features(self):
        image = cv2.cvtColor(self.observation, cv2.COLOR_BGR2HSV)
        image = image[:82, :,:]
        image = cv2.resize(image, (96*3,96*3))
        lower_green = np.array([35, 40,40])
        upper_green = np.array([85,255,255])
        mask = cv2.inRange(image, lower_green, upper_green)
        mask_inv = cv2.bitwise_not(mask)
        mask_inv = cv2.resize(mask_inv,(96*3,96*3))
        contours,_ = cv2.findContours(mask_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        contour_img = image.copy()
        print(len(contours))
        cv2.drawContours(contour_img, contours, -1, (0,255,0), 3)
        cv2.imshow("Contours", contour_img)
        cx, cy = self.getCarLocation(image)
        if cx !=-1 and cy !=-1 and contours:
            if len(contours) > 1:
                # Flatten and concatenate all contour points into one array
                lane_points = np.vstack([contour.reshape(-1, 2) for contour in contours])
            else:
                lane_points = contours[0].reshape(-1, 2)

            # Sort points by Y-coordinate
            lane_points = lane_points[np.argsort(lane_points[:, 1])]
            #xs = (np.convolve(lane_points[:,0], np.ones(5)/5, mode='valid'))
            #ys = (np.convolve(lane_points[:,1], np.ones(5)/5, mode='valid'))
            
            for x,y in lane_points:
                cv2.circle(image, (int(x),int(y)), radius=5, color=(0,0,255), thickness=3)
            #cv2.circle(image, (x,y), radius=5, color=(0,0,255), thickness=3)
            cv2.imshow('Lane', image)
            '''centerline_points = np.column_stack((spline[0],spline[1])).astype(np.int32)
                       
            for i in range(len(centerline_points) - 1):
                cv2.line(image, tuple(centerline_points[i]), tuple(centerline_points[i + 1]), (0, 0, 255), 2)
            cv2.imshow('Lane Centerline', image)'''
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return

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
        
        #cv2.circle(image,(cx,cy),5,(0,0,255),-1)
        #cv2.imshow("Location of Car", image)
        return cx,cy
    
    def random_play(self, steps=1000):
        self.reset_env()
        #time.sleep(3)
        for _ in range(steps):
            action = self.env.action_space.sample()
            self.step(action)
        self.close()
    def close(self):
        self.env.close()


if __name__=="__main__":
    agent = CarRacer()
    agent.random_play(steps=5000)