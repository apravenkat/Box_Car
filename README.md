# Simple AV Controls


This project implements an autonomous driving agent for the [`CarRacing-v3`](https://gymnasium.farama.org/environments/box2d/car_racing/) environment from **Gymnasium**.  
It combines **computer vision (OpenCV)**, **path extraction**, and a **Pure Pursuit + PD Controller** to control the car in a racing simulation.  
There aren't a lot of easy to use simulators that teach the basics of vehicle controls (steering, braking and throttle). CARLA kinda sucks if you're working with low powered laptop.
I came across this envronment, that is mostly used for RL, but tweaked it a bit to give all that you need to implement lateral and longitudinal controls. This is great if to play around. The physics in the simulator are quite good. I'm going to keep improving this repo, feel free to drop suggestions.

The system detects the car’s location, extracts the lane, computes the centerline, and applies control actions for steering, throttle, and braking.

---

## 🚀 Features
- **Gymnasium `CarRacing-v3` integration** with continuous control.  
- **Computer Vision pipeline**:
  - HSV color filtering for track segmentation.  
  - Cropping around the car position (both x and y axes).  
  - Contour extraction and lane centerline computation.  
- **Curvature Estimation** for safe path following.  
- **Pure Pursuit Steering Controller** with geometric path tracking.  
- **PD Throttle/Brake Controller** for maintaining target speed.  
- Real-time **OpenCV visualization** of car position, lane contours, and next waypoint.  

---

Dependecies installation

### 🔧 Install with Conda (recommended)

```bash
conda create -n car-racing python=3.10 -y
conda activate car-racing
pip install gymnasium[box2d] opencv-python numpy scipy
