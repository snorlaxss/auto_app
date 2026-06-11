# sam_genpose

A 3D object pose estimation and visualization system combining SAM2 (Segment Anything Model) and GenPose for real-time object detection and 6D pose inference.

## Features

- **Interactive ROI Selection**: Point-based object selection with in-window semantic label input
- **SAM2 Segmentation**: Automatic instance segmentation using Segment Anything Model 2
- **6D Pose Estimation**: Category-agnostic object pose and size inference via GenPose
- **RealSense Integration**: Real-time RGB-D data capture from Intel RealSense cameras
- **3D Visualization**: Real-time rendering of 3D bounding boxes with coordinate axes

## Main Scripts

### `sam_genpose_single_frame.py`
Single-frame pose estimation with interactive workflow:
1. Capture one frame from RealSense camera
2. Select ROIs by clicking points on objects
3. Input semantic labels directly in OpenCV window
4. Run SAM2 segmentation and visualize results
5. Perform GenPose inference and display 3D bounding boxes with labels

**Usage:**
```bash
python sam_genpose_single_frame.py