# Deploy Segment Anything Model 2.0 (SAM2) to Intel RealSense D435

## Demos
<div align=center>
<p align="center">
<img src="./assets/blackswan.gif" width="880">
</p>
</div>

## Getting Started with SAM 2

### SAM2 installation

Please refer to [the installation guide](https://github.com/facebookresearch/sam2/tree/main) provided in the official repository for detailed setup instructions.

### Prepare Intel Realsense SDK python

The Intel RealSense SDK for Python can be installed via [pyrealsense2](https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/readme.md), the official Python package available on PyPI.

### Stream real-time segmentation with Intel Realsense Camera

```python
import torch
from sam2.build_sam import build_sam2_camera_predictor

sam2_checkpoint = "checkpoints/sam2.1_hiera_tiny.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)

import pyrealsense2 as rs
import numpy as np
import cv2
import time

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)

# Start streaming
pipeline.start(config)

if_init = False

prev_time = 0
current_time = 0

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    while True:
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        frames = pipeline.wait_for_frames()
        frame = frames.get_color_frame()
        color_frame = np.asanyarray(frame.get_data())

        width, height = color_frame.shape[:2][::-1]

        if not if_init:
            predictor.load_first_frame(color_frame)
            if_init = True

            ann_frame_idx = 0  # the frame index we interact with

            # First annotation
            ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)
            ##! add points, `1` means positive click and `0` means negative click
            points = np.array([[350, 350]], dtype=np.float32)
            labels = np.array([1], dtype=np.int32)

            _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
                frame_idx=ann_frame_idx, obj_id=ann_obj_id, points=points, labels=labels
            )
            ...
```

## References:

- SAM2 Repository: https://github.com/facebookresearch/sam2
- Real-time sam2 Repository: https://github.com/Gy920/segment-anything-2-real-time
