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

            # Second annotation
            # ann_obj_id = 2  # give a unique id to each object we interact with (it can be any integers)
            # ##! add points, `1` means positive click and `0` means negative click
            # points = np.array([[282, 391]], dtype=np.float32)
            # labels = np.array([2], dtype=np.int32)

            # _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
            #     frame_idx=ann_frame_idx, obj_id=ann_obj_id, points=points, labels=labels
            # )

        else:
            # track object
            out_obj_ids, out_mask_logits = predictor.track(color_frame)
            
            # initialize mask
            all_mask_vis = np.zeros((height, width, 3), dtype=np.uint8)

            # Iterate through each object ID and corresponding mask logits
            for i, obj_id in enumerate(out_obj_ids):
                # Get the mask logits for the current object ID
                mask_binary = (out_mask_logits[i] > 0.0).squeeze().cpu().numpy().astype(np.uint8)

                # Convert the mask to a binary format (0 or 1)
                contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # calculate the bounding box for the largest contour
                    largest_contour = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    
                    # draw the bounding box on the color frame
                    hue = int((obj_id % 10) / 10 * 180) # set hue based on object ID
                    box_color_hsv = np.uint8([[[hue, 255, 255]]])
                    box_color_bgr = cv2.cvtColor(box_color_hsv, cv2.COLOR_HSV2BGR)[0][0].tolist()
                    
                    cv2.rectangle(color_frame, (x, y), (x + w, y + h), box_color_bgr, 2)
                # finish mask processing

                # visualize the mask
                color_mask = np.zeros_like(all_mask_vis)
                color_mask[mask_binary == 1] = box_color_bgr
                all_mask_vis = cv2.add(all_mask_vis, color_mask)

            # overlay the mask on the color frame
            color_frame = cv2.addWeighted(color_frame, 1.0, all_mask_vis, 0.5, 0)
            
        # mark the FPS on the frame
        fps_text = f"FPS: {fps:.2f}"
        cv2.putText(color_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # show the frame
        cv2.imshow("frame", color_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Stop streaming
pipeline.stop()
cv2.destroyAllWindows()