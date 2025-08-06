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
            # 객체 추적
            out_obj_ids, out_mask_logits = predictor.track(color_frame)
            
            # 마스크 시각화를 위한 빈 이미지 생성
            all_mask_vis = np.zeros((height, width, 3), dtype=np.uint8)

            # 각 추적된 객체에 대해 반복
            for i, obj_id in enumerate(out_obj_ids):
                # 마스크 생성
                mask_binary = (out_mask_logits[i] > 0.0).squeeze().cpu().numpy().astype(np.uint8)

                # --- 바운딩 박스 계산 및 그리기 시작 ---
                contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # 가장 큰 컨투어에 대한 바운딩 박스 계산
                    largest_contour = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    
                    # 바운딩 박스 그리기 (색상은 마스크 색상과 동일하게)
                    hue = int((obj_id % 10) / 10 * 180) # 객체 ID 기반으로 색상 결정
                    box_color_hsv = np.uint8([[[hue, 255, 255]]])
                    box_color_bgr = cv2.cvtColor(box_color_hsv, cv2.COLOR_HSV2BGR)[0][0].tolist()
                    
                    cv2.rectangle(color_frame, (x, y), (x + w, y + h), box_color_bgr, 2)
                # --- 바운딩 박스 계산 및 그리기 종료 ---

                # 마스크 시각화 (색상 입히기)
                color_mask = np.zeros_like(all_mask_vis)
                color_mask[mask_binary == 1] = box_color_bgr
                all_mask_vis = cv2.add(all_mask_vis, color_mask)

            # 원본 프레임에 마스크 오버레이
            color_frame = cv2.addWeighted(color_frame, 1.0, all_mask_vis, 0.5, 0)
            
        # FPS 텍스트 표시
        fps_text = f"FPS: {fps:.2f}"
        cv2.putText(color_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # 결과 프레임 보여주기
        cv2.imshow("frame", color_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

 # Stop streaming
pipeline.stop()
cv2.destroyAllWindows() # 창을 닫는 코드 추가