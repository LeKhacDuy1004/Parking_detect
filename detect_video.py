import cv2
import numpy as np
import onnxruntime as ort
import time
import os
import argparse

# ==================== CẤU HÌNH ====================
ONNX_PATH = 'parking_detect.onnx'         # Đường dẫn tới model ONNX
IMG_SIZE = 480                            # Kích thước ảnh đầu vào của model
CONF_THRESHOLD = 0.4                      # Ngưỡng confidence
IOU_THRESHOLD = 0.45                      # Ngưỡng NMS
CLASS_NAMES = ['car', 'free']             # Các class của model
SCAN_INTERVAL = 3                         # Quét 3 giây 1 lần
# =====================================================

def letterbox(img, new_size=480, color=(114, 114, 114)):
    h, w = img.shape[:2]
    r = min(new_size / h, new_size / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw, dh = new_size - new_unpad[0], new_size - new_unpad[1]
    dw /= 2
    dh /= 2
    resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                 cv2.BORDER_CONSTANT, value=color)
    return padded, r, (dw, dh)

def preprocess(frame, img_size=480):
    img, ratio, (dw, dh) = letterbox(frame, img_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1)
    img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img, ratio, (dw, dh)

def xywh2xyxy(x):
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y

def nms(boxes, scores, iou_threshold):
    indices = cv2.dnn.NMSBoxes(
        boxes.tolist(), scores.tolist(),
        score_threshold=CONF_THRESHOLD, nms_threshold=iou_threshold
    )
    if len(indices) == 0:
        return []
    return indices.flatten()

def postprocess(output, ratio, pad, orig_shape):
    predictions = output[0][0]
    obj_conf = predictions[:, 4]
    mask = obj_conf > CONF_THRESHOLD
    predictions = predictions[mask]

    if len(predictions) == 0:
        return []

    boxes_xywh = predictions[:, :4]
    obj_conf = predictions[:, 4]
    class_scores = predictions[:, 5:]

    class_ids = np.argmax(class_scores, axis=1)
    class_conf = class_scores[np.arange(len(class_scores)), class_ids]
    final_scores = obj_conf * class_conf

    boxes_xyxy = xywh2xyxy(boxes_xywh)

    dw, dh = pad
    boxes_xyxy[:, [0, 2]] -= dw
    boxes_xyxy[:, [1, 3]] -= dh
    boxes_xyxy /= ratio

    h, w = orig_shape[:2]
    boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, w)
    boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, h)

    boxes_for_nms = np.copy(boxes_xyxy)
    boxes_for_nms[:, 2] -= boxes_for_nms[:, 0]
    boxes_for_nms[:, 3] -= boxes_for_nms[:, 1]

    keep = nms(boxes_for_nms, final_scores, IOU_THRESHOLD)

    results = []
    for i in keep:
        x1, y1, x2, y2 = boxes_xyxy[i]
        results.append({
            'box': (int(x1), int(y1), int(x2), int(y2)),
            'class_id': int(class_ids[i]),
            'confidence': float(final_scores[i]),
        })
    return results

def process_video(input_video):
    if not os.path.exists(input_video):
        raise FileNotFoundError(f"Không tìm thấy video: {input_video}")
        
    print("Đang tải model ONNX...")
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 4  # Tối ưu cho Raspberry Pi 4 (4 cores)
    session = ort.InferenceSession(ONNX_PATH, sess_options=sess_options,
                                    providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    print("Model đã sẵn sàng.\n")

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise RuntimeError("Không mở được video đầu vào")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30 # Mặc định nếu OpenCV không lấy được FPS
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Thông tin Video: FPS = {fps:.1f}, Tổng số frames = {total_frames}")
    print(f"Chế độ quét: {SCAN_INTERVAL} giây 1 lần\n")
    
    # Số frame cần nhảy để đạt được interval quét (3 giây)
    frame_step = int(fps * SCAN_INTERVAL)
    current_frame_idx = 0
    
    while current_frame_idx < total_frames:
        # Chuyển tới frame mục tiêu
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            break
            
        current_time_sec = current_frame_idx / fps
        
        # Tiền xử lý
        input_tensor, ratio, pad = preprocess(frame, IMG_SIZE)
        
        # Inference & đo thời gian CPU
        t0 = time.time()
        output = session.run(None, {input_name: input_tensor})
        infer_time = time.time() - t0
        
        # Hậu xử lý
        detections = postprocess(output, ratio, pad, frame.shape)
        
        # Tính toán thống kê
        car_dets = [d for d in detections if CLASS_NAMES[d['class_id']] == 'car']
        free_dets = [d for d in detections if CLASS_NAMES[d['class_id']] == 'free']

        car_count = len(car_dets)
        free_count = len(free_dets)
        total = car_count + free_count
        
        # In báo cáo kết quả
        print("=" * 50)
        print(f"Thời gian video: {current_time_sec:.1f}s")
        print(f"Thời gian xử lý CPU: {infer_time*1000:.1f} ms")
        print(f"Tổng số chỗ đỗ xe  : {total}")
        print(f"Số chỗ đã có xe    : {car_count}")
        print(f"Số chỗ trống       : {free_count}")
        print("=" * 50)

        # Chuyển tới frame tiếp theo cần xử lý
        current_frame_idx += frame_step

    cap.release()
    print("Hoàn tất xử lý video.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Quét video phát hiện chỗ trống bãi xe')
    parser.add_argument('--video', type=str, required=True, help='Đường dẫn tới video cần quét')
    args = parser.parse_args()
    
    process_video(args.video)