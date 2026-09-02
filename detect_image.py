import cv2
import numpy as np
import onnxruntime as ort
import time
import argparse
import os

# ==================== CẤU HÌNH ====================
ONNX_PATH = 'best.onnx'
IMG_SIZE = 480                            # PHẢI khớp với img-size lúc export
CONF_THRESHOLD = 0.4
IOU_THRESHOLD = 0.45
CLASS_NAMES = ['car', 'free']             # PHẢI đúng thứ tự trong data.yaml
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


def preprocess(frame, img_size=416):
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


def detect_image(session, input_name, image_path):
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {image_path}")

    input_tensor, ratio, pad = preprocess(frame, IMG_SIZE)

    t0 = time.time()
    output = session.run(None, {input_name: input_tensor})
    infer_time = time.time() - t0

    detections = postprocess(output, ratio, pad, frame.shape)
    return detections, infer_time


def print_report(image_path, detections, infer_time):
    car_dets = [d for d in detections if CLASS_NAMES[d['class_id']] == 'car']
    free_dets = [d for d in detections if CLASS_NAMES[d['class_id']] == 'free']

    car_count = len(car_dets)
    free_count = len(free_dets)
    total = car_count + free_count

    print("=" * 50)
    print(f"Ảnh: {os.path.basename(image_path)}")
    print(f"Thời gian inference: {infer_time*1000:.1f} ms")
    print("-" * 50)
    print(f"Tổng số slot phát hiện được : {total}")
    print(f"Đã có xe (car)               : {car_count}")
    print(f"Chỗ trống (free)             : {free_count}")
    if total > 0:
        occ_rate = car_count / total * 100
        print(f"Tỷ lệ lấp đầy                : {occ_rate:.1f}%")
    print("=" * 50)

    # In chi tiết từng box (tùy chọn, hữu ích khi debug)
    if detections:
        print("\nChi tiết từng vị trí:")
        for i, det in enumerate(detections, 1):
            cls_name = CLASS_NAMES[det['class_id']]
            x1, y1, x2, y2 = det['box']
            print(f"  [{i:02d}] {cls_name:6s}  conf={det['confidence']:.2f}  "
                  f"box=({x1},{y1},{x2},{y2})")
    print()


def main():
    parser = argparse.ArgumentParser(description='Quét ảnh phát hiện chỗ trống bãi xe')
    parser.add_argument('--image', type=str, required=True,
                         help='Đường dẫn ảnh cần quét')
    parser.add_argument('--conf', type=float, default=CONF_THRESHOLD,
                         help='Ngưỡng confidence (mặc định 0.4)')
    args = parser.parse_args()

    global CONF_THRESHOLD
    CONF_THRESHOLD = args.conf

    print("Đang tải model ONNX...")
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 4  # tận dụng 4 core của Pi 4
    session = ort.InferenceSession(ONNX_PATH, sess_options=sess_options,
                                    providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    print("Model đã sẵn sàng.\n")

    detections, infer_time = detect_image(session, input_name, args.image)
    print_report(args.image, detections, infer_time)


if __name__ == '__main__':
    main()