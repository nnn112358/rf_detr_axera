import numpy as np
import cv2
#import onnxruntime as ort
import axengine as ort

import argparse
from pathlib import Path

# COCOクラス定義
COCO_CLASSES = [
    None, "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", None, "stop sign", "parking meter", "bench", "bird", "cat", 
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", None, "backpack", 
    "umbrella", None, None, "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", 
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", 
    "tennis racket", "bottle", None, "wine glass", "cup", "fork", "knife", "spoon", "bowl", 
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", 
    "cake", "chair", "couch", "potted plant", "bed", None, "dining table", None, None, "toilet", 
    None, "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", 
    "toaster", "sink", "refrigerator", None, "book", "clock", "vase", "scissors", "teddy bear", 
    "hair drier", "toothbrush"
]

def create_session(model_path):
    """ONNX推論セッションを作成"""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"モデルファイルが見つかりません: {model_path}")
    
    session = ort.InferenceSession(str(model_path))
    input_name = session.get_inputs()[0].name
    return session, input_name

def preprocess_image(image_path, input_size=(448, 448)):
    """画像前処理"""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"画像を読み込めません: {image_path}")
    
    h, w = image.shape[:2]
    
    # リサイズ・正規化・次元変換
    resized = cv2.resize(image, input_size)
    #normalized = resized.astype(np.float32) / 255.0
    #input_tensor = np.transpose(normalized, (2, 0, 1))[np.newaxis, ...]
    input_tensor = np.expand_dims(resized, axis=0).astype(np.uint8)
    return input_tensor, (w, h)

def postprocess_detections(outputs, original_size, conf_threshold=0.6):
    """検出結果の後処理"""
    boxes, logits = outputs[0][0], outputs[1][0]  # [300, 4], [300, 91]
    detections = []
    
    for i in range(len(boxes)):
        scores = logits[i]
        max_score = np.max(scores)
        
        if max_score > conf_threshold:
            class_id = np.argmax(scores)
            cx, cy, w, h = boxes[i]
            
            # center形式 -> corner形式
            x1, y1 = cx - 0.5 * w, cy - 0.5 * h
            x2, y2 = cx + 0.5 * w, cy + 0.5 * h
            
            # 元画像サイズにスケール
            if original_size:
                orig_w, orig_h = original_size
                x1, y1, x2, y2 = [int(coord * scale + 0.5) 
                                 for coord, scale in zip([x1, y1, x2, y2], 
                                                       [orig_w, orig_h, orig_w, orig_h])]
            
            # クラス名取得
            class_name = (COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) 
                        else f"class_{class_id}")
            
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'class_id': int(class_id),
                'class_name': class_name,
                'confidence': float(max_score)
            })
    
    return detections

def predict(session, input_name, image_path, conf_threshold=0.6, input_size=(448, 448)):
    """推論実行"""
    input_tensor, original_size = preprocess_image(image_path, input_size)
    outputs = session.run(None, {input_name: input_tensor})
    return postprocess_detections(outputs, original_size, conf_threshold)

def visualize_detections(image_path, detections, output_path):
    """検出結果の可視化"""
    image = cv2.imread(str(image_path))
    
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        label = f"{det['class_name']}: {det['confidence']:.2f}"
        
        # バウンディングボックスとラベル描画
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.imwrite(str(output_path), image)
    print(f"結果画像を保存: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='RF-DETR ONNX推論')
    parser.add_argument('--model', default='rf-detr-nano_sim_16bit.axmodel', help='ONNXモデルパス')
    parser.add_argument('--image', default='test.jpg', help='入力画像パス')
    parser.add_argument('--conf', type=float, default=0.6, help='信頼度閾値')
    parser.add_argument('--output', default='out.jpg', help='出力画像パス')
    
    args = parser.parse_args()
    
    try:
        # セッション作成
        session, input_name = create_session(args.model)
        
        # 推論実行
        detections = predict(session, input_name, args.image, args.conf)
        
        # 結果表示
        print(f"検出された物体数: {len(detections)}")
        for i, det in enumerate(detections, 1):
            print(f"  {i}: {det['class_name']} (ID: {det['class_id']}), "
                  f"信頼度: {det['confidence']:.3f}, 座標: {det['bbox']}")
        
        # 可視化（常に実行）
        visualize_detections(args.image, detections, args.output)
            
    except Exception as e:
        print(f"エラー: {e}")

if __name__ == "__main__":
    main()