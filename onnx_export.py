#!/usr/bin/env python3
"""RF-DETR Nano ONNX Export Script"""

import os
import argparse
from rfdetr import RFDETRNano

def export_nano_model(weights_path, output_dir, resolution=512):
    """RF-DETR nanoモデルをONNXにエクスポート"""
    
    print(f"RF-DETR Nano ONNXエクスポートを開始...")
    print(f"重みファイル: {weights_path}")
    print(f"出力ディレクトリ: {output_dir}")
    print(f"解像度: {resolution}x{resolution}")
    
    # モデル初期化
    if weights_path:
        model = RFDETRNano(pretrain_weights=weights_path)
    else:
        model = RFDETRNano(resolution=resolution)
    
    # エクスポート実行
    model.export(
        output_dir=output_dir,
        simplify=True,
        opset_version=17,
        batch_size=1,
        shape=(resolution, resolution),
        verbose=True,
        force=True
    )
    
    print(f"✓ エクスポート完了: {output_dir}/inference_model.onnx")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RF-DETR Nano ONNX Export")
    parser.add_argument("--weights", type=str, help="学習済み重みファイルパス")
    parser.add_argument("--output", type=str, default="./export", help="出力ディレクトリ")
    parser.add_argument("--resolution", type=int, default=448, help="入力解像度")
    
    args = parser.parse_args()
    
    # 解像度検証
    if args.resolution % 32 != 0:
        raise ValueError(f"解像度は32で割り切れる必要があります: {args.resolution}")
    
    export_nano_model(args.weights, args.output, args.resolution)


