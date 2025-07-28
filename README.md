# RF-DETR ONNX Export and Inference
RF-DETR (Real-time DEtection TRansformer) Nanoモデルの ONNX 変換・推論ツール

<img width="1250" height="833" alt="image" src="https://github.com/user-attachments/assets/0911d0d6-f921-478f-8802-0a4ead0f84fd" />
*画像提供: <a href="https://www.pakutaso.com/20240245033post-50463.html">ぱくたそのフリー素材</a>*

## Features

- RF-DETR Nano モデルの ONNX 形式への変換
- ONNX モデルを使用した高速推論
- COCO データセットクラスに対応した物体検出
- 検出結果の可視化機能

## Installation

```bash
# UV を使用した場合
uv sync

# pip を使用した場合  
pip install -r requirements.txt
```

## Usage

### ONNX モデルのエクスポート

```bash
# デフォルト設定でエクスポート
python onnx_export.py

# カスタム設定でエクスポート
python onnx_export.py --weights path/to/weights.pth --output ./export --resolution 512
```

**オプション:**
- `--weights`: 学習済み重みファイルパス (オプション)
- `--output`: 出力ディレクトリ (デフォルト: `./export`)
- `--resolution`: 入力解像度 (デフォルト: 448, 32の倍数である必要があります)

### ONNX モデルでの推論

```bash
# デフォルト設定で推論
python onnx_infer.py

# カスタム設定で推論
python onnx_infer.py --model rf-detr-nano_sim.onnx --image test.jpg --conf 0.7 --output result.jpg
```

**オプション:**
- `--model`: ONNX モデルファイルパス (デフォルト: `rf-detr-nano_sim.onnx`)
- `--image`: 入力画像パス (デフォルト: `test.jpg`)
- `--conf`: 信頼度閾値 (デフォルト: 0.6)
- `--output`: 出力画像パス (デフォルト: `out.jpg`)

## Dependencies

- `onnx-graphsurgeon>=0.5.8`
- `onnxruntime>=1.22.1`
- `onnxsim>=0.4.36`
- `onnxslim>=0.1.61`
- `opencv-python>=4.11.0.86`
- `rfdetr>=1.2.1`

## Project Structure

```
rf-detr-onnx/
├── onnx_export.py      # ONNX エクスポートスクリプト
├── onnx_infer.py       # ONNX 推論スクリプト
├── export/             # エクスポート済みモデル
├── pyproject.toml      # プロジェクト設定
├── test.jpg           # テスト画像
└── out.jpg            # 推論結果画像
```

## License

MIT License - 詳細は [LICENSE](LICENSE) ファイルをご確認ください。

## Requirements

- Python >= 3.11
- 入力解像度は32で割り切れる値である必要があります
