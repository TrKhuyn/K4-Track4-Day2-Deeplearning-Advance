# Lab 2 — DeepWeeds

Code hoàn chỉnh nằm trong `code/`; `starter/` được giữ nguyên để test của đề vẫn kiểm tra đúng bộ khung. Dataset đã được chuẩn hóa thành:

```
data/
├── images/                 # 17.509 JPG
└── labels/
    ├── labels.csv
    ├── train_subset0.csv   # 10.501 ảnh
    ├── val_subset0.csv     # 3.501 ảnh
    └── test_subset0.csv    # 3.507 ảnh
```

## Chạy lại

Sinh viên: **Trần Ngọc Khuyến** — MSSV: **2A202602682**.

Notebook cục bộ: [`code/lab_day2_complete.ipynb`](code/lab_day2_complete.ipynb). Bài được chạy và lưu artefact trực tiếp trong repo; không sử dụng link Colab/Kaggle.

Môi trường đã dùng: Python 3.11.9, PyTorch 2.14.1+cu130, torchvision 0.29.1+cu130, timm 1.0.30, CUDA 13.0, NVIDIA GeForce RTX 4060 Laptop GPU.

Seed đã dùng: seed 0 cho các thí nghiệm sàng lọc; seed 0, 1, 2 cho cấu hình chung kết F01 và mốc T00.

Từ thư mục gốc của fork, chuyển vào bài nộp trước khi chạy. Giải nén DeepWeeds vào `data/images/` trong thư mục này và đặt các CSV fold 0 vào `data/labels/` như cấu trúc phía trên.

```powershell
Set-Location submissions/2A202602682_tran_ngoc_khuyen
python -m pip install -r code/requirements.txt
python code/run_experiments.py backbones
python code/run_experiments.py training
python code/run_experiments.py combine
python code/run_inference.py
python code/make_results.py
```

Chỉ sau khi đã chốt cấu hình hoàn toàn trên validation mới chạy test đúng một lần/seed:

```powershell
python code/run_experiments.py final --final-backbone convnext_tiny --confirm-test
python code/finalize_predictions.py
python code/eval.py score --pred "predictions/F01_seed*_test.csv" --test-csv data/labels/test_subset0.csv --labels data/labels/labels.csv --tag F01 --out eval_out
python code/eval.py grade --final "predictions/F01_seed*_test.csv" --baseline "predictions/T00_seed*_test.csv" --uncal "predictions/F01uncal_seed*_test.csv" --final-val "predictions/F01_seed*_val.csv" --latency-p95-ms 50.544245 --latency-method proper --test-csv data/labels/test_subset0.csv --val-csv data/labels/val_subset0.csv --labels data/labels/labels.csv --out eval_out
```

`code/train.py` tự lưu config, split audit, history, checkpoint tốt nhất theo macro-F1 val, logits, prediction CSV và training curve. Không có số liệu thí nghiệm giả trong repo; `results.xlsx` chỉ được tạo từ các run thật. Test chỉ được forward một lần/seed; calibration dùng logits đã lưu.

## Ma trận thí nghiệm

- Backbone: ResNet-50, ResNeXt-50, ConvNeXt-Tiny, DeiT-Small, EfficientNet-B0.
- Training: scratch/frozen/finetune; basic/color/RandAugment; Mixup/CutMix; CE/label smoothing/focal/class-balanced CE; balanced sampler; EMA.
- Inference API: 1-view, horizontal flip TTA, five-crop/multiscale, probability/logit aggregation, ensemble, temperature scaling, Conv-BN fusion, FP32/AMP/FP16 latency.

## Kết quả chung kết

F01 qua ba seed đạt top-1 test `0.9792 ± 0.0009`, macro-F1 `0.9750 ± 0.0011`, balanced accuracy `0.9776 ± 0.0024`, ECE `0.0059 ± 0.0015`. Baseline T00 đạt macro-F1 `0.8136 ± 0.0013`. Xem `report.md`, `results.xlsx` và `eval_out/`.
