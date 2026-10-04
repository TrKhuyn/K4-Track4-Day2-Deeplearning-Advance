# Báo cáo Lab 2 — Backbone, công thức huấn luyện và suy luận trên DeepWeeds

## 1. Tóm tắt

Bài toán là phân loại 17.509 ảnh DeepWeeds thành tám loài cỏ dại và lớp `Negatives`. Tôi dùng đúng fold 0 chính thức, so sánh năm backbone, 11 ablation huấn luyện và tám cấu hình suy luận; mọi lựa chọn được khóa bằng validation trước khi mở test. Cấu hình cuối là ConvNeXt-Tiny pretrained, RandAugment, balanced sampler, train 224, suy luận 256 và temperature scaling. Qua ba seed, cấu hình này đạt **top-1 test 0,9792 ± 0,0009**, **macro-F1 0,9750 ± 0,0011** và ECE **0,0059 ± 0,0015**. Macro-F1 cao hơn baseline ResNet-50 0,1613, lớn hơn std lớn nhất 0,0013. Recall Chinee Apple/Snake Weed đạt 96,2%/95,9%. Latency p95 batch 1 là 50,5 ms trên RTX 4060 Laptop, đạt ngân sách 100 ms. `eval.py grade` tự chấm phần chất lượng model đạt 20/20.

## 2. Dữ liệu và thiết lập

### 2.1 Dataset và kiểm tra split

Sử dụng fold 0 nguyên bản, không lọc hay chia lại: train 10.501, validation 3.501 và test 3.507 ảnh. Kiểm tra tự động cho thấy `train∩val = train∩test = val∩test = 0`, hợp ba tập có đúng 17.509 tên file và không thiếu ảnh trong `data/images`.

| Label | Lớp | Train | Val | Test | Tổng |
|---:|---|---:|---:|---:|---:|
| 0 | Chinee Apple | 675 | 225 | 226 | 1.126 |
| 1 | Lantana | 637 | 213 | 213 | 1.063 |
| 2 | Parkinsonia | 618 | 206 | 207 | 1.031 |
| 3 | Parthenium | 613 | 204 | 205 | 1.022 |
| 4 | Prickly Acacia | 637 | 212 | 213 | 1.062 |
| 5 | Rubber Vine | 605 | 202 | 202 | 1.009 |
| 6 | Siam Weed | 644 | 215 | 215 | 1.074 |
| 7 | Snake Weed | 609 | 203 | 204 | 1.016 |
| 8 | Negatives | 5.463 | 1.821 | 1.822 | 9.106 |

`Negatives` nhiều gấp 9,02 lần Rubber Vine, nên top-1 có thể bị lớp lớn chi phối. Macro-F1 là chỉ số chọn checkpoint; báo cáo thêm top-1, balanced accuracy, ECE và chỉ số từng lớp. Tổng theo CSV thực tế của Chinee Apple là 1.126, lệch một ảnh so với con số 1.125 trong Table 1 được trích ở đề; quá trình thực nghiệm giữ nguyên CSV chính thức.

### 2.2 Thiết lập thực nghiệm

- Phần cứng: NVIDIA GeForce RTX 4060 Laptop GPU, CUDA 13.0.
- Phần mềm: Python 3.11.9, PyTorch 2.14.1+cu130, torchvision 0.29.1+cu130, timm 1.0.30, NumPy 2.4.6, pandas 3.0.6, scikit-learn 1.9.1.
- Seed: 0 ở vòng sàng backbone và ablation; seed được cố định cho Python, NumPy, PyTorch và DataLoader worker.
- Tiền xử lý: train dùng RandomResizedCrop(224) và lật ngang; vòng đầu validation dùng center crop 224; cấu hình cuối được chọn trên validation ở 256 và áp dụng nguyên vẹn sang test; chuẩn hóa ImageNet.
- Công thức nền: pretrained ImageNet, fine-tune toàn bộ, AdamW, LR backbone/head `1e-4/1e-3`, weight decay 0,05 nhưng không áp dụng cho bias/norm, warmup một epoch rồi cosine, CE, batch 64, 12 epoch và AMP.
- Checkpoint: epoch có macro-F1 validation cao nhất; hòa chọn epoch sớm hơn. Test không được tạo ở các run B/T (`save_test_predictions=false`).

Pipeline đã kiểm tra forward, kích thước batch và tính chất focal loss với `γ=0` bằng CE (sai số 0). Logit đều cho CE 2,1972246, khớp `ln(9)=2,1972246`; mô hình overfit một batch nhỏ tới loss 0,0010 sau hai bước. Biểu đồ phân bố, ba ảnh mỗi lớp và ảnh sau augmentation nằm trong `curves/`.

## 3. Kết quả so sánh backbone

| ID | Backbone | Params (M) | GMAC | Epoch tốt | Macro-F1 val | Top-1 val | Train/epoch (s) | Latency p50 (ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| B01 | ResNet-50 | 23,53 | 4,109 | 10 | 0,8079 | 0,8603 | 43,39 | 8,73 |
| B02 | ResNeXt-50 32×4d | 23,00 | 4,257 | 11 | 0,8207 | 0,8632 | 50,24 | 8,86 |
| **B03** | **ConvNeXt-Tiny** | **27,83** | **4,470** | **12** | **0,9658** | **0,9737** | **44,74** | **6,67** |
| B04 | DeiT-Small/16 | 21,67 | 4,250 | 9 | 0,9487 | 0,9634 | 30,75 | 6,11 |
| B05 | EfficientNet-B0 | 4,02 | 0,398 | 10 | 0,8020 | 0,8549 | 24,17 | 13,15 |

ConvNeXt-Tiny đứng đầu rõ rệt, hơn DeiT-Small 0,0171 macro-F1 và hơn ResNet-50 0,1579. DeiT-Small là lựa chọn cân bằng đáng chú ý: chỉ thấp hơn ConvNeXt 0,0171 nhưng nhanh hơn khoảng 31% theo thời gian train/epoch. EfficientNet-B0 giảm khoảng 86% tham số và 90% GMAC so với ConvNeXt, nhưng macro-F1 thấp hơn 0,1638 và latency p50 lại cao nhất. Benchmark (FP32, batch 1, 10 warmup + 100 lượt) xác nhận FLOPs không trực tiếp suy ra latency.

Đường cong cho thấy ConvNeXt vẫn cải thiện đến epoch 12, trong khi DeiT tốt nhất ở epoch 9 và giảm nhẹ sau đó. EfficientNet tốt nhất ở epoch 10 nhưng macro-F1 cuối giảm từ 0,8020 xuống 0,7762, dấu hiệu quá khớp hoặc dao động cuối lịch học. Dựa trên chất lượng validation, ConvNeXt-Tiny được khóa làm ứng viên chung kết; DeiT-Small là ứng viên thời gian thực cần benchmark.

## 4. Kết quả công thức huấn luyện

Các ablation dưới đây chạy trên ResNet-50, mỗi run chỉ đổi một yếu tố so với `T00`.

| ID | Trục | Thay đổi so với T00 | Macro-F1 val | Δ F1 | Top-1 val | Nhận xét |
|---|---|---|---:|---:|---:|---|
| T00 | Nền | Fine-tune, basic aug, CE | 0,8079 | — | 0,8603 | Mốc |
| T01 | A | Train từ scratch | 0,4902 | −0,3177 | 0,6324 | Giảm mạnh |
| T02 | A | Đóng băng backbone | 0,6023 | −0,2057 | 0,7249 | Linear head không đủ |
| T03 | B | Color jitter | 0,7983 | −0,0096 | 0,8518 | Không cải thiện |
| **T04** | **B** | **RandAugment** | **0,8204** | **+0,0124** | **0,8658** | Tốt nhất trong ablation |
| T05 | B | Mixup α=0,4 | 0,7868 | −0,0211 | 0,8409 | Giảm |
| T06 | B | CutMix α=1,0 | 0,7827 | −0,0252 | 0,8412 | Giảm |
| T07 | C | Label smoothing 0,1 | 0,8054 | −0,0026 | 0,8620 | F1 gần mốc, ECE xấu hơn |
| T08 | C | Focal γ=2 | 0,7988 | −0,0091 | 0,8495 | Không cải thiện |
| T09 | C | Class-balanced CE β=0,9999 | 0,7934 | −0,0146 | 0,8201 | Balanced acc tăng nhưng top-1 giảm |
| T10 | D | Balanced sampler | 0,8170 | +0,0090 | 0,8466 | Tăng macro-F1, giảm top-1 |
| T11 | F | EMA decay 0,999 | 0,7084 | −0,0996 | 0,7932 | Decay quá lớn cho 12 epoch |

RandAugment là thay đổi đơn lẻ tốt nhất, nhưng tất cả ablation chỉ có một seed nên chưa thể khẳng định mức tăng 0,0124 vượt nhiễu giữa seed. Balanced sampler tăng macro-F1 và balanced accuracy (0,8489 so với 0,7934 của T00), đồng thời làm top-1 giảm 0,0137; đây là đánh đổi phù hợp với mục tiêu coi trọng các lớp hiếm. Class-balanced CE cũng nâng balanced accuracy lên 0,8485 nhưng macro-F1/top-1 đều thấp hơn mốc.

Scratch và frozen kém rõ rệt, cho thấy transfer learning và fine-tune toàn mạng quan trọng với ngân sách 12 epoch. EMA hiện tại có hại; nguyên nhân hợp lý là decay 0,999 cập nhật quá chậm trong một lịch ngắn, không phải bằng chứng rằng EMA luôn kém. Mixup/CutMix làm train loss cuối cao hơn và không giúp validation ở cấu hình đã thử.

Ablation ban đầu chạy trên ResNet-50. Sau đó, run kết hợp `C01` đưa RandAugment và balanced sampler vào ConvNeXt-Tiny: macro-F1 val tăng từ 0,9658 lên **0,9698** ở 224. Mức tăng 0,0040 nhỏ và mới có một seed nên không đủ để kết luận từng thành phần cộng dồn ổn định; tuy nhiên `C01` là cấu hình validation cao nhất tại thời điểm khóa recipe và được dùng cho vòng suy luận.

## 5. Kết quả suy luận

Tất cả phương pháp được chọn trên validation của `C01` seed 0. Latency dùng batch 1, warmup 10, 100 lượt đo và `cuda.synchronize()` trước/sau.

| ID | Phương pháp | K | Macro-F1 val | Top-1 val | ECE | p95 (ms) |
|---|---|---:|---:|---:|---:|---:|
| I00 | 1-view, 224 | 1 | 0,9698 | 0,9769 | 0,0111 | 15,0 (AMP) |
| I01 | Hflip TTA, gộp xác suất | 2 | 0,9719 | 0,9783 | 0,0107 | 138,7 |
| I02 | Hflip TTA, gộp logit | 2 | 0,9719 | 0,9783 | 0,0117 | 138,7 |
| I03 | Resolution 256 | 1 | **0,9777** | **0,9829** | 0,0072 | 50,5 (FP32) |
| I04 | Temperature scaling 224 | 1 | 0,9698 | 0,9769 | 0,0062 | 15,0 |
| I05 | Ensemble ConvNeXt + DeiT | 2 | 0,9761 | 0,9823 | 0,0163 | chưa đo chung |
| I06 | FP16/AMP deployment | 1 | 0,9698 | 0,9769 | 0,0111 | 11,2 (FP16) |
| I07 | Resolution 256 + TS | 1 | **0,9777** | **0,9829** | **0,0040** | **50,5** |

TTA tăng macro-F1 0,0021 nhưng vượt ngân sách 100 ms trong lần đo cuối; gộp xác suất có ECE tốt hơn gộp logit. Ensemble gần bằng resolution 256 nhưng ECE xấu hơn và cần hai model. Tăng resolution lên 256 đem lại mức tăng lớn nhất mà p95 vẫn dưới 100 ms. Temperature scaling không đổi accuracy/F1 nhưng giảm ECE. Ở batch 1, FP16 nhanh hơn AMP; điều này cho thấy không nên mặc định AMP luôn nhanh nhất.

## 6. Cấu hình tốt nhất và chạy test

Cấu hình được khóa trước test: ConvNeXt-Tiny `in12k_ft_in1k`, RandAugment + balanced sampler, CE, train 224, inference 256, AdamW, LR backbone/head `1e-4/1e-3`, 12 epoch và temperature scaling. Một temperature chung **T=1,298456** được fit bằng logits validation của ba seed rồi áp dụng sang test. Baseline là ResNet-50 `a1_in1k`, basic augmentation, 1-view 224. Mỗi nhóm chạy seed 0/1/2 và test đúng một lần/seed.

| Cấu hình | Top-1 test | Macro-F1 test | Balanced acc. | ECE |
|---|---:|---:|---:|---:|
| T00 baseline | 0,8627 ± 0,0023 | 0,8136 ± 0,0013 | 0,7852 ± 0,0147 | 0,0171 ± 0,0034 |
| **F01 final** | **0,9792 ± 0,0009** | **0,9750 ± 0,0011** | **0,9776 ± 0,0024** | **0,0059 ± 0,0015** |

Final tăng macro-F1 **0,1613**, lớn hơn std lớn hơn của hai nhóm (0,0013). Khoảng cách macro-F1 val/test chỉ 0,0021. Calibration giảm ECE test từ 0,0099 xuống 0,0059 mà không đổi nhãn dự đoán.

Theo lớp, Chinee Apple đạt precision/recall/F1 0,967/0,962/0,965; Snake Weed đạt 0,961/0,959/0,960. Cả hai recall vượt mốc bài báo 88,5%/88,8%. Ma trận nhầm lẫn cộng ba seed nằm tại `curves/F01_confusion_matrix.png`. Các lỗi lớn nhất là Negative → Prickly Acacia (39), Negative → Rubber Vine (19), Snake Weed → Negative (17), Chinee Apple → Negative (14) và Chinee Apple → Snake Weed (11). Điều này phù hợp với khó khăn thị giác: nền tự nhiên có thể chứa hình thái giống loài mục tiêu, còn Chinee Apple và Snake Weed có cấu trúc lá dễ nhầm ở góc/chất lượng ảnh nhất định. Ảnh lỗi minh họa nằm tại `curves/F01_hard_class_errors.png`.

## 7. Kết luận và khuyến nghị

Backbone là yếu tố đóng góp lớn nhất: đổi ResNet-50 sang ConvNeXt-Tiny tăng macro-F1 val 0,1579, lớn hơn mức tăng tốt nhất của một ablation ResNet (+0,0124 từ RandAugment) và mức tăng inference 224→256 (+0,0078). Công thức kết hợp và inference tiếp tục cải thiện, nhưng nhỏ hơn thay đổi backbone.

Với robot có ngân sách 30–100 ms/khung, chọn F01/I07: p95 50,5 ms, top-1 test 97,92% và macro-F1 97,50%. Nếu cần latency thấp hơn nữa, dùng 224 FP16 (p95 11,2 ms) nhưng chấp nhận macro-F1 validation thấp hơn khoảng 0,0078. TTA/ensemble phù hợp ngoại tuyến hơn do tăng chi phí; temperature scaling gần như miễn phí và nên giữ để cải thiện độ tin cậy.

## 8. Hạn chế và việc tiếp theo

- Vòng sàng và ablation đơn lẻ chỉ có một seed; chỉ vòng cuối có ba seed và std.
- Chỉ dùng một fold. Fold được chia ngẫu nhiên, không theo địa điểm, nên test có thể lạc quan khi gặp trang trại, mùa hoặc thiết bị mới.
- Bài lab dùng 12 epoch, khác khoảng 100 epoch và augmentation mạnh của bài báo gốc.
- Latency phụ thuộc trạng thái nguồn, nhiệt và tiến trình nền của laptop; số đo chỉ đại diện máy khai báo.
- Chưa đánh giá miền mới, ảnh theo mùa khác hoặc split theo địa điểm; đây là bước tiếp theo quan trọng.

## 9. Phụ lục: ánh xạ thí nghiệm

- `B01…B05`: ResNet-50, ResNeXt-50, ConvNeXt-Tiny, DeiT-Small, EfficientNet-B0.
- `T00`: công thức nền; `T01/T02`: scratch/frozen; `T03/T04`: color/RandAugment; `T05/T06`: Mixup/CutMix; `T07/T08/T09`: label smoothing/focal/class-balanced CE; `T10`: balanced sampler; `T11`: EMA; `C01`: ConvNeXt + RandAugment + balanced sampler.
- `I00…I07`: các phương pháp suy luận trong Bảng 5; `F01`: cấu hình chung kết ba seed.
- Mỗi run lưu config, history, summary, split audit, checkpoint và logits; prediction nằm ở `predictions/`, curve ở `curves/`, kết quả chấm ở `eval_out/`.
- Notebook chạy lại: `code/lab_day2_complete.ipynb`; hướng dẫn và phiên bản thư viện: `README.md`.
