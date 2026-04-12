# Dự báo giá bất động sản dựa trên đặc trưng địa lý và tiện ích

<p align="center">
  <em>Dự án học máy: hồi quy giá nhà theo đơn vị diện tích từ vị trí, hạ tầng giao thông và tiện ích lân cận.</em>
</p>

---

## Mục lục

- [Tổng quan](#tổng-quan)
- [Bài toán và dữ liệu](#bài-toán-và-dữ-liệu)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
- [Cài đặt môi trường](#cài-đặt-môi-trường)
- [Chuẩn bị dữ liệu](#chuẩn-bị-dữ-liệu)
- [Hướng dẫn chạy](#hướng-dẫn-chạy)
- [Phương pháp và mô hình](#phương-pháp-và-mô-hình)
- [Kết quả và đánh giá](#kết-quả-và-đánh-giá)
- [Ghi chú kỹ thuật](#ghi-chú-kỹ-thuật)
- [Tài liệu tham khảo](#tài-liệu-tham-khảo)

---

## Tổng quan

Dự án xây dựng các mô hình **hồi quy** để dự đoán **giá nhà trên mỗi đơn vị diện tích** dựa trên:

- **Đặc trưng địa lý**: vĩ độ, kinh độ (vị trí trên bản đồ).
- **Tiện ích & hạ tầng**: khoảng cách đến trạm MRT (tàu điện ngầm) gần nhất, số cửa hàng tiện lợi trong khu vực.

Ứng dụng minh họa quy trình chuẩn trong học máy có giám sát: đọc dữ liệu → tiền xử lý → chia tập huấn luyện/kiểm tra → huấn luyện → đánh giá (MSE, RMSE, R²) và trực quan hóa.

---

## Bài toán và dữ liệu

### Mục tiêu

| Thành phần | Mô tả |
|------------|--------|
| **Biến mục tiêu (Y)** | Giá nhà trên mỗi đơn vị diện tích (`Y house price of unit area`) |
| **Đặc trưng (X)** | Khoảng cách MRT, số cửa hàng tiện lợi, vĩ độ, kinh độ |

### Bộ dữ liệu

Mã nguồn sử dụng file Excel **`raw-data.xlsx`** (định dạng tương thích bộ [**Real Estate Valuation**](https://archive.ics.uci.edu/ml/datasets/Real+estate+valuation+data+set) — dữ liệu định giá BĐS khu vực Đài Loan, thường gặp trong giáo trình ML).

Các cột được mã trong project (theo tên gốc trong file):

| Tên cột trong file | Ý nghĩa |
|--------------------|---------|
| `X3 distance to the nearest MRT station` | Khoảng cách đến trạm MRT gần nhất (m) |
| `X4 number of convenience stores` | Số cửa hàng tiện lợi trong phạm vi gần đó |
| `X5 latitude` | Vĩ độ |
| `X6 longitude` | Kinh độ |
| `Y house price of unit area` | Giá nhà / đơn vị diện tích (biến cần dự đoán) |

*(Có thể có thêm cột như `No`, thứ tự dòng — một số script sẽ bỏ qua nếu cần.)*

---

## Cấu trúc thư mục

```
ML-project_middle/
├── data/
│   └── raw/
│       └── raw-data.xlsx      # Cần tự thêm (không kèm trong repo)
├── notebooks/
│   └── main.ipynb             # Notebook phân tích & thí nghiệm đầy đủ
├── src/
│   ├── main.py                # Linear & Ridge (scikit-learn)
│   └── field-center.py        # Ma trận tương quan + Feature Importance (Random Forest)
├── models/
│   └── train.py               # Hồi quy tuyến tính (Gradient Descent thuần NumPy)
├── results/
│   └── model_evaluation.png   # Ảnh đánh giá (sinh khi chạy train.py)
└── README.md
```

---

## Yêu cầu hệ thống

- **Python** 3.8 trở lên (khuyến nghị 3.10+)
- **Hệ điều hành**: Windows / macOS / Linux

---

## Cài đặt môi trường

### 1. Tạo môi trường ảo (khuyến nghị)

**Windows (PowerShell):**

```powershell
cd D:\projects\ML-project_middle
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
cd /path/to/ML-project_middle
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Cài thư viện

```bash
pip install numpy pandas matplotlib seaborn scikit-learn openpyxl
```

| Gói | Vai trò |
|-----|---------|
| `numpy` | Tính toán số, mảng |
| `pandas` | Đọc/ghi bảng dữ liệu |
| `matplotlib`, `seaborn` | Biểu đồ, histogram, heatmap |
| `scikit-learn` | Chia dữ liệu, chuẩn hóa, Linear/Ridge, Random Forest |
| `openpyxl` | Đọc file `.xlsx` bằng `pandas.read_excel` |

---

## Chuẩn bị dữ liệu

1. Tải bộ dữ liệu **Real Estate Valuation** (UCI) hoặc file Excel tương thích về máy.
2. Đặt file vào đường dẫn:

   `data/raw/raw-data.xlsx`

3. Đảm bảo các cột đặc trưng và nhãn **trùng tên** với phần [Bài toán và dữ liệu](#bài-toán-và-dữ-liệu) (hoặc chỉnh code cho khớp tên cột của bạn).

> **Lưu ý:** Trong repository có thể **chưa có** thư mục `data/` — bạn cần tạo `data/raw/` và copy file Excel vào.

---

## Hướng dẫn chạy

Các script dùng đường dẫn tương đối `../data/raw/raw-data.xlsx`. Trên Python, đường dẫn tương đối là so với **thư mục làm việc hiện tại (cwd)**, không phải vị trí file `.py`. Vì vậy cần **chạy đúng thư mục** như bên dưới.

### Notebook `notebooks/main.ipynb`

1. Kích hoạt môi trường ảo và cài thư viện như trên.
2. Mở Jupyter / VS Code / Cursor, chọn kernel Python đã cài package.
3. Đặt **thư mục làm việc** của notebook là `notebooks/` (mặc định thường đúng khi mở file trong thư mục đó).
4. Chạy lần lượt các ô: import → đọc dữ liệu → tiền xử lý → huấn luyện → đánh giá → biểu đồ.

Notebook chứa lý thuyết, Linear Regression, Ridge, so sánh MSE/RMSE và hình scatter thực tế vs dự đoán.

### Script `src/main.py` (scikit-learn: Linear + Ridge)

Từ thư mục gốc project:

```powershell
cd D:\projects\ML-project_middle\src
python main.py
```

Hoặc một dòng:

```powershell
Set-Location D:\projects\ML-project_middle\src; python main.py
```

In ra hệ số mô hình, bảng so sánh dự đoán, MSE/RMSE, R² và mở cửa sổ đồ thị (scatter Linear vs Ridge).

### Script `src/field-center.py` (EDA + Random Forest)

```powershell
cd D:\projects\ML-project_middle\src
python field-center.py
```

Chức năng:

- Vẽ **heatmap** ma trận tương quan giữa các biến.
- Huấn luyện **Random Forest** để xem **Feature Importance** (mức đóng góp của từng đặc trưng vào giá).

### Script `models/train.py` (Linear Regression + Gradient Descent)

```powershell
cd D:\projects\ML-project_middle\models
python train.py
```

- Huấn luyện lớp `LinearRegression` tự cài (gradient descent) trên dữ liệu đã chuẩn hóa Z-score.
- In MSE, RMSE, R² và trọng số.
- Lưu hình **`results/model_evaluation.png`** (đường cong loss, scatter thực tế vs dự đoán, histogram sai số).

---

## Phương pháp và mô hình

### Giả thuyết mô hình tuyến tính

Quan hệ dạng:

\[
y = w_0 + w_1 x_1 + w_2 x_2 + w_3 x_3 + w_4 x_4
\]

Trong đó \(x_1 \ldots x_4\) lần lượt tương ứng khoảng cách MRT, số cửa hàng tiện lợi, vĩ độ, kinh độ; \(y\) là giá trên đơn vị diện tích.

### Các cách triển khai trong repo

| File | Ý tưởng chính |
|------|----------------|
| `notebooks/main.ipynb`, `src/main.py` | **StandardScaler** → **LinearRegression** và **Ridge** (`alpha=0.1`) của sklearn; đánh giá trên tập test 20%. |
| `models/train.py` | Chuẩn hóa thủ công, thêm bias, **gradient descent** (MSE), epochs cố định. |
| `src/field-center.py` | Thăm dò tương quan + **Random Forest** để xếp hạng đặc trưng. |

### Chỉ số đánh giá

- **MSE** (Mean Squared Error): sai số bình phương trung bình.
- **RMSE** (Root MSE): căn MSE, cùng thứ nguyên với giá.
- **R²**: độ phù hợp của mô hình so với phương sai của nhãn (càng gần 100% càng tốt, tùy bối cảnh).

---

## Kết quả và đánh giá

- Khi chạy pipeline trong notebook hoặc `src/main.py`, bạn nhận được bảng so sánh **thực tế vs Linear vs Ridge** và giá trị MSE/RMSE (kết quả cụ thể phụ thuộc phiên bản thư viện và seed).
- Trong notebook đã ghi nhận ví dụ: Ridge đạt khoảng **RMSE ≈ 7.99** trên tập kiểm tra (tham khảo, không cam kết trùng khớp mọi máy).
- **`results/model_evaluation.png`**: tổng hợp đồ thị đánh giá sau khi chạy `models/train.py`.

---

## Ghi chú kỹ thuật

1. **Đường dẫn dữ liệu**: Luôn chạy script từ `src/` hoặc `models/` như hướng dẫn; nếu chạy `python src/main.py` từ thư mục gốc mà không đổi cwd, đường dẫn `../data/...` có thể **sai**.
2. **Tên file**: Code dùng `raw-data.xlsx`. Nếu file của bạn tên khác, đổi trong code hoặc đổi tên file.
3. **Hiển thị đồ thị**: `matplotlib` có thể mở cửa sổ GUI; trên server không màn hình, có thể cần backend khác (ví dụ `Agg`) hoặc chỉ dùng `savefig` như trong `train.py`.
4. **field-center.py**: Tên file gợi ý phân tích “trung tâm” thống kê / tầm quan trọng biến; không liên quan đến package `field` bên ngoài.

---

## Tài liệu tham khảo

- **UCI Machine Learning Repository**: [Real Estate Valuation Data Set](https://archive.ics.uci.edu/ml/datasets/Real+estate+valuation+data+set)
- **scikit-learn**: [User Guide — Linear Models](https://scikit-learn.org/stable/modules/linear_model.html)

---

<p align="center">
  <strong>Dự án phục vụ mục đích học tập và minh họa quy trình học máy có giám sát.</strong><br/>
  Nếu mở rộng thực tế, nên bổ sung kiểm định chéo, thử nhiều mô hình và xử lý ngoại lệ/outlier theo nghiệp vụ BĐS.
</p>
