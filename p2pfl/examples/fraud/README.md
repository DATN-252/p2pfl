# Phát hiện gian lận thẻ tín dụng (Fraud Detection) với P2PFL

Ví dụ này hướng dẫn cách triển khai mô hình học sâu (MLP) để phát hiện các giao dịch gian lận thẻ tín dụng trong môi trường học tập phân tán ngang hàng (P2PFL), sử dụng tập dữ liệu thực tế từ Kaggle.

## 1. Tổng quan
Bài toán phát hiện gian lận đối mặt với thách thức cực lớn về **mất cân bằng lớp** (tỷ lệ gian lận thường < 0.6%). Ví dụ này giải quyết bằng cách:
- **Feature Engineering**: Trích xuất các đặc trưng hành vi người dùng (vận tốc, mức chi tiêu trung bình).
- **Weighted Loss**: Sử dụng `pos_weight` trong hàm mất mát để nhấn mạnh các mẫu gian lận.
- **P2PFL**: Huấn luyện mô hình trên nhiều nút mà không cần tập trung dữ liệu.

## 2. Tiền xử lý dữ liệu

Trước khi chạy thực nghiệm, bạn cần chuẩn bị dữ liệu đã qua xử lý đặc trưng.

### Bước 1: Chạy script tiền xử lý
Script này sẽ tự động tải tập dữ liệu `kartik2112/fraud-detection` từ Kaggle, tính toán các đặc trưng và lưu thành file CSV.

```bash
python p2pfl/examples/fraud/preprocessing_data.py
```

**Các đặc trưng được trích xuất (`transforms.py`):**
- **Khoảng cách**: Khoảng cách Haversine giữa vị trí chủ thẻ và cửa hàng.
- **Thời gian**: Giờ trong ngày và thứ trong tuần.
- **Hành vi (Behavioral)**: 
    - `amt_diff_avg_30d`: Chênh lệch số tiền so với trung bình 30 ngày qua của chính khách hàng đó.
    - `trans_count_24h`: Số lượng giao dịch trong 24 giờ qua.
    - `distance_velocity`: Tốc độ di chuyển ước tính giữa 2 giao dịch liên tiếp (km/h).

Dữ liệu sau xử lý được lưu tại: `p2pfl/examples/fraud/processed_data/`.

### Bước 2: Phân tích và kiểm tra dữ liệu (Tùy chọn)
Để xem biểu đồ phân phối lớp và ma trận tương quan của dữ liệu đã xử lý:

```bash
python p2pfl/examples/fraud/evaluate_data.py
```
Kết quả biểu đồ sẽ nằm trong thư mục `p2pfl/examples/fraud/evaluate_charts/`.

## 3. Cấu hình thực nghiệm (`fraud.yaml`)

Tệp cấu hình định nghĩa môi trường mô phỏng:
- **Mạng lưới**: 10 nút (`nodes: 10`).
- **Dữ liệu**: Trỏ đến các file CSV đã xử lý ở Bước 1.
- **Mô hình**: MLP với `pos_weight: 20.0` để xử lý mất cân bằng.
- **Thuật toán hợp nhất**: `DFedAdp` (Decentralized Federated Adaptive Aggregation).

## 4. Cách chạy thực nghiệm

Sử dụng lệnh sau để bắt đầu quá trình huấn luyện phân tán:

```bash
python -m p2pfl --config p2pfl/examples/fraud/fraud.yaml
```

## 5. Cấu trúc thư mục
- `model/`: Chứa định nghĩa mô hình MLP và các chiến lược cân bằng.
- `processed_data/`: (Tự động tạo) Chứa dữ liệu CSV sau khi Feature Engineering.
- `transforms.py`: Logic xử lý đặc trưng và chuẩn hóa.
- `preprocessing_data.py`: Script tải và xử lý dữ liệu thô.
- `fraud.yaml`: File cấu hình chính cho thực nghiệm.

## 5. Copy file csv vào server

Câu lệnh để có thể copy file từ local vào server

```bash
scp -i "path_to_ssh_key" "path_to_project\p2pfl\p2pfl\examples\fraud\processed_data\*.csv" username@external_ip:/path_in_server
scp -i "D:\College\Semester 8\DoAnTotNghiep\my_ssh_key" "D:\College\Semester 8\DoAnTotNghiep\p2pfl\p2pfl\examples\fraud\processed_data\*.csv" hoap@34.172.33.204:/home/hoap/p2pfl/p2pfl/examples/fraud/processed_data
```