import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# 1. Load dữ liệu thật từ đường dẫn bạn cung cấp
file_path = '../data/raw/raw-data.xlsx'

try:
    # Đọc file Excel (Lưu ý: Cần cài openpyxl: pip install openpyxl)
    df = pd.read_excel(file_path)

    # 2. Tiền xử lý dữ liệu
    # Thông thường tập dữ liệu này có cột 'No' (số thứ tự), ta cần loại bỏ nó
    if 'No' in df.columns:
        df = df.drop(columns=['No'])

    # Hiển thị 5 dòng đầu để kiểm tra
    print("Dữ liệu đã nạp thành công:")
    print(df.head())

    # 3. Vẽ Ma trận tương quan (Correlation Matrix)
    # Đây là cách nhanh nhất để tìm "cột trung tâm" về mặt thống kê
    plt.figure(figsize=(12, 8))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='RdYlGn', fmt=".2f")
    plt.title("Ma trận tương quan giữa các biến (Dữ liệu thật)")
    plt.show()

    # 4. Dùng máy học (Random Forest) để tìm Feature Importance
    # Xác định biến mục tiêu Y và các biến đặc trưng X
    # (Tìm cột có chữ 'Y' hoặc 'price' trong tên)
    target_col = [col for col in df.columns if 'Y' in col or 'price' in col.lower()][0]
    X = df.drop(columns=[target_col])
    y = df[target_col]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Lấy độ quan trọng của các biến
    importances = pd.Series(model.feature_importances_, index=X.columns)
    importances = importances.sort_values(ascending=True)

    # Vẽ biểu đồ mức độ quan trọng
    plt.figure(figsize=(10, 6))
    importances.plot(kind='barh', color='teal')
    plt.title("Xếp hạng các cột quan trọng nhất (Feature Importance)")
    plt.xlabel("Mức độ ảnh hưởng đến giá nhà")
    plt.tight_layout()
    plt.show()

    print(f"\n==> Cột 'TRUNG TÂM' đóng vai trò quan trọng nhất là: {importances.idxmax()}")

except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file tại '{file_path}'. Hãy kiểm tra lại thư mục nhé!")
except Exception as e:
    print(f"Đã có lỗi xảy ra: {e}")