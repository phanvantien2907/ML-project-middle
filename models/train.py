import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split

def normalize(X_train, X_test):
    """Chuẩn hoá Z-score: (x - mean) / std  (dùng tham số của tập train)."""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8  # +eps tránh chia 0
    return (X_train - mean) / std, (X_test - mean) / std


def add_bias(X):
    """Thêm cột 1 vào đầu ma trận X để học hệ số w0 (bias)."""
    n = X.shape[0]
    return np.hstack([np.ones((n, 1)), X])  # shape (n, d+1)


# ── Lớp Linear Regression ───────────────────
class LinearRegression:
    """
    Hồi quy tuyến tính huấn luyện bằng Gradient Descent.
    Tham số
    -------
    eta   : learning rate
    epochs: số lần duyệt toàn bộ tập huấn luyện
    """

    def __init__(self, eta=0.1, epochs=5000):
        self.eta = eta
        self.epochs = epochs
        self.w = None  # vector trọng số [w0, w1, ..., wd]
        self.loss_history = []  # lưu MSE sau mỗi epoch để vẽ đồ thị

    def fit(self, X, y):
        n, d = X.shape
        self.w = np.zeros((d, 1))  # khởi tạo w = 0

        for _ in range(self.epochs):
            y_hat = X @ self.w  # dự đoán: (n,1)
            error = y_hat - y.reshape(-1, 1)  # sai số:  (n,1)
            grad = (2 / n) * (X.T @ error)  # gradient: (d,1)
            self.w = self.w - self.eta * grad  # cập nhật trọng số

            mse = float((error ** 2).mean())
            self.loss_history.append(mse)

    def predict(self, X):
        return (X @ self.w).flatten()


# ── Các hàm đánh giá ────────────────────────
def mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true, y_pred):
    return math.sqrt(mse(y_true, y_pred))


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - ss_res / ss_tot


if __name__ == "__main__":
    df = pd.read_excel("../data/raw/raw-data.xlsx")
    X = [
        "X3 distance to the nearest MRT station",
        "X4 number of convenience stores",
        "X5 latitude",
        "X6 longitude",
    ]
    Y = "Y house price of unit area"

    X_raw = df[X].values.astype(float)  # (n, 4)
    y_raw = df[Y].values.astype(float)  # (n,)

    # 2. Chia train / test
    X_tr, X_te, y_tr, y_te = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)

    # 3. Chuẩn hoá
    X_tr_sc, X_te_sc = normalize(X_tr, X_te)

    # 4. Thêm cột bias
    X_tr_b = add_bias(X_tr_sc)  # (n_train, 5)
    X_te_b = add_bias(X_te_sc)  # (n_test,  5)

    # ── 2.3  Huấn luyện ─────────────────────
    LR = LinearRegression(eta=0.05, epochs=2000)
    LR.fit(X_tr_b, y_tr)

    # ── 2.3  Đánh giá ───────────────────────
    y_pred_lr = LR.predict(X_te_b)

    print("=" * 35)
    print(f"{'Chỉ số':<20} {'Linear Model':>10}")
    print("-" * 35)
    print(f"{'MSE':<20} {mse(y_te, y_pred_lr):>10.4f}")
    print(f"{'RMSE':<20} {rmse(y_te, y_pred_lr):>10.4f}")
    print(f"{'R² Score (%)':<20} {r2_score(y_te, y_pred_lr) * 100:>9.2f}%")
    print("=" * 35)

    print("\nTrọng số Linear:", np.round(LR.w.flatten(), 4))

    # ── 2.3  Đồ thị ─────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Đánh giá mô hình hồi quy giá nhà BĐS (Linear Regression)", fontsize=14, fontweight="bold")

    # (a) Đường cong hàm mất mát
    axes[0].plot(LR.loss_history, label="Linear Loss", color="royalblue")
    axes[0].set_title("Hàm mất mát (Loss) qua các epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle=":")

    # (b) Thực tế vs Dự đoán
    axes[1].scatter(y_te, y_pred_lr, alpha=0.6, label="Dự đoán Linear", color="royalblue", s=30)
    diag = [y_te.min(), y_te.max()]
    axes[1].plot(diag, diag, "k--", linewidth=1.5, label="Lý tưởng (y=x)")
    axes[1].set_title("Thực tế vs Dự đoán")
    axes[1].set_xlabel("Giá thực tế")
    axes[1].set_ylabel("Giá dự đoán")
    axes[1].legend()
    axes[1].grid(True, linestyle=":")

    # (c) Sai số tuyệt đối (residuals)
    err_lr = np.abs(y_te - y_pred_lr)
    axes[2].hist(err_lr, bins=20, alpha=0.6, label="Sai số Linear", color="royalblue")
    axes[2].set_title("Phân phối sai số tuyệt đối")
    axes[2].set_xlabel("|Thực tế − Dự đoán|")
    axes[2].set_ylabel("Số mẫu")
    axes[2].legend()
    axes[2].grid(True, linestyle=":")

    plt.tight_layout()
    plt.savefig("../results/model_evaluation.png", dpi=150)
    plt.show()