import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import seaborn as sns
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

read_data = pd.read_excel("../data/raw/raw-data.xlsx")
print(f"5 dữ liệu đầu tiên: {read_data.head()}")
print(read_data.shape)
print(read_data.info())
print(f"Mô tả dữ liệu: {read_data.describe()} ")

X =read_data[[
    "X3 distance to the nearest MRT station",
    "X4 number of convenience stores",
    "X5 latitude",
    "X6 longitude"
]]

Y = read_data["Y house price of unit area"]
X_train, X_test, Y_train, Y_test = train_test_split(
    X,Y, test_size=0.2, random_state=42
)

scalar = StandardScaler()
X_train_scaler = scalar.fit_transform(X_train)
X_test_scaler = scalar.transform(X_test)

regr = linear_model.LinearRegression()
regr.fit(X_train_scaler, Y_train)
print("[w1,.., w_n] =", regr.coef_)
print("[w0 = ]", regr.intercept_)

print("Giá trị đúng", Y_test.iloc[0])

y_pred_linear = regr.predict(X_test_scaler[0:1])
print("Giá trị dự đoán mô hình linear là: ", y_pred_linear)
y_pred_linear_0 = sum(regr.coef_*X_test_scaler[0])+regr.intercept_
print("Giá trị dự đoán mô hình linear theo công thức là: ", y_pred_linear_0)

Y_pred_linear = regr.predict(X_test_scaler)
log_data = pd.DataFrame({
    "Thực tế": Y_test.values,
    "Linear": Y_pred_linear,
    "Lệch Linear": abs(Y_test.values - Y_pred_linear),
})
print(log_data)

mse_linear_model = mean_squared_error(Y_test, Y_pred_linear)
print("Giá trị MSE của mô hình Linear: ", mse_linear_model)
rmse_linear_model = math.sqrt(mean_squared_error(Y_test, Y_pred_linear))
print("Giá trị RMSE của mô hình Linear: ", rmse_linear_model)

draw_values_real_life = sns.histplot(Y_test, kde=True)
render_values_real_life = pd.DataFrame(data=Y_test.values, columns=["values real life"]).describe()
print(render_values_real_life)

draw_values_linear = sns.histplot(Y_pred_linear, kde=True)
render_value_linear = pd.DataFrame(data=Y_pred_linear, columns=["values of linear"]).describe()
print(render_value_linear)
plt.savefig("../results/linear_distribution.png", dpi=150)
plt.show()

print(f"Độ chính xác của mô hình (R2 Score): {r2_score(Y_test, Y_pred_linear)*100:.2f}%")

plt.figure(figsize=(6, 6))
plt.scatter(Y_test, Y_pred_linear, label="Linear", color="royalblue")
plt.plot(
    [Y_test.min(), Y_test.max()],
    [Y_test.min(), Y_test.max()],
    "k--", linewidth=1.5, label="Lý tưởng (y=x)"
)
plt.xlabel("Thực tế")
plt.ylabel("Dự đoán")
plt.title("Dự đoán: Linear Regression")
plt.legend()
plt.savefig("../results/linear_sklearn_scatter.png", dpi=150)
plt.show()
