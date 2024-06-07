import os
import pandas as pd
from PIL import Image
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


train_csv_path = 'C:/Users/dmmao/Desktop/5100 AI/HW2/data/Combined_Dataset/train_data.csv'
test_csv_path = 'C:/Users/dmmao/Desktop/5100 AI/HW2/data/Combined_Dataset/testing_data.csv'
image_folder = 'C:/Users/dmmao/Desktop/5100 AI/HW2/data/Combined_Dataset/All_img/'


train_df = pd.read_csv(train_csv_path).head(100)
test_df = pd.read_csv(test_csv_path).head(100)


X_train = []
y_train = []
X_test = []
y_test = []


img_size = (16, 16)


for index, row in train_df.iterrows():
    img_path = os.path.join(image_folder, row['Filename'])
    img = Image.open(img_path)
    img = img.resize(img_size)
    img = np.array(img).flatten()
    X_train.append(img)
    y_train.append(row['AQI'])


for index, row in test_df.iterrows():
    img_path = os.path.join(image_folder, row['Filename'])
    img = Image.open(img_path)
    img = img.resize(img_size)
    img = np.array(img).flatten()
    X_test.append(img)
    y_test.append(row['AQI'])


X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


pca = PCA(n_components=100)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)


poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train_pca)
X_test_poly = poly.transform(X_test_pca)


print("Training set samples:", X_train_poly.shape)
print("Testing set samples:", X_test_poly.shape)


model = LinearRegression()


kf = KFold(n_splits=5, shuffle=True, random_state=42)
best_score = float('inf')
best_model = None


for train_indices, val_indices in kf.split(X_train_poly):
    model.fit(X_train_poly[train_indices], y_train[train_indices])
    y_val_pred = model.predict(X_train_poly[val_indices])
    fold_score = mean_squared_error(y_train[val_indices], y_val_pred)

    if fold_score < best_score:
        best_score = fold_score
        best_model = model

print(f"Best cross-validation MSE: {best_score}")


best_model.fit(X_train_poly, y_train)


y_pred = best_model.predict(X_test_poly)


mse = mean_squared_error(y_test, y_pred)
print(f"Test set MSE: {mse}")


plt.scatter(y_test, y_pred)
plt.xlabel("True Values (AQI)")
plt.ylabel("Predicted Values (AQI)")
plt.title("True vs Predicted Values")
plt.show()