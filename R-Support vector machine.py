import os
import pandas as pd
from PIL import Image
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


train_csv_path = 'C:/Users/dmmao/Desktop/5100 AI/HW2/data/Combined_Dataset/train_data.csv'
test_csv_path = 'C:/Users/dmmao/Desktop/5100 AI/HW2/data/Combined_Dataset/testing_data.csv'
image_folder = 'C:/Users/dmmao/Desktop/5100 AI/HW2/data/Combined_Dataset/All_img/'


train_df = pd.read_csv(train_csv_path).head(200)
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


param_grid = {
    'C': [0.1, 1, 10],
    'epsilon': [0.1, 0.2, 0.5],
    'kernel': ['linear', 'rbf']
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(SVR(), param_grid, cv=kf, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)


print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation MSE: {-grid_search.best_score_}")


best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)


y_pred = best_model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
print(f"Test set MSE: {mse}")


plt.scatter(y_test, y_pred)
plt.xlabel("True Values (AQI)")
plt.ylabel("Predicted Values (AQI)")
plt.title("True vs Predicted Values")
plt.show()