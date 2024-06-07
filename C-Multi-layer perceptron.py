import os
import pandas as pd
from PIL import Image
import numpy as np
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder


train_csv_path = 'C:/Users/dmmao/Desktop/5100 AI/HW2/data/Combined_Dataset/train_data.csv'
test_csv_path = 'C:/Users/dmmao/Desktop/5100 AI/HW2/data/Combined_Dataset/testing_data.csv'
image_folder = 'C:/Users/dmmao/Desktop/5100 AI/HW2/data/Combined_Dataset/All_img/'


train_df = pd.read_csv(train_csv_path).head(200)
test_df = pd.read_csv(test_csv_path).head(100)


X_train = []
y_train = []
X_test = []
y_test = []


img_size = (64, 64)


for index, row in train_df.iterrows():
    img_path = os.path.join(image_folder, row['Filename'])
    img = Image.open(img_path)
    img = img.resize(img_size)
    img = np.array(img).flatten()
    X_train.append(img)
    y_train.append(row['AQI_Class'])


for index, row in test_df.iterrows():
    img_path = os.path.join(image_folder, row['Filename'])
    img = Image.open(img_path)
    img = img.resize(img_size)
    img = np.array(img).flatten()
    X_test.append(img)
    y_test.append(row['AQI_Class'])


X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


print("train set number:", X_train.shape)
print("test set number:", X_test.shape)


param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (100, 50)],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate_init': [0.001, 0.01, 0.1]
}
kf = KFold(n_splits=5, shuffle=True, random_state=42)
best_score = 0
best_params = None


for hidden_layer_sizes in param_grid['hidden_layer_sizes']:
    for alpha in param_grid['alpha']:
        for learning_rate_init in param_grid['learning_rate_init']:
            fold_scores = []
            for train_indices, val_indices in kf.split(X_train):
                model = MLPClassifier(
                    max_iter=1000,
                    hidden_layer_sizes=hidden_layer_sizes,
                    alpha=alpha,
                    learning_rate_init=learning_rate_init,
                    random_state=42
                )
                model.fit(X_train[train_indices], y_train[train_indices])
                y_val_pred = model.predict(X_train[val_indices])
                fold_score = f1_score(y_train[val_indices], y_val_pred, average='weighted', zero_division=1)
                fold_scores.append(fold_score)

            avg_fold_score = np.mean(fold_scores)
            if avg_fold_score > best_score:
                best_score = avg_fold_score
                best_params = {
                    'hidden_layer_sizes': hidden_layer_sizes,
                    'alpha': alpha,
                    'learning_rate_init': learning_rate_init
                }

print(f"best parameter: {best_params}")
print(f"best f1 score: {best_score}")


best_model = MLPClassifier(
    max_iter=1000,
    hidden_layer_sizes=best_params['hidden_layer_sizes'],
    alpha=best_params['alpha'],
    learning_rate_init=best_params['learning_rate_init'],
    random_state=42
)
best_model.fit(X_train, y_train)


y_pred = best_model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=1)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"accuracy: {accuracy}")
print(f"precision: {precision}")
print(f"recall: {recall}")
print(f"f1: {f1}")
print(f"conf_matrix:\n{conf_matrix}")