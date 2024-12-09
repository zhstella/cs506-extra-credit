import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import classification_report, precision_recall_curve
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression

# Load data
data = pd.read_csv("train.csv")  # Replace with your dataset path
test_data = pd.read_csv("test.csv")

# Data preprocessing
def preprocess_dataset(data):
    # Columns to keep and their operations
    label_cols = ['category', 'gender', 'city', 'state', 'cc_num', 'merchant', 'zip', 'job', 'trans_date']
    numeric_cols = ['amt', 'city_pop', 'dob']
    
    # Encode categorical features
    for col in label_cols:
        data[col] = LabelEncoder().fit_transform(data[col].astype(str))
    
    # Convert dob to age
    current_year = pd.Timestamp.now().year
    data['dob'] = current_year - pd.to_datetime(data['dob'], errors='coerce').dt.year
    data['trans_hour'] = data['trans_time'].str.split(':').str[0].astype(int)
    data['delta_lat'] = data['lat'] - data['merch_lat']
    data['delta_long'] = data['long'] - data['merch_long']
    
    # Standardize numerical features
    scaler = StandardScaler()
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
    
    # Drop all other columns
    data = data[label_cols + numeric_cols + ['trans_hour']]
    
    return data

X = preprocess_dataset(data)
test_data_X = preprocess_dataset(test_data)
# import pdb; pdb.set_trace()
# Split the data into features and target
y = data['is_fraud']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)


lgb_model = LGBMClassifier(
    n_estimators=800,
    learning_rate=0.1,
    max_depth=7,
    num_leaves=31,
    objective="binary",
    metric="None",
    random_state=42,
    boosting_type="gbdt",
    
    # early_stopping_rounds=100,
)
xgb_model = XGBClassifier(n_estimators=400, 
                        eta=0.2, 
                        max_depth=8, 
                        tree_method='hist', 
                        device="cpu", 
                        objective='binary:logistic',
                        gamma=0.05,
                        eval_metric='auc',)

# cat_model = CatBoostClassifier(
#     iterations=400,
#     learning_rate=0.1,
#     depth=8,
#     loss_function="Logloss",  # Binary classification
#     eval_metric="F1",         # Use F1 as the evaluation metric
#     class_weights=[1, 3],     # Adjust class weights for imbalance if needed
#     random_seed=42,
#     verbose=100
# )

model = VotingClassifier(estimators=[
    ('xgb', xgb_model),
    ('lgb', lgb_model),
    # ('cat', cat_model)
], voting='soft')


model.fit(
    X_train,
    y_train
)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
f1 = f1_score(y_test, y_pred)
accuracy = (y_test == y_pred).mean()
print(f"F1 Score on Test Set: {f1:.4f}")
print(f"Accuracy on Test Set: {accuracy:.4f}")

# # 使用 predict_proba 获取预测概率
# y_probs = model.predict_proba(X_test)[:, 1]  # 获取属于正类（1）的概率

# # 设置新的阈值
# threshold = 0.95 
# y_pred_adjusted = (y_probs >= threshold).astype(int)

# # 打印分类报告
# print("Classification Report with threshold =", threshold)
# print(classification_report(y_test, y_pred_adjusted))

# # 通过 Precision-Recall Curve 确定最佳阈值
# precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
# f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
# best_threshold = thresholds[np.argmax(f1_scores)]

# print("Best threshold based on F1 score:", best_threshold)

# # 使用最佳阈值重新分类
# y_pred_best = (y_probs >= best_threshold).astype(int)
# best_f1 = f1_score(y_test, y_pred_best)
# print(f"F1 Score with best threshold: {best_f1:.4f}")
# print("Classification Report with best threshold:")
# print(classification_report(y_test, y_pred_best))

# Predict on the test data
y_test_pred = model.predict(test_data_X)

output_df = pd.DataFrame({
    'id': test_data['id'],  # Use the 'id' column from your test dataset
    'is_fraud': y_test_pred   # Predicted values from the model
})

# # Save the DataFrame to a CSV file
output_file = 'predictions.csv'
output_df.to_csv(output_file, index=False)

# print(f"Predictions saved to {output_file}")
# from catboost import CatBoostClassifier, Pool
# from sklearn.metrics import f1_score
# import numpy as np

# # Initialize CatBoost model


# # Train the model with CatBoost's internal Pool
# train_pool = Pool(X_train, y_train)
# valid_pool = Pool(X_test, y_test)

# model.fit(
#     train_pool,
#     eval_set=valid_pool,
#     early_stopping_rounds=50
# )

# # Predict probabilities for the validation set
# y_valid_proba = model.predict_proba(X_test)[:, 1]

# # Optimize the decision threshold for the best F1 Score
# best_threshold = 0.5
# best_f1 = 0

# for threshold in np.arange(0.1, 0.9, 0.01):
#     y_valid_pred = (y_valid_proba > threshold).astype(int)
#     f1 = f1_score(y_test, y_valid_pred)
#     if f1 > best_f1:
#         best_f1 = f1
#         best_threshold = threshold

# print(f"Best Threshold: {best_threshold}, Best F1 Score: {best_f1}")

# # Use the best threshold for predictions on the test set
# y_test_proba = model.predict_proba(test_data_X)[:, 1]
# y_test_pred = (y_test_proba > best_threshold).astype(int)

# # Evaluate F1 Score on the test set
# # f1_test = f1_score(y_test, y_test_pred)
# # print(f"F1 Score on Test Set: {f1_test}")