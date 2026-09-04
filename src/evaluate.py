# src/evaluate.py
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

cnn = tf.keras.models.load_model("models/cnn_mel.keras")
bilstm = tf.keras.models.load_model("models/bilstm_mfcc.keras")

data = np.load("models/test_data.npz")
Xmel_test = data["Xmel_test"]
Xmfcc_test = data["Xmfcc_test"]
y_test = data["y_test"]

p_cnn = cnn.predict(Xmel_test).reshape(-1)
p_lstm = bilstm.predict(Xmfcc_test).reshape(-1)

p_ensemble = (p_cnn + p_lstm) / 2.0
y_pred = (p_ensemble >= 0.5).astype(int)

print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report")
print(classification_report(
    y_test,
    y_pred,
    target_names=["real", "deepvoice"]
))

print("Accuracy:", accuracy_score(y_test, y_pred))