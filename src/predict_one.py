# src/predict_one.py
import sys
import numpy as np
import tensorflow as tf

from features import make_mel_image, make_mfcc_sequence

if len(sys.argv) != 2:
    print("Usage: python src/predict_one.py path/to/audio.wav")
    sys.exit(1)

audio_path = sys.argv[1]

cnn = tf.keras.models.load_model("models/cnn_mel.keras")
bilstm = tf.keras.models.load_model("models/bilstm_mfcc.keras")

mel = make_mel_image(audio_path)[np.newaxis, ...]
mfcc = make_mfcc_sequence(audio_path)[np.newaxis, ...]

p_cnn = float(cnn.predict(mel)[0][0])
p_lstm = float(bilstm.predict(mfcc)[0][0])
p_final = (p_cnn + p_lstm) / 2.0

label = "deepvoice" if p_final >= 0.5 else "real"

print({
    "cnn_probability": p_cnn,
    "bilstm_probability": p_lstm,
    "ensemble_probability": p_final,
    "label": label
})