# src/train.py
import os   #운영체제 기능을 사용하기 위한 라이브러리, 이 코드에서는 폴더를 만들 때 사용됨
import tensorflow as tf #딥러닝 모델 학습, 저장, 콜백 기능을 사용하기 위한 라이브러리

from make_dataset import load_train_val_test    #학습용, 검증용, 테스트용 데이터를 한번에 불러오는 함수
from models import build_cnn_mel, build_bilstm_mfcc #Mel 이미지용 CNN 모델과, MFCC 시퀀스용 BiLSTM 모델을 만드는 함수

os.makedirs("models", exist_ok=True)    #학습된 모델을 저장할 models 폴더를 만들고, 이미 있으면 에러 없이 넘어감

#데이터셋을 학습용, 검증용, 테스트용으로 나누어 불러옴
(
    Xmel_train, Xmfcc_train, y_train,   #Mel 이미지, MFCC 시퀀스 형태의 학습 데이터와 정답 라벨
    Xmel_val, Xmfcc_val, y_val,         #Mel 이미지, MFCC 시퀀스 형태의 검증 데이터와 정답 라벨
    Xmel_test, Xmfcc_test, y_test       #Mel 이미지, MFCC 시퀀스 형태의 테스트 데이터와 정답 라벨
) = load_train_val_test(subset_per_class=500)   #각 클래스마다 최대 500개씩 데이터를 불러와서 사용함

print("Mel train:", Xmel_train.shape)   #Mel 학습 데이터의 모양을 출력해서 데이터가 잘 준비됐는지 확인
print("MFCC train:", Xmfcc_train.shape) #MFCC 학습 데이터의 모양을 출력해서 데이터가 잘 준비됐는지 확인

cnn = build_cnn_mel(input_shape=Xmel_train.shape[1:])   #Mel 이미지 데이터의 입력 모양에 맞는 CNN/VGG19 모델을 만듦
bilstm = build_bilstm_mfcc(input_shape=Xmfcc_train.shape[1:])   #MFCC 시퀀스 데이터의 입력 모양에 맞는 BiLSTM 모델을 만듦

#CNN 모델을 학습할 때 사용할 보조 기능들을 모아둔 리스트
callbacks_cnn = [
    #검증 성능이 5번 연속 좋아지지 않으면 학습을 일찍 멈추고, 가장 좋았던 상태로 되돌림
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),

    #학습 중 가장 성능이 좋은 CNN 모델만 models/cnn_mel.keras 파일로 저장
    tf.keras.callbacks.ModelCheckpoint(
        "models/cnn_mel.keras",
        save_best_only=True
    )
]

#BiLSTM 모델을 학습할 때 사용할 보조 기능들을 모아둔 리스트
callbacks_lstm = [
    #검증 성능이 5번 연속 좋아지지 않으면 학습을 일찍 멈추고, 가장 좋았던 상태로 되돌림
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),

    #학습 중 가장 성능이 좋은 CNN 모델만 models/bilstm_mfcc.keras 파일로 저장
    tf.keras.callbacks.ModelCheckpoint(
        "models/bilstm_mfcc.keras",
        save_best_only=True
    )
]


#지금부터 Mel Spectrogram 이미지를 사용하는 CNN 모델을 학습하겠다고 출력
print("\nTraining CNN / Mel-Spectrogram model...")

#CNN 모델을 실제로 학습시킴
cnn.fit(
    Xmel_train, #Mel 이미지 학습 데이터
    y_train,    #학습 데이터의 정답
    validation_data=(Xmel_val, y_val),  #학습 중간중간 성능 확인에 사용할 검증 데이터
    epochs=20,  #전체 학습 데이터를 최대 20번 반복해서 학습
    batch_size=16,  #한번에 16개씩 데이터를 묶어서 학습
    callbacks=callbacks_cnn #보조 기능 리스트(조기 종료, 최고 모델 저장 기능)
)

#지금부터 MFCC 시퀀스를 사용하는 BiLSTM 모델을 학습하겠다고 출력
print("\nTraining BiLSTM / MFCC model...")
bilstm.fit(
    Xmfcc_train,    #MFCC 시퀀스 학습 데이터
    y_train,        #학습 데이터의 정답
    validation_data=(Xmfcc_val, y_val), #학습 중간중간 성능 확인에 사용할 검증 데이터
    epochs=20,  #전체 학습 데이터를 최대 20번 반복해서 학습
    batch_size=16,  #한번에 16개씩 데이터를 묶어서 학습
    callbacks=callbacks_lstm    #보조 기능 리스트(조기 종료, 최고 성능 모델 저장 기능)
)

cnn.save("models/cnn_mel_final.keras") #CNN 모델의 최종 학습 상태를 파일로 저장
bilstm.save("models/bilstm_mfcc_final.keras")   #BiLSTM 모델의 최종 학습 상태를 파일로 저장

# 테스트 데이터도 저장해두면 평가 스크립트에서 사용하기 편함
import numpy as np  #테스트 데이터를 압축 파일 형태로 저장하기 위해 NumPy를 불러옴

#나중에 평가 코드에서 바로 사용할 수 있도록 테스트 데이터를 하나의 npz 파일로 저장
np.savez(
    "models/test_data.npz", #테스트 데이터가 저장될 파일 경로
    Xmel_test=Xmel_test,    #Mel 이미지 테스트 데이터 저장
    Xmfcc_test=Xmfcc_test,  #MFCC 시퀀스 테스트 데이터 저장
    y_test=y_test           #테스트 데이터의 정답 라벨 저장
)