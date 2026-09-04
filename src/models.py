# src/models.py
import tensorflow as tf #딥러닝 모델을 만들고 학습시키는 라이브러리
from tensorflow.keras import layers, models #layers는 모델의 부품, models는 부품을 연결하는 도구
from tensorflow.keras.applications.vgg19 import VGG19   #VGG19는 CNN 모델로, 이미지 학습을 많이 해둔 모델임


# CNN 모델 생성
def build_cnn_mel(input_shape):
    inputs = layers.Input(shape=input_shape)    #모델에 들어올 입력 데이터의 모양을 정함 ex. (128, 128, 3)

    # VGG19 이미지 분석 모델을 불러옴
    base = VGG19(
        include_top=False,  #VGG19의 특징 추출 부분만 사용함
        weights="imagenet", #ImageNet 데이터로 미리 학습된 지식 사용
        input_shape=input_shape #입력 이미지의 크기와 채널 수 지정
    )
    base.trainable = False  #VGG19의 기존 학습 내용을 고정하여 새로 학습 중에 바뀌지 않게 함

    x = base(inputs, training=False)    #Mel 이미지를 VGG19에 통과시켜 특징 추출
    x = layers.Flatten()(x)             #VGG19가 추출한 특징 지도를 일렬의 숫자 배열로 펼침
    x = layers.Dense(512, activation="relu")(x) #펼친 특징 지도를 바탕으로 중요한 패턴을 학습하는 완전연결층
    x = layers.Dropout(0.5)(x)          #Dropout = 학습 중 일부 연결을 랜덤하게 꺼서 과적합을 줄임

    # 논문은 Dense(2)를 사용하지만, 실습에서는 binary sigmoid가 간단함
    outputs = layers.Dense(1, activation="sigmoid")(x)  #sigmoid = 최종 결과를 0 ~ 1 사이 숫자로 출력함

    model = models.Model(inputs, outputs, name="cnn_mel_vgg19") #입력부터 출력까지 연결하여 하나의 모델로 합침
    #모델이 어떻게 학습할지 설정
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), #Adam은 모델이 틀린 정도를 줄이도록 가중치를 조정하는 학습 방법, 점점 덜 틀리도록 조정
        loss="binary_crossentropy", #binary_crossentropy는 정답이 두 종류일 때 자주 쓰는 손실 함수
        metrics=["accuracy"]    #학습 중 정확도를 함께 확인
    )
    return model    #완성된 CNN/VGG19 모델을 반환


# BiLSTM 모델 생성
def build_bilstm_mfcc(input_shape):
    inputs = layers.Input(shape=input_shape)    #모델에 들어올 MFCC 시퀀스 데이터 모양을 정함 ex.(시간, 특징 개수)

    x = layers.Bidirectional(   #BiLSTM은 시간 순서를 앞->뒤, 뒤->앞 양방향으로 읽는 LSTM 모델임
        layers.LSTM(512, return_sequences=False)    #그렇기에 시간 순서가 있는 데이터에서 흐름과 패턴을 학습함
    )(inputs)

    x = layers.Dropout(0.8)(x)  #Dropout = 학습 중 일부 연결을 랜덤하게 꺼서 과적합을 줄임
    outputs = layers.Dense(1, activation="sigmoid")(x)  #sigmoid = 최종 결과를 0 ~ 1 사이 숫자로 출력함

    model = models.Model(inputs, outputs, name="bilstm_mfcc")   #입력부터 출력까지 연결하여 하나의 모델로 합침
    
    #모델이 어떻게 학습할지 설정
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), #Adam은 모델이 틀린 정도를 줄이도록 가중치를 조정하는 학습 방법
        loss="binary_crossentropy", #binary_crossentropy는 정답이 두 종류일 때 자주 쓰는 손실 함수
        metrics=["accuracy"]    #학습 중 정확도를 함께 확인
    )
    return model    #완성된 BiLSTM/MFCC 모델을 반환