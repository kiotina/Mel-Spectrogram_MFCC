# src/features.py
# 입력 전처리 코드
import numpy as np
import librosa #음성 특징 추출 라이브러리
import tensorflow as tf #딥러닝 라이브러리
from tensorflow.keras.applications.vgg19 import preprocess_input    #VGG19 관련 함수

SR = 16000  #샘플링레이트, 음성을 1초당 16,000개의 숫자로 변환함
DURATION = 3.0  #모든 음성 파일을 3초 길이로 맞춤, 논문에는 고정 길이 방식이 없어 실습용으로 3초 사용
SAMPLES = int(SR * DURATION)    #16,000 X 3초 = 48,000개의 음성 숫자 사용함

N_FFT = 400 #음성을 주파수 성분으로 나눌 때 사용하는 창 크기, 한 번에 분석할 소리 조각의 크기, 몇 개의 음성 숫자를 묶어서 주파수를 분석할 것인지
HOP_LENGTH = 160    #음성 분석 시에 창을 얼마나 자주 옮길지 결정하는 값, 다음 조각으로 얼마나 이동할지
N_MELS = 128    #Mel-spectrogram에서 사용할 주파수 구간 개수, 주파수를 몇 개의 줄로 요약할지
N_MFCC = 100    #MFCC에서 뽑을 음성 특징 개수

IMG_SIZE = 128  #Mel 이미지 크기(128X128), 빠른 실습용. 논문 재현에 가깝게는 224 사용 가능


#음성 파일을 읽고 길이를 3초로 통합해 음성 파형을 숫자로 바꾸는 전처리 함수
def load_audio_fixed(path: str) -> np.ndarray:
    y, _ = librosa.load(path, sr=SR, mono=True) #음성 파일을 읽어와서 샘플링레이트를 16,000으로 맞춤, 스테레오가 아닌 모노

    if len(y) > SAMPLES:    #음성이 3초보다 길면 앞부분 3초만 사용
        y = y[:SAMPLES]
    else:   #음성이 3초보다 짧으면 뒤에 0을 채워서 3초로 만듦(padding)
        y = np.pad(y, (0, SAMPLES - len(y)))

    return y.astype(np.float32) #딥러닝 모델에서 자주 쓰이는 float32 숫자 배열로 반환


#(CNN) 음성을 Mel-Spectrogram 이미지로 변환
def make_mel_image(path: str) -> np.ndarray:
    y = load_audio_fixed(path)  #load_audio_fixed 함수를 통해 음성 파일을 항상 3초 길이의 숫자 배열로 변환함

    #음성을 시간-주파수 이미지인 Mel-spectrogram으로 변환함
    mel = librosa.feature.melspectrogram(
        y=y,    #3초 길이로 맞춘 음성 숫자 배열
        sr=SR,  #음성을 읽을 때 사용한 샘플링레이트(16,000)
        n_fft=N_FFT,    #한 번에 분석할 음성 조각의 크기, 한 번에 몇 개의 음성 숫자를 묶어서 분석할지
        hop_length=HOP_LENGTH,  #분석 창을 옮기는 간격, 다음 분석으로 넘어갈 때 얼마나 이동할지
        n_mels=N_MELS   #Mel 주파수 구간 개수
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)   #Mel 값의 스케일을 사람이 보기 쉬운 dB 단위로 바꿈

    # 0~1 정규화
    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)

    # VGG19는 3채널 입력이 필요하므로 grayscale을 3채널로 복제
    mel_img = mel_norm[..., np.newaxis]
    mel_img = tf.image.resize(mel_img, (IMG_SIZE, IMG_SIZE)).numpy()
    mel_img = np.repeat(mel_img, 3, axis=-1)

    # VGG preprocess_input에 맞추기 위해 0~255 스케일
    #return (mel_img * 255.0).astype(np.float32)
    
    # preprocess_input 수정
    mel_img = (mel_img * 255.0).astype(np.float32)
    mel_img = preprocess_input(mel_img)

    return mel_img  #CNN/VGG19 모델에 넣을 수 있는 Mel-spectrogram 이미지 배열 반환


#(BiLSTM) 음성에서 MFCC 특징을 추출하여 시간 순서 데이터로 변환
def make_mfcc_sequence(path: str) -> np.ndarray:
    y = load_audio_fixed(path)  #load_audio_fixed 함수를 통해 음성 파일을 항상 3초 길이의 숫자 배열로 변환함

    #음성에서 MFCC 특징 추출
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=SR,
        n_mfcc=N_MFCC,  #추출할 MFCC 특징의 개수
        n_fft=N_FFT,    #한 번에 분석할 음성 조각의 크기
        hop_length=HOP_LENGTH   #분석 창을 옮기는 간격
    )

    # 계수별 표준화
    mfcc = (mfcc - mfcc.mean(axis=1, keepdims=True)) / (    #각 MFCC 특징의 평균을 0, 표준편차를 1에 가깝게 맞춰 학습을 안정적으로 만듦
        mfcc.std(axis=1, keepdims=True) + 1e-6
    )

    # LSTM 입력: (time, features)
    return mfcc.T.astype(np.float32)    #LSTM은 시간 순서 데이터를 받기 때문에, 모양을 (시간, 특징) 형태로 바꿔서 반환함