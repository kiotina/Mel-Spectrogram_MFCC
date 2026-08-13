# src/make_dataset.py
import glob #파일 경로
import random   #파일 순서 랜덤하게 섞기 위한 라이브러리
import numpy as np
from tqdm import tqdm   #반복 작업의 진행률을 화면에 보여주는 도구
from sklearn.model_selection import train_test_split    #데이터를 학습용, 검증용, 테스트용으로 나누기 위한 함수

from features import make_mel_image, make_mfcc_sequence #(features.py) 음성 파일을 Mel 이미지와 MFCC 시퀀스로 바꾸는 전처리 함수


#음성 파일에 진짜/가짜 라벨을 붙임
def collect_files(real_dir="data/raw/real", fake_dir="data/raw/fake", subset_per_class=1000):
    real_files = glob.glob(f"{real_dir}/**/*.wav", recursive=True)  #real 폴더 안에 잇는 모든 wav 음성 파일 경로를 찾아옴
    fake_files = glob.glob(f"{fake_dir}/**/*.wav", recursive=True)  #fake 폴더 안에 잇는 모든 wav 음성 파일 경로를 찾아옴

    random.seed(42) #랜덤 섞기를 해도 같은 결과가 나오도록 기준값을 고정함
    random.shuffle(real_files)  #real 음성 파일 목록의 순서 랜덤으로 섞기
    random.shuffle(fake_files)  #fake 음성 파일 목록의 순서 랜덤으로 섞기

    real_files = real_files[:subset_per_class]  #real 음성 파일을 최대 subset_per_class개까지만 사용
    fake_files = fake_files[:subset_per_class]  #fake 음성 파일을 최대 subset_per_class개까지만 사용

    files = real_files + fake_files #real 파일 목록 + fake 파일 목록 = 하나의 파일 목록
    labels = [0] * len(real_files) + [1] * len(fake_files)  #real은 0, fake는 1로 정답 라벨을 붙임

    combined = list(zip(files, labels)) #파일 경로와 정답 라벨을 하나씩 짝지어서 묶어둠
    random.shuffle(combined)    #전체 음성 파일 목록을 한번 더 섞음

    files, labels = zip(*combined)  #섞인 목록에서 파일 경로 목록과 라벨 목록을 다시 분리
    return list(files), np.array(labels, dtype=np.float32)  #파일 경로는 리스트로, 라벨은 float32 배열로 반환


#파일 경로를 실제 모델 입력 배열로 변환
def build_arrays(files, labels):
    X_mel = []  #Mel 이미지 데이터를 담아둘 빈 리스트
    X_mfcc = [] #MFCC 시퀀스 데이터를 담아둘 빈 리스트

    for path in tqdm(files, desc="Extracting features"):    #파일을 하나씩 꺼내면서 음성 특징 추출
        X_mel.append(make_mel_image(path))      #현재 음성 파일을 CNN/VGG19에 넣을 Mel 이미지 형태로 변환해서 저장
        X_mfcc.append(make_mfcc_sequence(path)) #현재 음성 파일을 BiLSTM에 넣을 MFCC 시퀀스 형태로 변환해 저장

    X_mel = np.stack(X_mel)     #여러 개의 Mel 이미지를 하나의 큰 NumPy 배열로 묶어둠
    X_mfcc = np.stack(X_mfcc)   #여러 개의 MFCC 시퀀스를 하나의 큰 NumPy 배열로 묶어둠
    y = np.array(labels, dtype=np.float32)  #정답 라벨을 float32 배열로 변환

    return X_mel, X_mfcc, y #Mel 데이터, MFCC 데이터, 정답 라벨을 함께 반환


#전체 데이터셋을 학습/검증/테스트용으로 나눠서 준비
def load_train_val_test(subset_per_class=1000):
    files, labels = collect_files(subset_per_class=subset_per_class)    #real/fake 음성 파일 경로와 정답 라벨을 모아둠

    #전체 데이터를 학습용 80% 테스트용 20% 비율로 나눠둠
    train_files, test_files, y_train, y_test = train_test_split(
        files,  #전체 파일 경로
        labels, #각 파일의 정답 라벨 목록
        test_size=0.2,  #전체 데이터 중 20%를 테스트 데이터로 사용
        random_state=42,    #데이터를 나눠도 매번 같은 결과가 나오도록 기준값 고정
        stratify=labels #real/fake 비율이 비슷하게 유지되도록 나눠둠
    )

    #학습용으로 남겨둔 데이터 중 일부를 검증용 데이터로 다시 나눠둠
    train_files, val_files, y_train, y_val = train_test_split(
        train_files,    #앞에서 만든 학습용 파일 목록
        y_train,    #학습용 라벨 목록
        test_size=0.2,  #학습 데이터 중 20%를 검증 데이터로 사용
        random_state=42,    #기준값 고정
        stratify=y_train    #비율 유지
    )

    #학습용 파일들을 Mel, MFCC, 라벨(정답)로 변환
    Xmel_train, Xmfcc_train, y_train = build_arrays(train_files, y_train)
    #검증용 파일들을 Mel, MFCC, 라벨로 변환
    Xmel_val, Xmfcc_val, y_val = build_arrays(val_files, y_val)
    #테스트용 파일들을 Mel, MFCC, 라벨로 변환
    Xmel_test, Xmfcc_test, y_test = build_arrays(test_files, y_test)

    #train.py에서 바로 받을 수 있도록 학습/검증/테스트 데이터를 한번에 반환
    return (
        Xmel_train, Xmfcc_train, y_train,
        Xmel_val, Xmfcc_val, y_val,
        Xmel_test, Xmfcc_test, y_test
    )