# Mel-spectrogram & MFCC Deepvoice Detection

음성 파일에서 Mel-spectrogram과 MFCC 특징을 추출하고, VGG19 기반 CNN과 BiLSTM의 예측을 결합해 실제 음성과 합성 음성을 분류하는 프로젝트입니다.

## 구성

- `src/features.py`: 오디오 로드, Mel-spectrogram 및 MFCC 특징 추출
- `src/models.py`: VGG19 기반 CNN과 BiLSTM 모델 정의
- `src/make_dataset.py`: 데이터 수집, 분할 및 배열 생성
- `src/train.py`: 두 모델 학습 및 결과 저장
- `src/evaluate.py`: 앙상블 모델 평가
- `src/predict_one.py`: 단일 WAV 파일 예측

## 설치

Python 3.11 환경을 권장합니다.

```bash
python -m venv .venv
pip install -r requirements.txt
```

## 데이터 구조

데이터셋은 용량과 배포 권한 문제로 저장소에 포함하지 않습니다. 다음 구조로 WAV 파일을 준비하세요.

```text
data/
  raw/
    real/
      *.wav
    fake/
      *.wav
```

## 실행

프로젝트 루트에서 다음 명령을 실행합니다.

```bash
python src/train.py
python src/evaluate.py
python src/predict_one.py path/to/audio.wav
```

학습된 모델과 평가용 배열은 `models/`에 생성되며 저장소에는 포함되지 않습니다.
