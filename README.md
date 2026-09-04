# Mel-Spectrogram & MFCC 기반 딥보이스 탐지 실습

음성에서 **Mel-Spectrogram**과 **MFCC** 특징을 각각 추출하고, VGG19 기반 CNN과 BiLSTM의 예측 확률을 소프트 보팅하여 실제 음성과 합성 음성을 분류하는 PBL 논문 실습 프로젝트입니다.

> 참고 논문: [Mel-Spectrogram과 MFCC를 이용한 딥러닝 기반 딥보이스 탐지시스템 개발에 관한 연구](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11516239)

## 1. 실습 목표와 전체 구조

```text
음성 파일
  ├─ Mel-Spectrogram 변환 → VGG19 기반 CNN → 딥보이스 확률 p1
  ├─ MFCC 변환            → BiLSTM        → 딥보이스 확률 p2
  └─ Soft Voting: (p1 + p2) / 2 → 최종 판정
```

두 모델의 평균 확률이 `0.5` 이상이면 `deepvoice`, 미만이면 `real`로 판정합니다.

| 파일 | 역할 |
| --- | --- |
| `src/features.py` | WAV 파일을 3초로 맞추고 Mel 이미지와 MFCC 시퀀스로 변환 |
| `src/make_dataset.py` | real/fake 파일 수집, 라벨 지정, 학습·검증·테스트 분할 |
| `src/models.py` | VGG19 기반 CNN과 BiLSTM 모델 정의 |
| `src/train.py` | 데이터 전처리, 두 모델 학습, 모델과 테스트 배열 저장 |
| `src/evaluate.py` | 저장된 두 모델을 소프트 보팅으로 평가 |
| `src/predict_one.py` | WAV 파일 하나를 입력받아 확률과 최종 라벨 출력 |

## 2. 실습 환경

- 운영체제: Windows, PowerShell 기준
- Python: 3.11.2로 가상환경 생성
- 로컬 실습 환경에서 확인한 주요 패키지 버전
  - TensorFlow 2.21.0
  - librosa 0.11.0
  - NumPy 2.4.5
  - scikit-learn 1.8.0
  - SoundFile 0.13.1
  - tqdm 4.67.3

버전은 실습 당시 로컬 환경을 기록한 것이며, 새 환경에서는 `requirements.txt`를 기준으로 설치합니다.

## 3. 프로젝트 내려받기와 가상환경 구축

### 3.1 저장소 내려받기

```powershell
git clone https://github.com/kiotina/Mel-Spectrogram_MFCC.git
cd Mel-Spectrogram_MFCC
```

### 3.2 가상환경 생성 및 활성화

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

PBL 실습에서는 다음 패키지를 한 번에 설치하는 명령도 사용했습니다. `pandas`, `matplotlib`, `pillow`, `flask`는 현재 학습 코드의 직접 의존성은 아니지만 데이터 분석, 시각화 또는 확장 실습에 사용할 수 있습니다.

```powershell
pip install tensorflow librosa soundfile scikit-learn numpy pandas matplotlib pillow flask tqdm
```

PowerShell 실행 정책 때문에 가상환경 활성화가 차단될 때만 다음 명령을 현재 사용자 범위에서 한 번 실행합니다.

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\.venv\Scripts\Activate.ps1
```

`RemoteSigned`는 로컬 스크립트 실행을 허용하고 인터넷에서 받은 서명되지 않은 스크립트는 제한합니다. 학교나 기관 관리 PC에서는 정책을 임의로 바꾸지 말고 관리자 지침을 따릅니다.

### 3.3 설치 확인

```powershell
python --version
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__); print('GPU:', tf.config.list_physical_devices('GPU'))"
python -c "import librosa, sklearn, numpy; print('librosa:', librosa.__version__); print('scikit-learn:', sklearn.__version__); print('NumPy:', numpy.__version__)"
```

GPU 목록이 비어 있어도 CPU로 실행할 수 있지만 학습 시간이 길어질 수 있습니다.

## 4. 데이터셋 다운로드 및 배치

AIHub 데이터는 용량과 이용 조건 때문에 GitHub 저장소에 포함하지 않습니다. AIHub 로그인, 이용 신청 또는 데이터 사용 동의가 필요할 수 있으며 각 데이터셋의 라이선스와 배포 조건을 따라야 합니다.

### 4.1 실제 음성 데이터 (`real`)

- AIHub: [자유대화 음성(일반남여)](https://www.aihub.or.kr/aihubdata/data/view.do?pageIndex=1&currMenu=115&topMenu=100&srchOptnCnd=OPTNCND001&searchKeyword=%EC%9E%90%EC%9C%A0+%EB%8C%80%ED%99%94+%EC%9D%8C%EC%84%B1&srchDetailCnd=DETAILCND001&srchOrder=ORDER001&srchPagePer=20&aihubDataSe=data&dataSetSn=109)
- AIHub 내부 다운로드 경로: `Training/[원천]2.음성수집도구_5.zip`
- 압축 해제 후 사용할 WAV 파일 위치: `data/raw/real/`

### 4.2 합성 음성 데이터 (`fake`)

- AIHub: [다화자 음성합성 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?pageIndex=1&currMenu=115&topMenu=100&srchOptnCnd=OPTNCND001&searchKeyword=%EB%8B%A4%ED%99%94%EC%9E%90+%EC%9D%8C%EC%84%B1+%ED%95%A9%EC%84%B1+%EB%8D%B0%EC%9D%B4%ED%84%B0&srchDetailCnd=DETAILCND001&srchOrder=ORDER001&srchPagePer=20&aihubDataSe=data&dataSetSn=542)
- AIHub 내부 다운로드 위치: `샘플 데이터/원천 데이터`
- 압축 해제 후 사용할 WAV 파일 위치: `data/raw/fake/`

원본 파일명은 바꾸지 않아도 됩니다. 코드는 하위 폴더를 포함해 `*.wav` 파일을 재귀적으로 찾습니다.

### 4.3 최종 폴더 구조

```text
Mel-Spectrogram_MFCC/
├─ .venv/                       # 로컬 Python 가상환경, Git 제외
├─ data/                        # AIHub 데이터, Git 제외
│  └─ raw/
│     ├─ real/                  # 실제 음성 WAV
│     └─ fake/                  # 합성 음성 WAV
├─ models/                      # 학습 결과, Git 제외
│  ├─ cnn_mel.keras
│  ├─ cnn_mel_final.keras
│  ├─ bilstm_mfcc.keras
│  ├─ bilstm_mfcc_final.keras
│  └─ test_data.npz
├─ src/
│  ├─ features.py
│  ├─ make_dataset.py
│  ├─ models.py
│  ├─ train.py
│  ├─ evaluate.py
│  └─ predict_one.py
├─ .gitignore
├─ README.md
└─ requirements.txt
```

### 4.4 현재 로컬 데이터 기록

2026년 PBL 실습 기록에는 처음 클래스당 100개로 시험한 뒤 약 1,700개씩으로 늘려 실험한 과정이 남아 있습니다. 현재 로컬 폴더에서 확인한 파일 수는 다음과 같습니다.

| 분류 | 폴더 | WAV 파일 수 | 라벨 |
| --- | --- | ---: | ---: |
| 실제 음성 | `data/raw/real/` | 1,819 | 0 |
| 합성 음성 | `data/raw/fake/` | 1,772 | 1 |

파일 수 확인 명령:

```powershell
(Get-ChildItem data\raw\real -Recurse -File -Filter *.wav).Count
(Get-ChildItem data\raw\fake -Recurse -File -Filter *.wav).Count
```

중요: 데이터 폴더에 약 1,700개씩 있어도 현재 `src/train.py`는 `subset_per_class=500`으로 호출하므로 **클래스당 최대 500개**, 총 1,000개만 학습 파이프라인에 사용합니다. 모든 파일을 사용하려면 메모리 용량을 고려하여 이 값을 변경해야 합니다.

## 5. 전처리 설정

`src/features.py`에 정의된 현재 설정입니다.

| 항목 | 값 | 의미 |
| --- | ---: | --- |
| `SR` | 16,000 Hz | 모든 음성을 16 kHz mono로 로드 |
| `DURATION` | 3.0초 | 긴 음성은 앞 3초만 사용하고 짧은 음성은 뒤를 0으로 패딩 |
| `SAMPLES` | 48,000 | 한 파일에서 사용하는 샘플 수 |
| `N_FFT` | 400 | 25 ms 길이의 분석 창 |
| `HOP_LENGTH` | 160 | 10 ms마다 분석 창 이동 |
| `N_MELS` | 128 | Mel 주파수 구간 수 |
| `N_MFCC` | 100 | MFCC 특징 수 |
| `IMG_SIZE` | 128 | Mel 이미지를 128×128로 변환 |

Mel-Spectrogram은 dB 변환과 0~1 정규화 후 3채널로 복제하고, VGG19의 `preprocess_input`을 적용합니다. MFCC는 계수별 평균 0, 표준편차 1에 가깝게 표준화한 뒤 `(시간, 특징)` 형태로 반환합니다.

논문 설정에 더 가깝게 큰 이미지를 실험하려면 `IMG_SIZE=224`를 사용할 수 있지만 계산량과 메모리 사용량이 크게 증가합니다.

## 6. 데이터 분할

`src/make_dataset.py`는 다음 순서로 데이터를 준비합니다.

1. `real`과 `fake` WAV 파일을 찾아 각각 라벨 0과 1 지정
2. `random.seed(42)`로 파일 순서를 섞음
3. 클래스별 `subset_per_class`개 선택
4. 전체의 20%를 테스트 데이터로 분리
5. 남은 80% 중 20%를 검증 데이터로 분리
6. 각 파일에서 Mel 이미지와 MFCC 시퀀스를 모두 생성

현재 클래스당 500개를 사용하므로 예상 분할은 다음과 같습니다.

| 용도 | 클래스별 개수 | 전체 개수 | 전체 비율 |
| --- | ---: | ---: | ---: |
| 학습 | 320 | 640 | 64% |
| 검증 | 80 | 160 | 16% |
| 테스트 | 100 | 200 | 20% |

`random_state=42`와 계층 분할(`stratify`)을 사용하여 재실행 시 분할과 클래스 비율을 유지합니다. TensorFlow의 전체 난수 시드는 별도로 고정하지 않았으므로 최종 학습 결과가 완전히 같지는 않을 수 있습니다.

## 7. 모델 및 학습 설정

### 7.1 Mel-Spectrogram 모델

- ImageNet 사전학습 VGG19의 분류 헤드를 제외하고 특징 추출부 사용
- VGG19 가중치는 동결
- `Flatten → Dense(512, ReLU) → Dropout(0.5) → Dense(1, sigmoid)`

### 7.2 MFCC 모델

- `Bidirectional(LSTM(512)) → Dropout(0.8) → Dense(1, sigmoid)`

### 7.3 공통 학습 설정

| 항목 | 값 |
| --- | ---: |
| Optimizer | Adam |
| Learning rate | `1e-4` |
| Loss | Binary cross-entropy |
| Metric | Accuracy |
| Epochs | 최대 20 |
| Batch size | 16 |
| Early stopping | 검증 성능이 5 epoch 동안 개선되지 않으면 종료 |
| 최종 결합 | 두 모델의 합성 음성 확률 평균 |
| 판정 임계값 | 0.5 |

VGG19 가중치가 로컬 캐시에 없으면 첫 학습 시 ImageNet 가중치를 인터넷에서 내려받습니다.

## 8. 실습 실행 명령

모든 명령은 가상환경을 활성화한 뒤 **프로젝트 루트**에서 실행합니다.

### 8.1 모델 학습

```powershell
python src/train.py
```

학습 과정에서 모든 선택 파일의 Mel과 MFCC 배열을 메모리에 올린 뒤 CNN과 BiLSTM을 순서대로 학습합니다. 완료되면 `models/` 폴더에 모델 4개와 평가용 배열 1개가 생성됩니다.

### 8.2 테스트셋 평가

```powershell
python src/evaluate.py
```

출력 항목:

- Confusion Matrix
- `real`, `deepvoice`별 precision, recall, F1-score
- 전체 Accuracy

평가는 최고 검증 성능으로 저장된 `cnn_mel.keras`, `bilstm_mfcc.keras`와 `test_data.npz`를 사용합니다.

### 8.3 음성 파일 하나 예측

```powershell
python src/predict_one.py "data\raw\test.wav"
```

프로젝트 밖의 WAV 파일도 절대 경로나 상대 경로로 지정할 수 있습니다.

```powershell
python src/predict_one.py "C:\Users\User\Downloads\test.wav"
```

예상 출력 형태:

```text
{
  'cnn_probability': 0.82,
  'bilstm_probability': 0.76,
  'ensemble_probability': 0.79,
  'label': 'deepvoice'
}
```

세 확률은 모두 합성 음성일 확률로 해석합니다. 위 숫자는 출력 형식을 보여주기 위한 예시이며 실제 결과가 아닙니다.

### 8.4 가상환경 종료

```powershell
deactivate
```

## 9. 생성 파일

| 파일 | 설명 |
| --- | --- |
| `models/cnn_mel.keras` | 검증 성능이 가장 좋았던 CNN/VGG19 모델 |
| `models/cnn_mel_final.keras` | CNN 학습 종료 시점의 최종 모델 |
| `models/bilstm_mfcc.keras` | 검증 성능이 가장 좋았던 BiLSTM 모델 |
| `models/bilstm_mfcc_final.keras` | BiLSTM 학습 종료 시점의 최종 모델 |
| `models/test_data.npz` | 두 모델 평가에 사용하는 Mel, MFCC, 정답 배열 |

모델 파일은 각각 수십~수백 MB이며 데이터와 함께 `.gitignore`로 Git 업로드에서 제외합니다.

## 10. 실습 결과와 확인된 한계

PBL 기록상 클래스당 100개로 먼저 시험한 뒤 약 1,700개씩으로 데이터를 늘려 다시 실험했습니다. 단일 음성 예측도 실행했지만, **Naver Clova 합성 음성을 `real`로 오분류한 사례**가 확인되었습니다.

이 결과는 다음 한계를 보여줍니다.

- 특정 AIHub 합성 방식으로 학습한 모델이 학습에 없던 다른 TTS 서비스에 일반화되지 않을 수 있음
- 모든 음성을 앞부분 3초로 자르므로 뒤쪽에 있는 합성 흔적을 놓칠 수 있음
- VGG19는 자연 이미지로 사전학습되어 음성 스펙트로그램에 완전히 최적화된 모델은 아님
- 현재 코드는 선택된 모든 파일의 Mel과 MFCC 배열을 동시에 메모리에 올려 데이터 수를 늘리면 메모리 부족이 발생할 수 있음
- 단순 평균과 고정 임계값 0.5가 항상 최적의 앙상블 방식은 아님

따라서 이 프로젝트는 논문 구조를 이해하고 재현하는 **실습용 구현**이며, 실제 포렌식 판정이나 서비스 운영에 바로 사용해서는 안 됩니다.

## 11. 자주 발생하는 문제

### 가상환경이 활성화되지 않음

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\.venv\Scripts\Activate.ps1
```

### 데이터가 없거나 클래스 폴더가 비어 있음

`data/raw/real`과 `data/raw/fake`에 WAV 파일이 모두 있는지 확인합니다. 한 클래스라도 파일이 없으면 데이터 결합 또는 계층 분할 단계에서 오류가 발생합니다.

```powershell
Get-ChildItem data\raw\real -Recurse -File -Filter *.wav | Select-Object -First 5
Get-ChildItem data\raw\fake -Recurse -File -Filter *.wav | Select-Object -First 5
```

### 평가나 단일 예측에서 모델 파일을 찾지 못함

먼저 `python src/train.py`를 끝까지 실행해 `models/cnn_mel.keras`, `models/bilstm_mfcc.keras`, `models/test_data.npz`가 생성되었는지 확인합니다.

### 메모리 부족

- `src/train.py`의 `subset_per_class`를 줄임
- `batch_size`를 줄임
- `src/features.py`의 `IMG_SIZE`를 128로 유지
- 다른 메모리 사용 프로그램을 종료

### VGG19 가중치 다운로드 오류

첫 실행에는 인터넷 연결이 필요할 수 있습니다. 네트워크를 확인한 뒤 다시 실행하거나, 이미 받은 Keras 가중치 캐시를 사용합니다.

## 12. GitHub에 포함하지 않는 항목

다음 항목은 크기, 재생성 가능성, AIHub 배포 조건 때문에 저장소에 올리지 않습니다.

- `.venv/`: 컴퓨터별 가상환경
- `data/`: AIHub 원본 음성 데이터
- `models/`: 학습 모델과 테스트 배열
- `__pycache__/`, `*.pyc`: Python 캐시

다른 컴퓨터에서 실습하려면 저장소를 clone한 뒤 이 문서의 순서대로 가상환경을 만들고, AIHub 데이터를 다시 내려받아 배치하고, 모델을 새로 학습해야 합니다.
