# DiT-TI: Diffusion Transformer Textual Inversion for FSCIL

FLUX.1-dev 모델을 사용한 CUB-200 Few-Shot Class-Incremental Learning (FSCIL) Textual Inversion 학습 코드입니다.

## 환경 설정

### 방법 1: Conda 환경 생성 (권장)

```bash
# 환경 파일로부터 생성
conda env create -f environment.yml
conda activate diffusion-fscil-dit
```

### 방법 2: 수동 설치

```bash
# Conda 환경 생성
conda create -n diffusion-fscil-dit python=3.10 -y
conda activate diffusion-fscil-dit

# PyTorch 설치 (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# diffusers 최신 버전 설치 (source에서)
pip install git+https://github.com/huggingface/diffusers.git

# 나머지 패키지 설치
pip install transformers accelerate safetensors sentencepiece protobuf peft
```

## 데이터셋 준비

CUB-200-2011 데이터셋을 다음 경로에 준비해주세요:

```
datasets/cub-200-2011/CUB_200_2011/images/
├── 001.Black_footed_Albatross/
├── 002.Laysan_Albatross/
...
└── 200.Common_Yellowthroat/
```

## 사용법

### 기본 실행 (클래스 65-96, GPU 0-3, GPU당 8개 클래스)

```bash
bash run_ti_fscil.sh
```

### 커스텀 설정으로 실행

```bash
bash run_ti_fscil.sh [START_CLASS] [END_CLASS] [GPUS] [CLASSES_PER_GPU]
```

**예시:**

```bash
# 클래스 65-96, GPU 4개 사용 (0,1,2,3), GPU당 8개씩
bash run_ti_fscil.sh 65 96 "0,1,2,3" 8

# 클래스 1-40, GPU 4개 사용, GPU당 10개씩
bash run_ti_fscil.sh 1 40 "0,1,2,3" 10

# 클래스 101-132 (Incremental), GPU 2개만 사용
bash run_ti_fscil.sh 101 132 "0,1" 16

# 전체 200개 클래스, GPU 8개 사용
bash run_ti_fscil.sh 1 200 "0,1,2,3,4,5,6,7" 25
```

## 학습 파라미터

- **모델**: black-forest-labs/FLUX.1-dev
- **학습 스텝**: 2000
- **옵티마이저**: Prodigy (lr=1.0)
- **정밀도**: bfloat16
- **샘플 수**:
  - Base 클래스 (1-100): 랜덤 5장
  - Incremental 클래스 (101-200): FSCIL 고정 5-shot 샘플

## 출력 구조

```
cub_fscil_ti/
├── 001.Black_footed_Albatross/
│   └── Black_footed_Albatross_checkpoint_2000.safetensors
├── 002.Laysan_Albatross/
│   └── Laysan_Albatross_checkpoint_2000.safetensors
...
```

## 추론 (이미지 생성)

학습된 모델로 이미지를 생성하려면:

```bash
python inference_ti_class_name.py \
    --step 2000 \
    --checkpoint-dir cub_fscil_ti/029.American_Crow \
    --checkpoint-prefix American_Crow_checkpoint \
    --output-dir ti_bird_outputs \
    --num-inference-steps 25
```

## 파일 설명

- `run_ti_fscil.sh`: 메인 실행 스크립트 (multi-GPU, 클래스 범위 설정)
- `train_dreambooth_lora_flux_advanced.py`: FLUX Textual Inversion 학습 코드
- `inference_ti_class_name.py`: 학습된 TI 임베딩으로 이미지 생성
- `CUB_load.py`: FSCIL 5-shot 샘플 정의 (클래스 101-200)
- `environment.yml`: Conda 환경 설정
- `requirements.txt`: Python 패키지 목록

## 주의사항

- GPU당 할당되는 클래스 수를 조정하여 GPU 메모리를 효율적으로 사용하세요
- Base 클래스(1-100)는 랜덤 5장, Incremental 클래스(101-200)는 미리 정의된 5-shot 샘플을 사용합니다
- 전체 200개 클래스 학습 시 상당한 시간이 소요됩니다 (클래스당 약 10-15분)

## 라이선스

이 프로젝트는 연구 목적으로 제공됩니다.
