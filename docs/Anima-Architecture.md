# Anima — DiT 기반 이미지 생성 모델 아키텍처

> **개발사:** CircleStone Labs × Comfy Org  
> **기반 모델:** NVIDIA Cosmos-Predict2-2B-Text2Image  
> **라이선스:** CircleStone Labs Non-Commercial License (NVIDIA OMAL 준용)  
> **현재 버전:** Base v1.0 (Preview 3 기준 정리)

---

## 1. 모델 개요

Anima는 **20억(2B) 파라미터** 규모의 텍스트-투-이미지 확산 모델로, 애니메이션·일러스트·비사실적 예술 이미지 생성에 특화되어 있습니다.  
SDXL 계열(NoobAI, Illustrious 등)과 달리 **NVIDIA Cosmos-Predict2** 아키텍처를 기반으로 하는 완전히 다른 계보의 첫 번째 애니메이션 모델입니다.

| 항목 | 값 |
|---|---|
| 파라미터 수 | ~2B |
| 학습 데이터 | 애니메이션 이미지 수백만 장 + 비사진 예술 이미지 ~80만 장 |
| 합성 데이터 사용 | 없음 |
| 기본 해상도 | ~1MP (1024×1024, 896×1152 등) |
| 권장 샘플링 스텝 | 30–50 steps |
| 권장 CFG | 4–5 |
| 양자화 없는 모델 크기 | ~7GB |
| 최소 VRAM | 8GB |

---

## 2. 전체 아키텍처 구성

Anima는 세 가지 독립 컴포넌트로 구성된 **Latent Diffusion** 파이프라인입니다.

```
[텍스트 입력]
    │
    ▼
┌─────────────────────┐
│  Qwen3-0.6B         │  ← 텍스트 인코더 (LLM)
│  (Text Encoder)     │
└────────┬────────────┘
         │ Hidden States (Qwen3 임베딩 공간)
         ▼
┌─────────────────────┐
│  LLM Adapter        │  ← 6-레이어 트랜스포머 브릿지
│  (Bridge)           │     Qwen3 → T5 호환 Cross-Attention 공간
└────────┬────────────┘
         │ 조건부 임베딩
         ▼
┌─────────────────────────────────────────────────────┐
│  MiniTrainDIT (Diffusion Transformer)               │
│                                                     │
│  ┌───────────┐   ┌───────────┐        ┌──────────┐  │
│  │AnimaBlock │ → │AnimaBlock │ → ... → │AnimaBlock│  │
│  │  ×N       │   │           │        │          │  │
│  └───────────┘   └───────────┘        └──────────┘  │
│                                                     │
│  • Self-Attention (3D RoPE 적용)                    │
│  • Cross-Attention (LLM Adapter 출력 조건부)         │
│  • MLP (Feed-Forward)                               │
│  • AdaLN (Adaptive Layer Norm, 타임스텝 조건부)      │
└────────────────────────┬────────────────────────────┘
                         │ 노이즈 제거된 Latent
                         ▼
┌─────────────────────┐
│  Qwen-Image VAE     │  ← 디코더
│  (16ch, 8× 다운스케일)│
└─────────────────────┘
         │
         ▼
[생성된 이미지]
```

---

## 3. 핵심 컴포넌트 상세

### 3.1 텍스트 인코더: Qwen3-0.6B

- **모델:** Qwen3-0.6B (causal LM 기반, `Qwen2ForCausalLM`)
- **역할:** 프롬프트를 토큰화하여 hidden state 임베딩 시퀀스 생성
- **특징:**
  - 자연어 및 Danbooru 스타일 태그 혼합 입력 지원
  - 파라미터 규모가 작아(0.6B) 일반 모델 대비 텍스트 이해 능력에 제약 있음
  - `--llm_adapter_lr=0` 옵션으로 파인튜닝 시 어댑터 학습 비활성화 권장 (LLM Adapter는 쉽게 품질 저하됨)

**이중 토크나이저 전략 (`AnimaTokenizeStrategy`)**

| 토크나이저 | 용도 |
|---|---|
| Qwen3 토크나이저 | 실제 텍스트 인코더 입력 — Hidden State 생성 |
| T5 토크나이저 | LLM Adapter의 타겟 Input ID로 사용 (T5 모델 자체는 사용 안 함) |

---

### 3.2 LLM Adapter (핵심 브릿지)

Anima의 가장 독특한 설계 요소로, Qwen3의 임베딩 공간을 DiT 블록이 이해할 수 있는 **T5 호환 Cross-Attention 공간**으로 변환합니다.

```
Qwen3 Hidden States
        │
        ▼
┌──────────────────────────────┐
│  LLM Adapter (6-Layer Transformer)  │
│                              │
│  Layer 1: Self-Attention     │
│  Layer 2: Self-Attention     │
│  ...                         │
│  Layer 6: Self-Attention     │
│            + out_proj        │
└──────────────────────────────┘
        │
        ▼
T5-Compatible Cross-Attention Embeddings
(AnimaBlock의 cross_attn 레이어로 전달)
```

- **구현 클래스:** `LLMAdapter` (`library/anima_models.py:534-583`)
- LLM Adapter 가중치는 DiT 모델 파일 내부에 번들링됨 (`llm_adapter.out_proj.weight` 키로 식별)
- 이미지 생성 품질에 가장 큰 영향을 미치는 컴포넌트

---

### 3.3 MiniTrainDIT (Diffusion Transformer 본체)

Cosmos-Predict2 스타일의 Diffusion Transformer 구현체입니다. U-Net을 완전히 대체합니다.

#### AnimaBlock 구조

각 트랜스포머 블록(`AnimaBlock`)은 다음 순서로 처리됩니다:

```
입력 노이즈 Latent (패치화된 토큰 시퀀스)
        │
        ▼
┌─────────────────────────┐
│  AdaLN (Adaptive LN)    │ ← 타임스텝 t 조건부 정규화
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Self-Attention          │ ← 3D RoPE 위치 임베딩 적용
│  (이미지 패치 간 관계)   │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Cross-Attention         │ ← LLM Adapter 출력을 Key/Value로 사용
│  (텍스트 조건부 주의)    │   (텍스트-이미지 정렬의 핵심)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  MLP (Feed-Forward)      │
└────────────┬────────────┘
             │
             ▼
다음 AnimaBlock으로
```

#### 구현 클래스 매핑

| 컴포넌트 | 클래스명 | 파일 위치 |
|---|---|---|
| 메인 모델 | `Anima` | `anima_models.py:1143` |
| 트랜스포머 블록 | `AnimaBlock` | `anima_models.py:978` |
| 텍스트 조건부 브릿지 | `LLMAdapter` | `anima_models.py:534` |
| 위치 임베딩 (3D) | `PositionEmbedding3D` | `anima_models.py:364` |
| 위치 임베딩 (2D) | `PositionEmbedding2D` | `anima_models.py:328` |

---

### 3.4 위치 임베딩: 3D RoPE

Anima는 일반적인 절대 위치 임베딩 대신 **3D Rotary Positional Embedding(RoPE)** 을 사용합니다.

| 차원 | 설명 |
|---|---|
| H (Height) | 이미지 세로 공간 정보 |
| W (Width) | 이미지 가로 공간 정보 |
| T (Temporal) | 프레임/시간 차원 (비디오 확장 대비) |

- **`apply_rotary_pos_emb`** (`anima_models.py:175-212`): Self-Attention 수행 전 Q, K 벡터에 RoPE 회전 변환 적용
- **`PositionEmbedding2D`** (`anima_models.py:328-361`): 이미지 단독 처리 시 H×W 2차원 인코딩
- **`PositionEmbedding3D`** (`anima_models.py:364-419`): T×H×W 3차원 인코딩 (비디오/멀티프레임 지원 구조)

---

### 3.5 VAE: Qwen-Image VAE

이미지와 Latent 공간 간의 인코딩/디코딩을 담당합니다.

| 항목 | 값 |
|---|---|
| 구현 클래스 | `Qwen2ImageAutoencoderKL` |
| Latent 채널 수 | **16채널** (SD1.5의 4채널 대비 4배) |
| 공간 다운스케일 비율 | **8× spatial downscale** |
| 1024×1024 이미지 → Latent | 128×128×16 텐서 |
| 아키텍처 | KL-Autoencoder (VAE) 변형 |

> Qwen-Image VAE는 Cosmos-Predict2와 동일한 아키텍처를 공유하므로, 이미 보유하고 있다면 재사용 가능합니다.

---

## 4. 확산 프로세스: Rectified Flow

Anima는 DDPM 대신 **Rectified Flow (정류 흐름)** 방식으로 학습됩니다.

```
노이즈 x_1  ──────────────────────────→  데이터 x_0
              직선 경로로 보간
              v_θ(x_t, t, c) 학습
```

- **타임스텝 샘플링:** FLUX 방식과 동일한 옵션 지원
  - `sigma`, `uniform`, `sigmoid`, `shift`, `flux_shift`
- **목적함수:** Flow Matching loss (벡터장 v_θ 예측)
- **CFG (Classifier-Free Guidance):** caption_dropout_rate=0.15로 학습 → 추론 시 CFG 4–5 적용

---

## 5. 조건부 정규화: AdaLN

타임스텝 `t` 정보를 각 블록에 주입하는 방식으로 **Adaptive Layer Normalization**을 사용합니다.

```
타임스텝 t → 임베딩 → scale(γ), shift(β) 파라미터 생성
                              │
                              ▼
LayerNorm(x) × γ + β → 이후 어텐션/MLP 레이어
```

- FLUX의 단일 스트림 구조나 SD3의 MMDiT와는 다른 별도 구현
- `AdaLN` 모듈은 파라미터 그룹에서 독립적으로 학습률 조정 가능

---

## 6. ControlNet-LLLite 확장 (경량 조건부 제어)

Kohya-ss가 구현한 경량 이미지 조건부 제어 모듈입니다.

```
조건부 이미지 (depth, pose, lineart 등)
        │
        ▼
conditioning_image → 공유 conditioning1 임베딩
        │
        ▼ (broadcast)
각 AnimaBlock 내 선택된 Linear 레이어에 삽입된 소형 어댑터 모듈
(LoRA와 유사한 주입 방식, self_attn 레이어 대상)
```

- `--cond_emb_dim`, `--lllite_cond_dim`, `--lllite_mlp_dim` 파라미터로 임베딩 차원 설정
- U-Net 기반 ControlNet 대비 훨씬 가벼운 파라미터 수
- Anima의 **MiniTrainDIT** 아키텍처에 맞게 포팅

---

## 7. LoRA 학습 지원

| 적용 대상 레이어 | 설명 |
|---|---|
| `.self_attn` | Self-Attention Q/K/V/O 프로젝션 |
| `.cross_attn` | Cross-Attention Q/K/V/O 프로젝션 |
| `.mlp` | Feed-Forward 네트워크 |

- **권장 rank:** 32
- **권장 학습률:** 2e-5 (Base 모델 특성상 낮게 시작)
- **LLM Adapter 학습:** `llm_adapter_lr=0`으로 반드시 비활성화 권장
- **최소 VRAM (rank 32, 512px):** ~10GB

---

## 8. 메모리 최적화 전략

| 기법 | 설명 |
|---|---|
| Gradient Checkpointing (Unsloth 방식) | 활성화를 CPU RAM으로 비동기 오프로드 |
| Block Swapping | DiT 블록 단위로 CPU ↔ GPU 동적 이동 |
| FP8 Quantization | DiT 모델을 FP8 정밀도로 로드 |
| BF16 | 기본 연산 정밀도 (권장) |
| Disk Offload | VRAM 부족 시 가중치를 디스크로 오프로드 |

---

## 9. 타 아키텍처와의 비교

| 항목 | Anima (MiniTrainDIT) | FLUX.1 | SD3.5 | SDXL |
|---|---|---|---|---|
| 기반 구조 | DiT (Cosmos-Predict2) | DiT (MMDiT 변형) | MMDiT | U-Net |
| 텍스트 인코더 | Qwen3-0.6B + LLM Adapter | CLIP + T5-XXL | CLIP×2 + T5-XXL | CLIP×2 |
| VAE 채널 수 | 16ch | 16ch | 16ch | 4ch |
| 공간 다운스케일 | 8× | 8× | 8× | 8× |
| 위치 임베딩 | 3D RoPE | RoPE | Learned | 없음 |
| 확산 방식 | Rectified Flow | Rectified Flow | Rectified Flow | DDPM |
| 파라미터 수 | 2B | 12B | 8B | 3.5B |

---

## 10. 알려진 한계

- **사실적 이미지 생성 불가:** 의도적으로 사진적 사실성을 배제하고 훈련
- **고해상도 한계:** 약 2MP부터 품질 저하 (현재 ~1MP 최적)
- **텍스트 렌더링:** 단어 수준은 가능하나 긴 문자열은 불안정
- **텍스트 인코더 제약:** Qwen3-0.6B은 동급 모델(4B 표준) 대비 소규모라 복잡한 텍스트 이해에 한계
- **Preview 체크포인트:** 아직 미적 튜닝(RLHF 등) 미적용으로 아티스트·품질 태그 없이는 스타일이 중립적

---

## 참고 자료

- [circlestone-labs/Anima (HuggingFace)](https://huggingface.co/circlestone-labs/Anima)
- [kohya-ss/sd-scripts — Anima Training Docs](https://github.com/kohya-ss/sd-scripts/blob/main/docs/anima_train_network.md)
- [kohya-ss/sd-scripts — Anima Models Source](https://github.com/kohya-ss/sd-scripts/blob/main/library/anima_models.py)
- [nvidia-cosmos/cosmos-predict2 (기반 모델)](https://github.com/nvidia-cosmos/cosmos-predict2)
- [DiffSynth-Studio Anima 문서](https://github.com/modelscope/DiffSynth-Studio/blob/main/docs/en/Model_Details/Anima.md)
- [DeepWiki — Anima Training Architecture](https://deepwiki.com/kohya-ss/sd-scripts/7.3-anima-training)