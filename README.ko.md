# Anima Artist Mix

[English README](README.md)

여러 Anima 작가 태그를 하나의 positive `CONDITIONING` 출력으로 합치는 ComfyUI 커스텀 노드입니다.

positive prompt를 매번 직접 다시 만들지 않고, 여러 작가 스타일의 혼합을 비교하고 싶을 때 사용합니다.

## 노드

표시 이름:

```text
Anima Artst Mixer
```

카테고리:

```text
conditioning/anima
```

입력:

- `clip`: 체크포인트 워크플로에서 연결하는 CLIP/text encoder 입력입니다.
- `base_prompt`: 모든 작가 conditioning에 공통으로 들어갈 기본 프롬프트입니다.
- `artist_text`: 쉼표로 구분한 작가 태그입니다. 선택적으로 가중치를 줄 수 있습니다.
- `blend_mode`: 작가 conditioning을 합치는 방식입니다.

출력:

- `positive`: KSampler positive 입력에 연결할 `CONDITIONING`입니다.

이 노드는 텍스트 conditioning만 만들기 때문에 `MODEL` 입력이 필요하지 않습니다. 실제 샘플링은 KSampler나 다른 sampler 노드가 처리합니다.

## 작가 태그

작가 태그는 쉼표로 구분해서 입력합니다:

```text
@artist_one, @artist_two, @artist_three
```

가중치는 ComfyUI 스타일 prompt weighting 문법을 사용합니다:

```text
(@artist_one:1.2), (@artist_two:0.8), @artist_three
```

가중치를 적지 않은 태그는 `1.0`으로 처리됩니다.

Anima에서는 모델 문서의 권장 방식처럼 작가 태그에 보통 `@` prefix를 붙이는 것이 좋습니다.

## Blend Modes

### `average`

일관된 혼합 스타일을 원할 때 기본으로 추천하는 방식입니다.

노드는 각 작가를 아래 형태로 따로 encode합니다:

```text
base_prompt, artist
```

그 다음 결과 conditioning tensor를 정규화된 가중치로 평균내고, 하나의 conditioning 항목만 반환합니다.

예를 들어:

```text
(@artist_one:1.2), (@artist_two:0.8)
```

는 대략 아래처럼 합쳐집니다:

```text
artist_one conditioning * 0.6 + artist_two conditioning * 0.4
```

sampler에는 positive conditioning 항목 하나만 전달되기 때문에 보통 `exact`보다 빠릅니다. 핵심 blend가 weighted tensor average라서 작가 태그 순서의 영향도 매우 적습니다.

### `exact`

각 작가의 개별적인 영향이 더 살아있는 결과를 원할 때 적합합니다.

노드는 작가마다 별도의 conditioning 항목을 만들고, 정규화된 `strength` 값을 붙입니다. KSampler는 샘플링 중 이 조건들을 섞습니다.

`average`보다 작가별 특징이 더 뚜렷하게 남을 수 있지만, 작가 수가 늘어나면 샘플링이 느려질 수 있습니다.

### `prompt`

가장 빠른 baseline으로 쓰기 좋은 방식입니다.

노드는 아래처럼 하나의 prompt를 만든 뒤 한 번만 encode합니다:

```text
base_prompt, (@artist_one:1.2), (@artist_two:0.8), @artist_three
```

positive prompt에 작가 태그를 직접 모두 적는 방식과 가장 비슷합니다. conditioning 수준의 blend는 하지 않습니다.

## 어떤 모드를 쓰면 좋나요?

| 목표 | 추천 모드 |
| --- | --- |
| 일관된 혼합 스타일 | `average` |
| 작가별 개성 보존 | `exact` |
| 가장 빠르고 일반 prompt에 가까운 동작 | `prompt` |

`artist_text`가 비어 있으면 `base_prompt`만 encode합니다.

## 설치

이 폴더를 ComfyUI custom nodes 아래에 넣습니다:

```text
ComfyUI/custom_nodes/anima-artist-mix
```

설치하거나 업데이트한 뒤 ComfyUI를 재시작하세요.

## 개발 확인

이 repository root에서 실행합니다:

```bash
python3 -m py_compile nodes.py __init__.py
```

## 파일

- `__init__.py`: ComfyUI 노드 등록.
- `nodes.py`: prompt parsing과 conditioning blend logic.
