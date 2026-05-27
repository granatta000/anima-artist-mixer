# Anima Artist Mix

[한국어 README](README.ko.md)

ComfyUI custom node for combining multiple Anima artist tags into a single positive `CONDITIONING` output.

The node is intended for workflows where you want to test blended artist styles without manually rebuilding the positive prompt each time.

## Node

Display name:

```text
Anima Artst Mixer
```

Category:

```text
conditioning/anima
```

Inputs:

- `clip`: CLIP/text encoder input from your loaded checkpoint workflow.
- `base_prompt`: Main prompt text that should be shared by every artist condition.
- `artist_text`: Comma-separated artist tags, optionally with weights.
- `blend_mode`: How the artist conditions are combined.

Output:

- `positive`: A `CONDITIONING` output for KSampler positive input.

This node does not need a `MODEL` input because it only encodes text conditioning. Sampling is still handled by KSampler or another sampler node.

## Artist Tags

Enter artists as comma-separated tags:

```text
@artist_one, @artist_two, @artist_three
```

Weights use ComfyUI-style prompt weighting:

```text
(@artist_one:1.2), (@artist_two:0.8), @artist_three
```

Tags without an explicit weight use `1.0`.

For Anima, artist tags should usually include the `@` prefix, as recommended by the model documentation.

## Blend Modes

### `average`

Best default when you want a consistent mixed style.

The node encodes each artist separately as:

```text
base_prompt, artist
```

Then it averages the resulting conditioning tensors by normalized weight and returns one conditioning entry.

Example:

```text
(@artist_one:1.2), (@artist_two:0.8)
```

becomes approximately:

```text
artist_one conditioning * 0.6 + artist_two conditioning * 0.4
```

This is usually faster than `exact` because the sampler receives one positive conditioning entry. It is also mostly independent of artist order, because the main blend is a weighted tensor average.

### `exact`

Best when you prefer stronger preservation of each artist's individual influence.

The node encodes each artist as a separate conditioning entry and assigns a normalized `strength` value. KSampler then blends those conditions during sampling.

This can produce more distinctive artist influence than `average`, but sampling can slow down as the number of artists increases.

### `prompt`

Best as a fast baseline.

The node builds one prompt:

```text
base_prompt, (@artist_one:1.2), (@artist_two:0.8), @artist_three
```

and encodes it once. This is closest to manually typing every artist tag into the positive prompt. It does not do conditioning-level blending.

## Which Mode Should I Use?

| Goal | Recommended mode |
| --- | --- |
| Consistent blended style | `average` |
| Stronger individual artist character | `exact` |
| Fastest/manual prompt-like behavior | `prompt` |

If `artist_text` is empty, the node encodes only `base_prompt`.

## Install

Place this folder under ComfyUI custom nodes:

```text
ComfyUI/custom_nodes/anima-artist-mix
```

Restart ComfyUI after installing or updating.

## Development Check

From this repository root:

```bash
python3 -m py_compile nodes.py __init__.py
```

## Files

- `__init__.py`: ComfyUI node registration.
- `nodes.py`: Prompt parsing and conditioning blend logic.
