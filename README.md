# Anima Artist Mix

[한국어 README](README.ko.md)

ComfyUI custom node for combining multiple Anima artist tags into a single positive `CONDITIONING` output.

The node is intended for workflows where you want to test blended artist styles without manually rebuilding the positive prompt each time.

Anima uses a Qwen3 text encoder and an LLM Adapter to create cross-attention conditioning for its DiT model. Because artist tags are part of that text conditioning path, blending separately encoded artist conditions can be useful when you want to explore style combinations more predictably than by only appending many tags to one long prompt.

Use this node when you want to:

- Compare artist mixes while keeping the same base prompt.
- Adjust each artist's influence with simple weights.
- Keep artist blending separate from the rest of your positive prompt.

## Node

Display name:

```text
Anima Artist Mixer
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

Use `average` first. It is the recommended mode for normal use because it gives a stable mixed style, returns one positive conditioning entry, and keeps sampling behavior straightforward.

`exact` and `prompt` are included as experimental alternatives. They can be useful for comparison or for finding a specific look, but their behavior may be more workflow-dependent than `average`.

| Goal | Recommended mode |
| --- | --- |
| Consistent blended style | `average` |
| Stronger individual artist character | `exact` |
| Fastest/manual prompt-like behavior | `prompt` |

If `artist_text` is empty, the node encodes only `base_prompt`.

## Install

Clone this repository into your ComfyUI `custom_nodes` folder:

```bash
cd ComfyUI/custom_nodes
git clone <repository-url> anima-artist-mix
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
