import math
import re
from dataclasses import dataclass
import logging

import torch

WEIGHTED_TAG_RE = re.compile(r"^\(\s*(?P<tag>.*?)\s*:\s*(?P<weight>[+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*\)$")


@dataclass(frozen=True)
class ArtistEntry:
    tag: str
    weight: float


def split_prompt_tags(text):
    tags = []
    current = []
    depth = 0

    for char in text or "":
        if char == "(":
            depth += 1
        elif char == ")" and depth > 0:
            depth -= 1

        if char == "," and depth == 0:
            tag = "".join(current).strip()
            if tag:
                tags.append(tag)
            current = []
            continue

        current.append(char)

    tag = "".join(current).strip()
    if tag:
        tags.append(tag)

    return tags


def parse_prompt_artists(text):
    artists = []

    for raw_tag in split_prompt_tags(text):
        raw_tag = raw_tag.strip()
        match = WEIGHTED_TAG_RE.match(raw_tag)
        if match:
            tag = match.group("tag").strip()
            try:
                weight = float(match.group("weight"))
            except ValueError:
                continue
        elif raw_tag.startswith("(") and raw_tag.endswith(")"):
            tag = raw_tag[1:-1].strip()
            weight = 1.0
        else:
            tag = raw_tag
            weight = 1.0

        if tag and math.isfinite(weight) and weight > 0:
            artists.append(ArtistEntry(tag=tag, weight=weight))

    return artists


def append_artist_prompt(base_prompt, artist_tag):
    base_prompt = (base_prompt or "").strip()
    artist_tag = artist_tag.strip()
    if not base_prompt:
        return artist_tag
    return f"{base_prompt}, {artist_tag}"


def format_weighted_artist_tag(artist):
    tag = artist.tag.strip()
    if math.isclose(artist.weight, 1.0):
        return tag
    return f"({tag}:{artist.weight:g})"


def build_artist_prompt(base_prompt, artists, weighted=False):
    if weighted:
        artist_tags = ", ".join(format_weighted_artist_tag(artist) for artist in artists)
    else:
        artist_tags = ", ".join(artist.tag.strip() for artist in artists)
    return append_artist_prompt(base_prompt, artist_tags)


def pad_conditioning_tensor(tensor, target_length):
    if tensor.shape[1] >= target_length:
        return tensor[:, :target_length]
    padding = torch.zeros(
        (tensor.shape[0], target_length - tensor.shape[1], tensor.shape[2]),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    return torch.cat([tensor, padding], dim=1)


def encode_base_prompt(clip, base_prompt):
    tokens = clip.tokenize(base_prompt or "")
    return clip.encode_from_tokens_scheduled(tokens)


def encode_prompt(clip, prompt):
    tokens = clip.tokenize(prompt)
    return clip.encode_from_tokens_scheduled(tokens)


def encode_artist_conditionings(clip, base_prompt, artists):
    total_weight = sum(artist.weight for artist in artists)
    encoded = []

    for artist in artists:
        prompt = append_artist_prompt(base_prompt, artist.tag)
        tokens = clip.tokenize(prompt)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        strength = artist.weight / total_weight

        for cond_tensor, meta in conditioning:
            output_meta = meta.copy()
            output_meta["strength"] = strength
            encoded.append([cond_tensor, output_meta])

    if not encoded:
        tokens = clip.tokenize(base_prompt or "")
        return (clip.encode_from_tokens_scheduled(tokens),)

    return (encoded,)


def encode_composite_average(clip, base_prompt, artists):
    total_weight = sum(artist.weight for artist in artists)
    encoded_artists = []

    for artist in artists:
        prompt = append_artist_prompt(base_prompt, artist.tag)
        conditioning = encode_prompt(clip, prompt)
        if len(conditioning) != 1:
            return None
        encoded_artists.append((artist, conditioning[0]))

    composite_prompt = build_artist_prompt(base_prompt, artists, weighted=True)
    composite_conditioning = encode_prompt(clip, composite_prompt)
    if len(composite_conditioning) != 1:
        return None

    max_length = max(cond_tensor.shape[1] for _, (cond_tensor, _) in encoded_artists)
    mixed_cond = None
    mixed_pooled = None

    for artist, (cond_tensor, meta) in encoded_artists:
        normalized_weight = artist.weight / total_weight
        padded = pad_conditioning_tensor(cond_tensor, max_length)
        weighted_cond = padded * normalized_weight
        mixed_cond = weighted_cond if mixed_cond is None else mixed_cond + weighted_cond

        pooled_output = meta.get("pooled_output")
        if pooled_output is not None:
            weighted_pooled = pooled_output * normalized_weight
            mixed_pooled = weighted_pooled if mixed_pooled is None else mixed_pooled + weighted_pooled

    _, composite_meta = composite_conditioning[0]
    output_meta = composite_meta.copy()
    output_meta.pop("strength", None)
    if mixed_pooled is not None:
        output_meta["pooled_output"] = mixed_pooled
    else:
        output_meta.pop("pooled_output", None)

    return ([[mixed_cond, output_meta]],)


def encode_prompt_mode(clip, base_prompt, artists):
    prompt = build_artist_prompt(base_prompt, artists, weighted=True)
    return (encode_prompt(clip, prompt),)


def encode_artist_blend(clip, base_prompt, artists, blend_mode="average"):
    if clip is None:
        raise RuntimeError("ERROR: clip input is invalid: None")

    if not artists:
        return (encode_base_prompt(clip, base_prompt),)

    if blend_mode == "prompt":
        return encode_prompt_mode(clip, base_prompt, artists)

    if blend_mode == "exact":
        return encode_artist_conditionings(clip, base_prompt, artists)

    composite_average = encode_composite_average(clip, base_prompt, artists)
    if composite_average is not None:
        return composite_average

    logging.warning("Anima artist composite average failed, falling back to sampler strength blend.")
    return encode_artist_conditionings(clip, base_prompt, artists)


class AnimaArtistMixerTextBlend:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "base_prompt": ("STRING", {"multiline": True, "default": "", "dynamicPrompts": True}),
                "artist_text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "(a:1.2), (b:0.3), c",
                        "dynamicPrompts": False,
                    },
                ),
                "blend_mode": (
                    ["average", "exact", "prompt"],
                    {"default": "average"},
                ),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("positive",)
    FUNCTION = "encode"
    CATEGORY = "conditioning/anima"

    def encode(self, clip, base_prompt, artist_text, blend_mode="average"):
        artists = parse_prompt_artists(artist_text)
        return encode_artist_blend(clip, base_prompt, artists, blend_mode=blend_mode)
