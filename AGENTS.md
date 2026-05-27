# Repository Guidelines

## Project Structure & Module Organization

This repository is a ComfyUI custom node package.

- `__init__.py`: registers ComfyUI node classes and display names.
- `nodes.py`: contains the node implementation, prompt parsing, and conditioning blend logic.
- `README.md`: English usage documentation.
- `README.ko.md`: Korean usage documentation.

There are currently no dedicated `tests/`, `web/`, or asset directories. Add new files only when they support the node directly.

## Build, Test, and Development Commands

Run commands from the repository root:

```bash
python3 -m py_compile nodes.py __init__.py
```

Checks Python syntax without starting ComfyUI.

```bash
git status --short
```

Reviews local changes before committing.

To test inside ComfyUI, place this repository under:

```text
ComfyUI/custom_nodes/anima-artist-mix
```

Then restart ComfyUI and confirm the `Anima Artist Mixer` node appears under `conditioning/anima`.

## Coding Style & Naming Conventions

Use Python 3 with 4-space indentation. Keep functions small and focused. Prefer explicit helper functions for parsing, formatting, and conditioning operations.

Naming patterns:

- Classes: `PascalCase`, e.g. `AnimaArtistMixerTextBlend`.
- Functions and variables: `snake_case`, e.g. `parse_prompt_artists`.
- ComfyUI display names are defined in `NODE_DISPLAY_NAME_MAPPINGS`.

Avoid unrelated refactors when changing node behavior.

## Testing Guidelines

No automated test suite exists yet. At minimum, run:

```bash
python3 -m py_compile nodes.py __init__.py
```

For behavior changes, manually test in ComfyUI with representative `artist_text` values:

```text
(artist one:1.2), (artist two:0.8), artist three
```

Verify that the node returns a valid `CONDITIONING` output and handles empty `artist_text`.

## Commit & Pull Request Guidelines

This repository has no established commit history yet. Use concise, imperative commit messages:

```text
Add Korean README
Update artist conditioning blend logic
```

Pull requests should include:

- A short description of the behavior change.
- Manual test steps and results.
- Notes about ComfyUI compatibility if relevant.
- Screenshots only when UI-visible behavior changes.

## Agent-Specific Instructions

Keep documentation in both `README.md` and `README.ko.md` aligned when user-facing behavior changes. Do not commit generated cache directories such as `__pycache__/`.
