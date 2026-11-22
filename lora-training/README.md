# Style LoRA training tools

These tools make a style LoRA for Qwen Image 2.1 from a folder of frames. The folder has no dataset and no trained LoRA. Train only on images that you have the right to use, and obey the license of each source when you share a LoRA.

| File | Content |
|---|---|
| `build_dataset.py` | Takes frames from one or more folders. Drops near-duplicates (difference hash), small frames, and dark or gray frames. Crops letterbox bars. |
| `caption.py` | Writes a caption `.txt` for each image with a local vision model. Each caption starts with the trigger words, then gives 40 to 70 words of plain content. |
| `train.example.yaml` | The ai-toolkit configuration: rank 16, 4,500 steps, learning rate 1e-4, buckets 512 and 768 px, 12 GB VRAM with layer offloading. |

## Steps

1. Put the source frames in one folder for each source.
2. Build the set:
   ```bash
   python -I build_dataset.py frames/a frames/b --out dataset --per-folder 300
   ```
3. Start a vision model on an OpenAI-compatible server, for example Qwen3-VL 8B in LM Studio.
4. Write the captions:
   ```bash
   python -I caption.py dataset --trigger "mystyle style"
   ```
5. Copy `train.example.yaml`, and replace each `<...>` value.
6. Train with [ai-toolkit](https://github.com/ostris/ai-toolkit): `python run.py my-train.yaml`.

## Why the captions name no medium

The captions describe only the content: people, clothes, expressions, place, light, and shot. As a result, the trigger words carry the style. If a caption says "anime" or names a work, the LoRA ties the look to that word, and the trigger alone gives a weaker effect.

## Use in ComfyUI

Add the LoRA with **LoraLoaderModelOnly** after the diffusion model, at strength 0.8 to 1.0. Put the trigger words at the start of the prompt. In [`comfyui/2-storyteller/`](../comfyui/2-storyteller/README.md), give the file with `--extra-lora FILE:STRENGTH:TRIGGER`. Only a LoRA of a broad kind of look (for example photo to TV anime) goes into `loras.json` for automatic routing. A LoRA of one work does not go there.
