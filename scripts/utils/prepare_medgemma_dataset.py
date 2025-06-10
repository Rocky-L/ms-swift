#!/usr/bin/env python
"""Generate a JSONL dataset of messages for MedGemma inference.

Each image file under ``--image_dir`` should have a companion text file with the
same base name and extension specified by ``--text_suffix`` (default: ``.txt``).
The script will create one JSON object per image with the following format::

    {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": TEXT}
        ],
        "images": ["IMAGE_PATH"]
    }

The resulting objects are written to ``--output`` in JSONL format.
"""
import argparse
import json
import os
from typing import List

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}


def list_images(folder: str) -> List[str]:
    files = []
    for name in os.listdir(folder):
        ext = os.path.splitext(name)[1].lower()
        if ext in IMAGE_EXTS:
            files.append(name)
    files.sort()
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MedGemma inference dataset")
    parser.add_argument("--image_dir", required=True, help="Directory with images and text files")
    parser.add_argument("--output", required=True, help="Path to output jsonl file")
    parser.add_argument("--system_prompt", required=True, help="System prompt text")
    parser.add_argument("--text_suffix", default=".txt", help="Extension for accompanying text files")
    args = parser.parse_args()

    image_dir = os.path.abspath(args.image_dir)
    images = list_images(image_dir)
    with open(args.output, "w", encoding="utf-8") as f:
        for img_name in images:
            base = os.path.splitext(img_name)[0]
            text_path = os.path.join(image_dir, base + args.text_suffix)
            if os.path.isfile(text_path):
                with open(text_path, "r", encoding="utf-8") as t:
                    text = t.read().strip()
            else:
                text = ""
            obj = {
                "messages": [
                    {"role": "system", "content": args.system_prompt},
                    {"role": "user", "content": text},
                ],
                "images": [os.path.join(image_dir, img_name)],
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
