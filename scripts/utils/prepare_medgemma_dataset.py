#!/usr/bin/env python
"""Generate a JSONL dataset of messages for MedGemma inference.

Each image file under ``--image_dir`` will be processed to create one JSON
object. The resulting objects are written to ``--output`` in JSONL format.

The system prompt is read from the file specified by ``--system_prompt_path``.
The user content is a fixed string provided by ``--user_content`` for all images.

Each JSON object has the following format::

    {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_FROM_FILE},
            {"role": "user", "content": USER_CONTENT_ARGUMENT}
        ],
        "images": ["IMAGE_PATH"]
    }
"""
import argparse
import json
import os
from typing import List

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}


def list_images(folder: str) -> List[str]:
    """Finds all image files in a directory."""
    files = []
    for name in os.listdir(folder):
        ext = os.path.splitext(name)[1].lower()
        if ext in IMAGE_EXTS:
            files.append(name)
    files.sort()
    return files


def main() -> None:
    """Parses arguments and generates the JSONL dataset."""
    parser = argparse.ArgumentParser(description="Prepare MedGemma inference dataset")
    parser.add_argument("--image_dir", required=True, help="Directory with images")
    parser.add_argument("--output", required=True, help="Path to output jsonl file")
    parser.add_argument(
        "--system_prompt_path",
        required=True,
        help="Path to a text file containing the system prompt",
    )
    parser.add_argument(
        "--user_content",
        required=True,
        help="The user content text to be used for all images",
    )
    args = parser.parse_args()

    # Read the system prompt from the specified file
    with open(args.system_prompt_path, "r", encoding="utf-8") as f:
        system_prompt = f.read().strip()

    image_dir = os.path.abspath(args.image_dir)
    images = list_images(image_dir)
    with open(args.output, "w", encoding="utf-8") as f:
        for img_name in images:
            # The user content is now the fixed string from the argument.
            # The logic for reading companion text files has been removed.
            obj = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"<image>{args.user_content}"},
                ],
                "images": [os.path.join(image_dir, img_name)],
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    """
    python /root/storage/qiuyi.lqy/code/ms-swift/scripts/eval/tongue_diagnosis/data_preprocess.py \
        --image_dir /root/storage/qiuyi.lqy/data/mllm/tongue_diagnosis \
        --output /root/storage/qiuyi.lqy/data/mllm/tongue_diagnosis_predicted.jsonl \
        --system_prompt_path /root/storage/qiuyi.lqy/code/ms-swift/scripts/eval/tongue_diagnosis/system_prompt.txt \
        --user_content 请分析这个患者的舌象分类
    """
    main()
