import json
import os
from typing import Callable

from PIL import Image, ImageFilter


def process_images(
    image_dir: str,
    transformation_dir: str,
    output_dir: str,
    get_output_path: Callable[[str, str], str],
) -> None:
    """Run every pipeline on every image in order and save each result as PNG at get_output_path(image_name, json_name)."""
    raise NotImplementedError


def process_images_parallel(
    image_dir: str,
    transformation_dir: str,
    output_dir: str,
    get_output_path: Callable[[str, str], str],
    max_workers: int | None = None,
) -> None:
    """Same outputs as process_images, spread over at most max_workers worker processes; no image decoded once per pipeline, only paths and JSON-sized data cross process boundaries."""
    raise NotImplementedError


def process_images_shared_prefix(
    image_dir: str,
    transformation_dir: str,
    output_dir: str,
    get_output_path: Callable[[str, str], str],
    max_workers: int | None = None,
) -> None:
    """Like process_images_parallel, but each prefix shared by two or more pipelines of the same image is computed once."""
    raise NotImplementedError
