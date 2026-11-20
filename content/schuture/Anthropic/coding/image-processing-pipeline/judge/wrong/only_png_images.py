# expect: WA
# only processes files ending in .png; image_dir may hold any format Pillow decodes.
import json
import os
from concurrent.futures import ProcessPoolExecutor

from PIL import Image, ImageFilter


# ---------------------------------------------------------------- Part 1

def apply_step(image, step):
    kind = step["type"]
    if kind == "grayscale":
        return image.convert("L")
    if kind == "flip_horizontal":
        return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if kind == "flip_vertical":
        return image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    if kind == "scale":
        factor = step["factor"]
        # NOTE: round(), not int() -- truncating would bias every scaled dimension down
        new_size = (max(1, round(image.width * factor)), max(1, round(image.height * factor)))
        return image.resize(new_size, resample=Image.Resampling.BILINEAR)
    if kind == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=step["radius"]))
    if kind == "rotate":
        return image.rotate(step["angle"], resample=Image.Resampling.BILINEAR, expand=True)
    raise ValueError(f"unknown transformation type: {kind!r}")


def run_pipeline(image, steps):
    for step in steps:
        image = apply_step(image, step)
    return image


def _load_transformations(transformation_dir):
    """json_name -> its ordered list of steps."""
    steps_by_json = {}
    for json_name in sorted(os.listdir(transformation_dir)):
        with open(os.path.join(transformation_dir, json_name)) as f:
            steps_by_json[json_name] = json.load(f)["transformations"]
    return steps_by_json


def process_images(image_dir, transformation_dir, output_dir, get_output_path):
    steps_by_json = _load_transformations(transformation_dir)
    for image_name in sorted(n for n in os.listdir(image_dir) if n.endswith(".png")):
        with Image.open(os.path.join(image_dir, image_name)) as base:
            base.load()  # decode once per image
            for json_name, steps in steps_by_json.items():
                # NOTE: base is reused for every json_name -- safe since apply_step never mutates it
                run_pipeline(base, steps).save(get_output_path(image_name, json_name), format="PNG")


# ---------------------------------------------------------------- Part 2

def _process_one_image(task):
    image_path, steps_by_json, output_paths = task
    with Image.open(image_path) as base:  # NOTE: decode from a path, inside the worker -- not a pickled Image
        base.load()
        for json_name, steps in steps_by_json.items():
            run_pipeline(base, steps).save(output_paths[json_name], format="PNG")


def _dispatch(image_dir, transformation_dir, output_dir, get_output_path, worker, max_workers):
    steps_by_json = _load_transformations(transformation_dir)
    tasks = []
    for image_name in sorted(n for n in os.listdir(image_dir) if n.endswith(".png")):
        image_path = os.path.join(image_dir, image_name)
        # NOTE: resolved here, in the main process -- get_output_path itself never has to be picklable
        output_paths = {json_name: get_output_path(image_name, json_name) for json_name in steps_by_json}
        tasks.append((image_path, steps_by_json, output_paths))
    workers = max_workers or os.cpu_count() or 1
    chunksize = max(1, len(tasks) // (4 * workers))  # fewer IPC round-trips when there are many images
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(worker, tasks, chunksize=chunksize))


def process_images_parallel(image_dir, transformation_dir, output_dir, get_output_path, max_workers=None):
    _dispatch(image_dir, transformation_dir, output_dir, get_output_path, _process_one_image, max_workers)


# ---------------------------------------------------------------- Part 3

def _step_key(step):
    # NOTE: sorted() so a step's own JSON key order never matters for matching it against another
    return tuple(sorted(step.items()))


class _TrieNode:
    __slots__ = ("step", "children", "leaves")

    def __init__(self, step=None):
        self.step = step      # the step that produced this node's image from its parent's; None at the root
        self.children = {}    # _step_key(step) -> _TrieNode
        self.leaves = []      # json_names whose pipeline ends exactly at this node


def _build_trie(steps_by_json):
    root = _TrieNode()
    for json_name, steps in steps_by_json.items():
        node = root
        for step in steps:
            key = _step_key(step)
            if key not in node.children:  # NOTE: children is per-node, so the same step reached via a
                node.children[key] = _TrieNode(step)  # different prefix still gets its own node here
            node = node.children[key]
        node.leaves.append(json_name)
    return root


def _run_trie(node, image, output_paths):
    if node.step is not None:
        image = apply_step(image, node.step)
    applications = 0 if node.step is None else 1
    for json_name in node.leaves:
        image.save(output_paths[json_name], format="PNG")
    for child in node.children.values():
        applications += _run_trie(child, image, output_paths)
    return applications


def _process_one_image_shared_prefix(task):
    image_path, steps_by_json, output_paths = task
    trie = _build_trie(steps_by_json)
    with Image.open(image_path) as base:
        base.load()
        return _run_trie(trie, base, output_paths)


def process_images_shared_prefix(image_dir, transformation_dir, output_dir, get_output_path, max_workers=None):
    _dispatch(image_dir, transformation_dir, output_dir, get_output_path, _process_one_image_shared_prefix,
              max_workers)
