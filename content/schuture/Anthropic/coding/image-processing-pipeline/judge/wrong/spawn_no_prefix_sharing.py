# expect: WA
# Workers started with the "spawn" method (must work in the judge); Part 3 just reuses Part 2 and shares no prefix.
import json, os, multiprocessing
from concurrent.futures import ProcessPoolExecutor
from PIL import Image, ImageFilter

def apply_step(image, step):
    kind = step["type"]
    if kind == "grayscale":
        return image.convert("L")
    if kind == "flip_horizontal":
        return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if kind == "flip_vertical":
        return image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    if kind == "scale":
        f = step["factor"]
        return image.resize((max(1, round(image.width * f)), max(1, round(image.height * f))), resample=Image.Resampling.BILINEAR)
    if kind == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=step["radius"]))
    return image.rotate(step["angle"], resample=Image.Resampling.BILINEAR, expand=True)

def _load(d):
    out = {}
    for n in sorted(os.listdir(d)):
        with open(os.path.join(d, n)) as f:
            out[n] = json.load(f)["transformations"]
    return out

def process_images(image_dir, transformation_dir, output_dir, get_output_path):
    pipes = _load(transformation_dir)
    for img in sorted(os.listdir(image_dir)):
        with Image.open(os.path.join(image_dir, img)) as base:
            base.load()
            for js, steps in pipes.items():
                im = base
                for s in steps:
                    im = apply_step(im, s)
                im.save(get_output_path(img, js), format="PNG")

def _one(task):
    path, pipes, outs = task
    with Image.open(path) as base:
        base.load()
        for js, steps in pipes.items():
            im = base
            for s in steps:
                im = apply_step(im, s)
            im.save(outs[js], format="PNG")

def process_images_parallel(image_dir, transformation_dir, output_dir, get_output_path, max_workers=None):
    pipes = _load(transformation_dir)
    tasks = [(os.path.join(image_dir, i), pipes, {j: get_output_path(i, j) for j in pipes}) for i in sorted(os.listdir(image_dir))]
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        list(pool.map(_one, tasks))

def process_images_shared_prefix(image_dir, transformation_dir, output_dir, get_output_path, max_workers=None):
    process_images_parallel(image_dir, transformation_dir, output_dir, get_output_path, max_workers)
