import os, re
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import tensorflow_datasets as tfds

OUT_ROOT = "/vast/as20482/data/bridge/processed"

def sanitize(instr_tensor):
    # Convert TF scalar string tensor -> python str
    text = instr_tensor.numpy().decode("utf-8").strip().lower()
    # Replace spaces with underscores
    text = text.replace(" ", "_")
    text = re.sub(r"[^a-z_]", "", text)
    if not text:
        print(instr_tensor)
        text = "unknown"
    return text


def pick_first_available_image(step):
    return step["observation"]["image_0"]

def extract_first(ep):
    meta = {k: v.numpy() for k, v in ep["episode_metadata"].items()}
    first_step = next(iter(ep["steps"]))

    instr = first_step["language_instruction"]
    instr_key = sanitize(instr)
    img = pick_first_available_image(first_step).numpy()
    state = first_step["observation"]["state"].numpy()
    eid = int(meta["episode_id"])
    return instr_key, eid, img, state

def save_sample(instr_key, eid, img, state):
    d = os.path.join(OUT_ROOT, instr_key)
    os.makedirs(d, exist_ok=True)
    stem = f"ep{eid:08d}"

    # Save PNG
    png = tf.io.encode_png(img)  # img must be uint8 HWC
    tf.io.write_file(os.path.join(d, stem + ".png"), png)

    # Save state
    np.save(os.path.join(d, stem + ".state.npy"), state)

def export_by_instruction(ds, limit=None):
    os.makedirs(OUT_ROOT, exist_ok=True)
    counts = {}
    n = 0
    for ep in tqdm(ds):
        instr_key, eid, img, state = extract_first(ep)
        save_sample(instr_key, eid, img, state)
        counts[instr_key] = counts.get(instr_key, 0) + 1
        n += 1
        if limit and n >= limit:
            break
    return counts

ds_builder = tfds.builder_from_directory(builder_dir="/vast/as20482/data/bridge/1.0.0")
ds = ds_builder.as_dataset(split='val')
counts = export_by_instruction(ds)  # limit for testing
print("Episodes per instruction:", counts)
