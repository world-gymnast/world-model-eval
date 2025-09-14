from utils import *
from transformers import AutoProcessor, AutoModel
from world_model import WorldModel
import os
import numpy as np
from PIL import Image
# import mediapy as media
from tqdm import tqdm
import torch
from pathlib import Path

def normalize_actions(unnorm_actions, statistics, key="bridge_orig/1.0.0"):
    stats = statistics[key]["action"]
    action_low = np.array(stats["q01"])
    action_high = np.array(stats["q99"])
    mask = np.array(stats.get("mask", np.ones_like(action_low)), dtype=bool)

    norm_actions = np.where(
        mask,
        2 * (unnorm_actions - action_low) / (action_high - action_low) - 1,
        unnorm_actions,  # leave unmasked dimensions as-is
    )
    return norm_actions

def evaluate_spatialvla(wm, vla, processor, tasks, retries=5, rollout_length=40):
    with torch.no_grad():
        scores = []
        for task_i in tqdm(tasks, desc="completing tasks"):
            start_frame = np.array(
                Image.open(task_i["im_0_path"]+".png").resize(
                    (256, 256)
                )
            )
            for _ in range(retries):
                # media.show_image(start_frame)
                wm.reset(torch.from_numpy(start_frame).cuda().float() / 255.0)
        
                frames = [start_frame]
                for step in range(rollout_length):
                    curr_frame = Image.fromarray(frames[-1])
                    prompt = f"In: What action should the robot take to {task_i['instruction']}?\nOut:"
                    inputs = processor(images=[curr_frame], text=prompt, return_tensors="pt").to(
                        device="cuda", dtype=torch.bfloat16
                    )
                    generation_outputs = vla.predict_action(inputs)
                    unnorm_actions = processor.decode_actions(generation_outputs, unnorm_key="bridge_orig/1.0.0")['actions'][0]
                    actions = normalize_actions(unnorm_actions, processor.statistics)
                    a = torch.tensor(actions).cuda()
                    a = torch.cat([a, a.new_zeros(3)], dim=-1)  # pad with zeros
                    a = rescale_bridge_action(a, wv_lo=-1, wv_hi=1, rd_lo=-1, rd_hi=1)
        
                    for i, x in wm.generate_chunk(a):
                        new_frame = x[0, 0].cpu().numpy()
                        new_frame = np.clip(new_frame * 255, 0, 255).astype(np.uint8)
                        frames.append(new_frame)
                rollout_video = np.stack(frames)
                # media.show_video(rollout_video, fps=20)
                avg_score = predict(rollout_video, task=task_i)
                scores.append(avg_score)
        return np.array(scores)

CHECKPOINTS_TO_KWARGS = {
    "bridge_v2_ckpt.pt": {  # The demo model checkpoint from our original arxiv release.
        "use_pixel_rope": True,
    },
    "200k_20frame_cfg_bridgev2_ckpt.pt": {  # New in-progress model with CFG and EMA.
        "use_pixel_rope": False,
        "default_cfg": 3.0,
    },
}
FILESERVER_URL = "https://85daf289d906.ngrok.app"  # This might change.

ckpt_path = "200k_20frame_cfg_bridgev2_ckpt.pt"  # Take your pick from above.
if not Path(ckpt_path).exists():
    ckpt_url = FILESERVER_URL + "/" + ckpt_path
    print(f"{ckpt_url=}")
    os.system(f"wget {ckpt_url}")

wm = WorldModel(ckpt_path, **CHECKPOINTS_TO_KWARGS[ckpt_path])

model_name_or_path="IPEC-COMMUNITY/spatialvla-4b-224-pt"
processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16).eval().cuda()
model.eval()

for base_path in TASKS.keys():
    tasks = []
    for task in load_tasks(base_path):
        tasks.append(
            {
                "im_0_path": task,
                "instruction": TASKS[base_path]["instruction"],
                "subtasks": TASKS[base_path]["subtasks"],
            }
        )
    scores = evaluate_spatialvla(wm, model, processor, tasks)
    print(f"Example task: {tasks[0]}")
    evaluate(scores)
