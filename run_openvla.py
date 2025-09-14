from utils import *
from transformers import AutoModelForVision2Seq, AutoProcessor
from world_model import WorldModel
import os
import numpy as np
from PIL import Image
# import mediapy as media
from tqdm import tqdm
import torch
from pathlib import Path


def evaluate_openvla(wm, vla, processor, tasks, retries=5, rollout_length=40):
    """
    Rollout an OpenVLA model on a list of tasks, and return the score on each task.
    Arguments:
        wm: WorldModel
        vla: An OpenVLA model from `transformers`
        tasks: A list of N tasks in loaded from a json. See "put_carrot_on_plate.json" for an example of the format.
    Returns:
        scores: A list of N scores from the VLM corresponding to each input task.
    """
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
                for step in tqdm(range(rollout_length)):
                    curr_frame = Image.fromarray(frames[-1])
                    prompt = f"In: What action should the robot take to {task_i['instruction']}?\nOut:"
                    inputs = processor(prompt, curr_frame).to(
                        device="cuda", dtype=torch.bfloat16
                    )
                    actions = vla.predict_action(
                        **inputs, unnorm_key="bridge_orig", do_sample=False
                    )
        
                    a = torch.tensor(actions).cuda()
                    # NOTE: OpenVLA outputs 7-dim actions, while the world model was trained with up to 10-dim actions.
                    a = torch.cat([a, a.new_zeros(3)], dim=-1)  # pad with zeros
                    a = rescale_bridge_action(a)
        
                    for i, x in wm.generate_chunk(a):
                        new_frame = x[0, 0].cpu().numpy()
                        new_frame = np.clip(new_frame * 255, 0, 255).astype(np.uint8)
                        frames.append(new_frame)
                rollout_video = np.stack(frames)
                # media.show_video(rollout_video, fps=20)
                result = predict(rollout_video, task=task_i)
                scores.append(result)
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

# Load Processor & VLA
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
).cuda()
vla.eval()

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
    scores = evaluate_openvla(wm, vla, processor, tasks)
    print(f"Example task: {tasks[0]}")
    evaluate(scores)
