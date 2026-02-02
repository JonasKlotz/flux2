from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL.ImageFile import ImageFile
from datasets import load_dataset
from PIL import Image
from flux2.util import load_mistral_small_embedder, load_flow_model, load_ae
from image_gen_utils import build_prompt, generate_image
from io_utils import ensure_dir
from local_datasets.coco_dataset import COCODataset, plot_coco, COCO_CLASSES

import lightning

# fix all seeds for reproducibility
SEED = 42
lightning.seed_everything(SEED)

from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple

import torch

COCO_REPLACE_MAP = {
    "person": "bicycle",               # weak: human replaced by nearby object, but street context plausible
    "bicycle": "motorcycle",
    "car": "truck",
    "motorcycle": "bicycle",
    "airplane": "bus",                 # weak: airport shuttle bus is plausible context
    "bus": "train",
    "train": "bus",
    "truck": "car",
    "boat": "truck",                   # weak: marina truck is plausible, but not ideal
    "traffic light": "stop sign",
    "fire hydrant": "stop sign",
    "stop sign": "traffic light",
    "parking meter": "fire hydrant",   # weak but street context consistent
    "bench": "chair",

    "bird": "cat",                     # weak: outdoor animal swap; better would be "dog" but choose one
    "cat": "dog",
    "dog": "cat",
    "horse": "cow",
    "sheep": "cow",
    "cow": "sheep",
    "elephant": "giraffe",
    "bear": "dog",                     # weak
    "zebra": "horse",
    "giraffe": "elephant",

    "backpack": "handbag",
    "umbrella": "backpack",            # weak but street context plausible
    "handbag": "backpack",
    "tie": "backpack",                 # weak: clothing accessory swap is hard within COCO set
    "suitcase": "backpack",

    "frisbee": "sports ball",
    "skis": "snowboard",
    "snowboard": "skis",
    "sports ball": "frisbee",
    "kite": "frisbee",                 # weak
    "baseball bat": "tennis racket",
    "baseball glove": "sports ball",
    "skateboard": "surfboard",         # weak but board-like
    "surfboard": "skateboard",
    "tennis racket": "baseball bat",

    "bottle": "cup",
    "wine glass": "cup",
    "cup": "wine glass",
    "fork": "spoon",
    "knife": "fork",
    "spoon": "fork",
    "bowl": "cup",

    "banana": "apple",
    "apple": "orange",
    "sandwich": "hot dog",
    "orange": "apple",
    "broccoli": "carrot",
    "carrot": "broccoli",
    "hot dog": "sandwich",
    "pizza": "cake",                   # weak but food context ok
    "donut": "cake",
    "cake": "donut",

    "chair": "couch",
    "couch": "bed",
    "potted plant": "vase",
    "bed": "couch",
    "dining table": "chair",           # weak: table to chair changes scene logic, but indoor context ok
    "toilet": "sink",
    "tv": "laptop",
    "laptop": "tv",
    "mouse": "remote",
    "remote": "cell phone",
    "keyboard": "laptop",
    "cell phone": "remote",

    "microwave": "oven",
    "oven": "microwave",
    "toaster": "microwave",
    "sink": "toilet",
    "refrigerator": "oven",            # weak but kitchen context ok

    "book": "clock",
    "clock": "book",
    "vase": "potted plant",
    "scissors": "knife",
    "teddy bear": "cat",               # weak
    "hair drier": "toothbrush",        # weak: bathroom context, but scale mismatch
    "toothbrush": "hair drier",        # weak
}

import json
import os
from pathlib import Path
from typing import Dict, Any, List
import numpy as np


def atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def init_or_load_coco_attr_json(
    anno_path: Path,
    dataset,
) -> Dict[str, Any]:
    """
    COCO-like JSON extended with image-level attributes.
    """
    anno_path = Path(anno_path)
    if anno_path.exists():
        with open(anno_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # basic sanity
        for k in ["images", "categories", "image_attributes"]:
            if k not in data:
                raise KeyError(f"Missing key '{k}' in existing annotation file: {anno_path}")
        return data

    categories = []
    for cls_idx, coco_id in enumerate(dataset.cat_ids):
        categories.append(
            {
                "id": int(coco_id),
                "name": dataset.get_class_name(cls_idx),
                "supercategory": "none",
            }
        )

    return {
        "info": {"description": "Synthetic COCO with image-level attribute labels"},
        "licenses": [],
        "images": [],
        "categories": categories,
        "annotations": [],        # kept empty intentionally (no instance labels for synthetic)
        "image_attributes": [],   # custom: image-level multi-labels
    }


def mlc_to_attr_entries(
    image_id: int,
    mlc_vector,
    dataset,
    certainty: float = 1.0,
    time: str = "",
) -> List[Dict[str, Any]]:
    """
    Convert mlc_vector (contiguous idx space) to per-category presence entries using COCO category_id.
    """
    v = mlc_vector.detach().cpu().numpy().astype(np.int64)
    pos = np.where(v == 1)[0].tolist()
    entries = []
    for cls_idx in pos:
        coco_cat_id = int(dataset.cat_ids[cls_idx])
        entries.append(
            {
                "image_id": int(image_id),
                "category_id": coco_cat_id,
                "present": 1,
                "certainty": float(certainty),
                "time": time,
            }
        )
    return entries


def flip_remove_add_in_mlc(
    mlc_vector,
    remove_cls_idx: int,
    add_cls_idx: int,
):
    """
    Returns a new tensor-like vector with exactly these two edits applied (idempotent).
    """
    v = mlc_vector.clone()
    v[remove_cls_idx] = 0
    v[add_cls_idx] = 1
    return v


def add_image_record(data: Dict[str, Any], image_id: int, file_name: str, width: int, height: int) -> None:
    data["images"].append(
        {
            "id": int(image_id),
            "file_name": file_name,
            "width": int(width),
            "height": int(height),
        }
    )


def append_attr_entries(data: Dict[str, Any], entries: List[Dict[str, Any]]) -> None:
    data["image_attributes"].extend(entries)

def build_name_to_cls_idx(dataset) -> Dict[str, int]:
    d = {}
    for i in range(len(dataset.cat_ids)):
        d[dataset.get_class_name(i)] = i
    return d


def ann_area(ann: Dict[str, Any]) -> float:
    """
    Prefer COCO's stored 'area' if present, otherwise fall back to bbox area.
    """
    if "area" in ann and ann["area"] is not None:
        return float(ann["area"])
    # COCO bbox is [x, y, w, h]
    _, _, w, h = ann["bbox"]
    return float(w) * float(h)


def select_class_with_fewest_instances_and_area(
    mlc_vector: torch.Tensor,
    anns: List[Dict[str, Any]],
    dataset,
) -> Optional[Tuple[int, int, str, List[Dict[str, Any]], int, float]]:
    """
    Returns:
      (cls_idx, coco_cat_id, class_name, selected_anns, n_instances, total_area)

    Selection criterion among positive classes:
      minimize (n_instances, total_area) with n_instances > 0.
    """
    # positives in your contiguous index space
    pos = torch.nonzero(mlc_vector).squeeze().tolist()
    if isinstance(pos, int):
        pos = [pos]

    # group anns by coco category_id
    anns_by_cat: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for a in anns:
        anns_by_cat[int(a["category_id"])].append(a)

    best = None  # will store tuple comparable by (n, area)
    best_payload = None

    for cls_idx in pos:
        coco_cat_id = int(dataset.cat_ids[cls_idx])
        cand = anns_by_cat.get(coco_cat_id, [])
        if len(cand) == 0:
            continue  # ignore positives with no annotated instances (weak/noisy label)

        n = len(cand)
        area_sum = sum(ann_area(a) for a in cand)

        key = (n, area_sum)
        if best is None or key < best:
            best = key
            best_payload = (cls_idx, coco_cat_id, dataset.get_class_name(cls_idx), cand, n, area_sum)

    return best_payload


def build_prompt_remove_all_coco(target_class: str) -> str:
    return f"""
Task: counterfactual object removal edit.

Input: A single base photograph to be edited.

Target object class to remove: {target_class}.

Primary rule (hard priority): Preserve the input image exactly. Treat the original photo as ground truth for scene layout, identity, and all pixels except where the specified object class must be removed.

Allowed change (semantic removal only):
Remove all visible instances of {target_class} from the image. This includes every object of that class, regardless of size, distance, partial occlusion, or position in the frame.

After removal, fill the previously occupied regions with a realistic continuation of the surrounding scene, consistent with local geometry, texture, and perspective.

Invariances (must not change):
1. Camera viewpoint, framing, scale, and perspective.
2. Scene layout and spatial arrangement of all remaining objects.
3. Identity, pose, shape, and appearance of all objects that are not {target_class}.
4. Background structures.
5. Lighting direction, intensity, shadows, reflections, and overall color grading, except for minimal local adjustments strictly required by the physical absence of the removed objects.
6. Global weather, time of day, and atmosphere.
7. No stylistic changes; keep the image photorealistic.

Prohibitions (explicit negatives):
1. Do not add any new objects or people.
2. Do not remove or modify any object that is not a {target_class}.
3. Do not alter object shapes, poses, or positions except where pixels were previously occupied by a {target_class}.
4. Do not change the environment or setting.
5. Do not introduce stylization, blur, or artifacts.

Output requirement: One realistic photograph that matches the original image in every respect except that all instances of {target_class} are absent and the scene has been plausibly completed where they used to be.
""".strip()


def build_prompt_remove_and_add_coco(label_to_remove: str, label_to_add: str) -> str:
    return f"""
Task: counterfactual object-class substitution edit.

Input: A single base photograph to be edited.

Primary rule (hard priority): Preserve the input image exactly. Treat the original photo as ground truth for camera viewpoint, framing, background, and all objects except those belonging to the target class.

Target substitution (the only allowed semantic change):
Replace every visible instance of "{label_to_remove}" with an instance of "{label_to_add}". There might be only a single instance or multiple instances of "{label_to_remove}"; all must be replaced.

Localization constraint:
The substitutions must occur only at the locations where "{label_to_remove}" appears in the original image. Do not introduce "{label_to_add}" in any new locations. Do not remove or alter any object that is not "{label_to_remove}".

Geometric and physical plausibility:
Each new "{label_to_add}" must be consistent with the original scene geometry and perspective. Match the original object placement, approximate size, and viewpoint as closely as possible. If an exact pose transfer is impossible, apply the minimal geometric adjustment needed to keep the result physically plausible, while keeping the rest of the image unchanged.

Appearance and realism constraints:
Maintain photorealism. Keep global illumination, color grading, and atmosphere unchanged. Only adjust local shadows, reflections, and contact regions strictly required by the substituted objects.

Invariances (must not change):
1. Camera viewpoint, framing, scale, and perspective.
2. Background, environment, and scene layout.
3. All objects and regions that are not "{label_to_remove}".
4. Number of objects outside the substituted regions. No new objects. No deletions besides removing "{label_to_remove}" via substitution.

Prohibitions:
1. No stylization. No blur. No artifacts. No text. No watermarks.
2. Do not change weather, time of day, or overall lighting.
3. Do not modify the shapes, identities, or positions of non-target objects.

Output requirement: One realistic photograph that matches the original image in every respect except that all "{label_to_remove}" instances have been replaced by "{label_to_add}" in-place.
""".strip()

def main():
    coco_path = "/data/jonas/COCO"
    coco_train_path = os.path.join(coco_path, "train2017")
    annotation_file = os.path.join(
        coco_path, "annotations_trainval2017/annotations/instances_train2017.json"
    )
    output_dir = Path(ensure_dir("/home/jonas/PycharmProjects/flux2/outputs/syn_coco_images"))

    dataset = COCODataset(
        root_dir=coco_train_path,
        annotation_file=annotation_file,
        transform=None,
        normalize=True,
    )

    name_to_cls_idx = build_name_to_cls_idx(dataset)

    anno_out = output_dir / "coco_synthetic_image_attributes.json"
    coco_attr_data = init_or_load_coco_attr_json(anno_out, dataset)

    model_name: str = "flux.2-dev"
    debug_mode: bool = False
    cpu_offloading: bool = True
    torch_device = torch.device("cuda")

    mistral = load_mistral_small_embedder()
    model = load_flow_model(
        model_name,
        debug_mode=debug_mode,
        device="cpu" if cpu_offloading else torch_device,
    )
    ae = load_ae(model_name)
    ae.eval()
    mistral.eval()

    for i in range(15,35):
        img, bboxes, masks, category_ids, mlc_vector, anns, idx = dataset[i]

        pil_img = Image.fromarray((img.numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
        width, height = pil_img.size

        selection = select_class_with_fewest_instances_and_area(mlc_vector, anns, dataset)
        if selection is None:
            pos = torch.nonzero(mlc_vector).squeeze().tolist()
            if isinstance(pos, int):
                pos = [pos]
            cls_idx = pos[0]
            coco_cat_id = int(dataset.cat_ids[cls_idx])
            label_name = dataset.get_class_name(cls_idx)
            candidate_anns = [a for a in anns if int(a["category_id"]) == coco_cat_id]
        else:
            cls_idx, coco_cat_id, label_name, candidate_anns, n_inst, tot_area = selection

        label_to_remove = dataset.get_class_name(cls_idx)
        label_to_add = COCO_REPLACE_MAP[label_to_remove]
        add_cls_idx = name_to_cls_idx[label_to_add]

        prompt = build_prompt_remove_and_add_coco(label_to_remove, label_to_add)

        # file names
        orig_file = f"{idx}_orig_.png"
        syn_file = f"{idx}_removed{label_to_remove}_add{label_to_add}_syn.png"

        orig_image_path = output_dir / orig_file
        gen_image_path = output_dir / syn_file

        # generate synthetic
        with tempfile.TemporaryDirectory() as tmpdirname:
            image_path = Path(tmpdirname) / "input_image.png"
            pil_img.save(image_path)

            gen_img = generate_image(
                prompt=prompt,
                input_images=f"{str(image_path)}",
                match_image_size=0,
                num_steps=40,
                guidance=3.0,
                torch_device=torch_device,
                mistral=mistral,
                model=model,
                ae=ae,
                seed=SEED,
            )

        # save images
        pil_img.save(orig_image_path)
        gen_img.save(gen_image_path)

        # add COCO-like records
        orig_id = int(idx)
        syn_id = int(idx) + 1_000_000_000

        add_image_record(coco_attr_data, orig_id, orig_file, width, height)
        add_image_record(coco_attr_data, syn_id, syn_file, width, height)

        # original labels unchanged
        orig_attr_entries = mlc_to_attr_entries(orig_id, mlc_vector, dataset)

        # synthetic labels: only remove + add
        syn_mlc = flip_remove_add_in_mlc(mlc_vector, remove_cls_idx=cls_idx, add_cls_idx=add_cls_idx)
        syn_attr_entries = mlc_to_attr_entries(syn_id, syn_mlc, dataset)

        append_attr_entries(coco_attr_data, orig_attr_entries)
        append_attr_entries(coco_attr_data, syn_attr_entries)

        # persist after every sample (crash-safe)
        atomic_write_json(anno_out, coco_attr_data)

        torch.cuda.empty_cache()



if __name__ == "__main__":
    main()
