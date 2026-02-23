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
import json
import os
from pathlib import Path
from typing import Dict, Any, List
import numpy as np


import json
from pathlib import Path
from typing import Iterable, List
import pandas as pd

import argparse


import torch




def present_indices(mlc_vector: torch.Tensor) -> List[int]:
    # robust for any shape (C,) or (1,C) etc.
    v = mlc_vector.detach().cpu().view(-1)
    return torch.nonzero(v, as_tuple=False).view(-1).tolist()

def labels_to_json(indices: Iterable[int]) -> str:
    # stable, parseable, no CSV delimiter issues
    return json.dumps(list(map(int, indices)), ensure_ascii=False)

from filelock import FileLock
import pandas as pd

def append_metadata_row_locked(csv_path, row):
    lock = FileLock(str(csv_path) + ".lock")
    with lock:
        write_header = not csv_path.exists()
        pd.DataFrame([row]).to_csv(
            csv_path,
            mode="a",
            header=write_header,
            index=False,
        )


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

        key = (n, -area_sum)  # negative area to maximize it
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

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start_idx", type=int, default=0, help="Inclusive start index")
    ap.add_argument("--end_idx", type=int, default=None, help="Exclusive end index (default: len(dataset))")
    return ap.parse_args()



def main():
    args = parse_args()
    coco_path = "/data/jonas/COCO"
    coco_train_path = os.path.join(coco_path, "train2017")
    annotation_file = os.path.join(
        coco_path, "annotations_trainval2017/annotations/instances_train2017.json"
    )
    output_dir = ensure_dir("/home/jonas/PycharmProjects/flux2/outputs/syn_coco_images")
    output_image_dir = ensure_dir(output_dir / "images")

    dataset = COCODataset(
        root_dir=coco_train_path,
        annotation_file=annotation_file,
        transform=None,
        normalize=True,
    )

    # choose a run-specific name so multiple runs do not overwrite
    metadata_csv = output_dir / "metadata.csv"
    
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

    start_idx = int(args.start_idx)
    end_idx = len(dataset) if args.end_idx is None else int(args.end_idx)

    # clamp to valid range
    start_idx = max(0, min(start_idx, len(dataset)))
    end_idx = max(0, min(end_idx, len(dataset)))


    for i in range(start_idx, end_idx):
        img, bboxes, masks, category_ids, mlc_vector, anns, idx = dataset[i]

        pil_img = Image.fromarray((img.numpy().transpose(1, 2, 0) * 255).astype(np.uint8))

        selection = select_class_with_fewest_instances_and_area(mlc_vector, anns, dataset)
        if selection is None:
            pos = torch.nonzero(mlc_vector).squeeze().tolist()
            if isinstance(pos, int):
                pos = [pos]
            cls_idx = pos[0]

        else:
            cls_idx, _, _, _, _, _ = selection

        label_to_remove = dataset.get_class_name(cls_idx)
        # label_to_add = COCO_REPLACE_MAP[label_to_remove]
        # add_cls_idx = name_to_cls_idx[label_to_add]

        # prompt = build_prompt_remove_and_add_coco(label_to_remove, label_to_add)
        prompt = build_prompt_remove_all_coco(label_to_remove)
        # file names
        orig_file = f"{idx}_orig.png"
        syn_file = f"{idx}_removed{label_to_remove}_syn.png"

        orig_image_path = output_image_dir / orig_file
        gen_image_path = output_image_dir / syn_file

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

        # synthetic labels: only remove
        syn_mlc =mlc_vector.clone()
        syn_mlc[cls_idx] = 0

        orig_idx_list = present_indices(mlc_vector)
        syn_idx_list = present_indices(syn_mlc)
        if len(orig_idx_list) == 0 or max(orig_idx_list) >= len(dataset.cat_ids):
            print(f"Warning: class index out of bounds for image {idx}, skipping metadata entry.")
            continue

        row = {
            "coco_idx": int(idx),
            "orig_path": str(orig_image_path),
            "orig_labels": labels_to_json(orig_idx_list),
            "syn_path": str(gen_image_path),
            "syn_labels": labels_to_json(syn_idx_list),
            "removed_cls_idx": int(cls_idx),
            "removed_class": str(label_to_remove),
        }
        append_metadata_row_locked(metadata_csv, row)

        torch.cuda.empty_cache()



if __name__ == "__main__":

    main()
