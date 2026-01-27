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
from local_datasets.cub_dataset import CUBDataset
from prepare_image_generation import family
import lightning

# fix all seeds for reproducibility
SEED = 42
lightning.seed_everything(SEED)

ATTR_FALLBACK = {
    "has_breast_pattern::spotted": "has_breast_pattern::striped",
}

import argparse

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--start_idx", type=int, default=1, help="1-based inclusive")
    p.add_argument("--end_idx", type=int, default=None, help="1-based inclusive")
    p.add_argument("--shard_name", type=str, default=None, help="optional identifier for filenames")
    return p.parse_args()


def main():
    cub_root = Path("/home/jonas/PycharmProjects/flux2/outputs/syn_cub_dataset")
    out_images_root = ensure_dir(cub_root / "synthetic_images")
    out_attr_root = ensure_dir(cub_root / "attributes")
    reference_image_paths = cub_root / "reference_images"
    # load all dir names in reference_image_paths
    class_dirs = [
        d.name for d in reference_image_paths.iterdir() if d.is_dir()
    ]
    reference_image_files = {}
    for class_dir in class_dirs:
        class_path = reference_image_paths / class_dir
        image_files = list(class_path.glob("*.jpg"))
        reference_image_files[class_dir] = image_files



    cub_base = CUBDataset(
        cub_root,
        split="all",
        transform=None,
        return_segmentation=False,
    )

    # attribute name -> column index in attrs vector
    cub_attr_names = cub_base.attribute_names
    cub_attr_to_idx = {}
    for idx, name in cub_base.attr_map.items():
        cub_attr_to_idx[name] = idx


    sub = load_dataset("Jessica-bader/SUB")
    sub_attr_names = sub["test"].features["attr_label"].names
    sub_attr_names = [name.replace("--", "::") for name in sub_attr_names]



    replacement_attrs = cub_root /"images/replacement_attrs.txt"
    replacement_attr_dict = load_replacement_attrs(replacement_attrs)

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

    # build updated image_attribute_labels.txt rows
    # format: img_id attr_id present certainty time
    image_attr_rows = []
    next_img_id = 1  # will assign ids for saved images in this script

    for idx in range(len(cub_base)):
        print(f"Processing image {idx + 1} / {len(cub_base)}")
        img, label, attrs, key = cub_base[idx]
        class_name = cub_base.label_to_class_name(label)
        image_path = cub_base.get_image_path(idx)
        img.show()
        attrs_to_replace = replacement_attr_dict.get(class_name)
        for attr_to_replace in attrs_to_replace:

            # choose a new attribute within the same family (based on SUB list)
            fam = family(attr_to_replace)
            candidate_attr_names = [
                a for a in sub_attr_names
                if family(a) == fam and a != attr_to_replace
            ]

            if len(candidate_attr_names) == 0:
                candidate_attr_names = [ATTR_FALLBACK.get(attr_to_replace)]

            for new_attr_name in candidate_attr_names:
                print(f" - Replacing attribute '{attr_to_replace}' with '{new_attr_name}'")
                # attribute to replace must exist in CUB
                if attr_to_replace not in cub_attr_to_idx:
                    raise RuntimeError(
                        f"Attribute to replace '{attr_to_replace}' does not exist in CUB attribute space."
                    )

                # new attribute must exist in CUB, otherwise fallback or fail
                if new_attr_name not in cub_attr_to_idx:
                    fallback = ATTR_FALLBACK.get(attr_to_replace)
                    if fallback is None:
                        raise RuntimeError(
                            f"No valid replacement found for '{attr_to_replace}'. "
                            f"Candidate '{new_attr_name}' not in CUB and no fallback defined."
                        )
                    if fallback not in cub_attr_to_idx:
                        raise RuntimeError(
                            f"Fallback attribute '{fallback}' for '{attr_to_replace}' "
                            f"does not exist in CUB attribute space."
                        )
                    new_attr_name = fallback

                reference_images_for_new_attr = reference_image_files.get(new_attr_name)
                # sample one reference image if multiple are available
                random_idx = torch.randint(0, len(reference_images_for_new_attr), (1,)).item()
                ref_image_path = Path(reference_images_for_new_attr[random_idx])
                # load reference image
                ref_img = Image.open(ref_image_path).convert("RGB")

                prompt = build_prompt(attr_to_replace, new_attr_name, class_name, family=fam)
                # print(prompt)

                # save paths: classname/attrthatwaschanged/{oldname}_syn and _orig
                old_stem = Path(image_path).stem  # safe even if key is numeric; then stem==key
                safe_attr_dir = attr_to_replace.replace("/", "_")
                save_dir = ensure_dir(out_images_root / class_name / f"{safe_attr_dir}_to_{new_attr_name}")

                out_orig_path = save_dir / f"{old_stem}_orig.png"
                out_syn_path = save_dir / f"{old_stem}_syn.png"
                out_ref_path = save_dir / f"{old_stem}_ref.png"

                # generate synthetic


                gen_img = generate_image(
                    prompt=prompt,
                    input_images=f"{str(image_path)},{str(ref_image_path)}",
                    match_image_size=0,
                    num_steps=40,
                    guidance=3.0,
                    torch_device=torch_device,
                    mistral=mistral,
                    model=model,
                    ae=ae,
                    seed=SEED,
                )

                torch.cuda.empty_cache()

                # save images
                gen_img.save(out_syn_path)
                img.save(out_orig_path)
                # also save the reference image used
                ref_img.save(out_ref_path)

                update_new_attributes(attr_to_replace,
                                      attrs,
                                      cub_attr_names,
                                      cub_attr_to_idx,
                                      image_attr_rows,
                                      new_attr_name,
                                      next_img_id)

                print(f"Saved: {out_orig_path}")
                print(f"Saved: {out_syn_path}")


    # finally write updated attributes file (overwrite)
    out_file = out_attr_root / "syn_image_attribute_labels.txt"
    pd.DataFrame(image_attr_rows).to_csv(out_file, sep=" ", index=False, header=False)
    print(f"Wrote updated attributes to: {out_file}")


def load_reference_image(new_attr_name: str | Any, reference_image_files: dict[Any, Any]) -> ImageFile:
    reference_images_for_new_attr = reference_image_files.get(new_attr_name)
    # sample one reference image if multiple are available
    random_idx = torch.randint(0, len(reference_images_for_new_attr), (1,)).item()
    ref_image_path = Path(reference_images_for_new_attr[random_idx])
    print(f"   Using reference image for new attribute: {ref_image_path}")
    # load reference image
    ref_img = Image.open(ref_image_path).convert("RGB")
    return ref_img


def update_new_attributes(attr_to_replace, attrs, cub_attr_names: list[Any], cub_attr_to_idx: dict[Any, Any],
                          image_attr_rows: list[Any], new_attr_name: str | Any, next_img_id: int):
    # build updated attribute vectors
    orig_attrs = attrs.clone()
    syn_attrs = attrs.clone()

    old_i = cub_attr_to_idx[attr_to_replace]
    new_i = cub_attr_to_idx[new_attr_name]

    # enforce replacement: old off, new on
    syn_attrs[old_i] = 0.0
    syn_attrs[new_i] = 1.0

    # write attributes for BOTH images (orig and syn) into updated file
    # assign new sequential ids
    orig_img_id = next_img_id
    syn_img_id = next_img_id + 1
    next_img_id += 2

    certainty = 1
    time_val = 0.0

    for a_id in range(len(cub_attr_names)):
        present = int(orig_attrs[a_id].item() >= 0.5)
        image_attr_rows.append((orig_img_id, a_id + 1, present, certainty, time_val))

    for a_id in range(len(cub_attr_names)):
        present = int(syn_attrs[a_id].item() >= 0.5)
        image_attr_rows.append((syn_img_id, a_id + 1, present, certainty, time_val))


def load_replacement_attrs(replacement_attrs: Path) -> dict[Any, Any]:
    replacement_attr_dict = {}
    with open(replacement_attrs, "r") as f:
        for line in f:
            line = line.strip().split(";")
            key, value = line[0], eval(line[1])
            replacement_attr_dict[key] = value
    return replacement_attr_dict


if __name__ == "__main__":
    main()
