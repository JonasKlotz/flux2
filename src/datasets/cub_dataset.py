import csv
import os
import random
import sys

import pandas as pd
import rootutils
import torch
from einops import rearrange
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
import tempfile

# Set up project root
project_root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)

from flux2.sampling import encode_image_refs, batched_prc_txt, batched_prc_img, get_schedule, denoise, scatter_ids
from flux2.util import FLUX2_MODEL_INFO, load_mistral_small_embedder, load_flow_model, load_ae


OPPOSITE_ATTR = {
    # bill length
    "has_bill_length::shorter_than_head": "has_bill_length::longer_than_head",
    "has_bill_length::longer_than_head": "has_bill_length::shorter_than_head",
    "has_bill_length::about_the_same_as_head": "has_bill_length::longer_than_head",

    # eye color
    "has_eye_color::black": "has_eye_color::red",

    # patterns
    "has_back_pattern::solid": "has_back_pattern::striped",
    "has_belly_pattern::solid": "has_belly_pattern::striped",
    "has_breast_pattern::solid": "has_breast_pattern::striped",
    "has_tail_pattern::solid": "has_tail_pattern::striped",

    # common colors
    "has_under_tail_color::black": "has_under_tail_color::white",
    "has_upper_tail_color::black": "has_upper_tail_color::white",
    "has_upperparts_color::black": "has_upperparts_color::white",
    "has_belly_color::white": "has_belly_color::black",
    "has_underparts_color::white": "has_underparts_color::black",
    "has_breast_color::white": "has_breast_color::black",
    "has_throat_color::white": "has_throat_color::black",

    "has_underparts_color::yellow": "has_underparts_color::blue",
    "has_breast_color::yellow": "has_breast_color::blue",
    "has_throat_color::yellow": "has_throat_color::blue",

    "has_primary_color::blue": "has_primary_color::yellow",
    "has_nape_color::blue": "has_nape_color::yellow",

    # shapes
    "has_shape::duck-like": "has_shape::perching-like",
    "has_shape::perching-like": "has_shape::duck-like",

    # bill shape
    "has_bill_shape::spatulate": "has_bill_shape::pointed",
    "has_bill_shape::all-purpose": "has_bill_shape::hooked",

    # size
    "has_size::medium_(9_-_16_in)": "has_size::small_(5_-_9_in)",
}

ATTR_VALUE_TO_TEXT = {
    # bill length
    "shorter_than_head": "clearly shorter than the head",
    "about_the_same_as_head": "about the same length as the head",
    "longer_than_head": "clearly longer than the head",

    # common colors
    "black": "black",
    "white": "white",
    "red": "red",
    "yellow": "yellow",
    "blue": "blue",
    "brown": "brown",

    # patterns
    "solid": "solid (no visible pattern)",
    "striped": "striped (strong visible stripes)",

    # shapes
    "duck-like": "duck-like",
    "perching-like": "perching-like",

    # bill shapes
    "spatulate": "spatulate (broad, spoon-like)",
    "all-purpose": "all-purpose",
    "pointed": "pointed and narrow",
    "hooked": "hooked",

    # sizes (verbatim is acceptable)
    "medium_(9_-_16_in)": "medium-sized (9 to 16 inches)",
    "small_(5_-_9_in)": "small-sized (5 to 9 inches)",
}


def split_attr(a: str):
    if "::" not in a:
        return a, ""
    k, v = a.split("::", 1)
    return k, v


def describe_attr(a: str) -> str:
    k, v = split_attr(a)
    v_txt = ATTR_VALUE_TO_TEXT.get(v, v.replace("_", " "))
    k_txt = k.replace("has_", "").replace("_", " ")
    return f"{k_txt} = {v_txt}"


def build_prompt(anchor_attr: str, target_attr: str) -> str:
    # Strong, explicit counterfactual edit instruction with strict preservation constraints.
    return (
        "Edit the input bird photograph. " # add species name?
        f"Strongly change exactly one attribute: replace {describe_attr(anchor_attr)} with {describe_attr(target_attr)}. "
        "Make the change unambiguous and clearly visible. "
        "Preserve the bird identity, species appearance, pose, scale, viewpoint, background, lighting, "
        "and all other colors and patterns. "
        "Do not add or remove objects. Do not change anything except the specified attribute."
    )

from scripts.cli import DEFAULTS, parse_key_values, apply_updates, print_config


class CUBDataset(Dataset):
    def __init__(
        self,
        root,
        split="train",  # "train", "val", or "test"
        transform=None,
        return_segmentation=False,
        val_split=0.15,
        seed=42,
    ):
        self.root = root
        self.transform = transform
        self.return_segmentation = return_segmentation

        # Load images, split flags, and class labels
        images_txt = os.path.join(root, "images.txt")
        split_txt = os.path.join(root, "train_test_split.txt")
        labels_txt = os.path.join(root, "image_class_labels.txt")
        class_names_txt = os.path.join(root, "classes.txt")
        class_attributes_txt = os.path.join(
            root, "attributes", "class_attribute_labels_continuous.txt"
        )
        attributes_names_txt = os.path.join(root, "attributes", "attributes.txt")

        images = pd.read_csv(images_txt, sep=" ", names=["img_id", "filepath"])
        split_flags = pd.read_csv(split_txt, sep=" ", names=["img_id", "is_train"])
        labels = pd.read_csv(labels_txt, sep=" ", names=["img_id", "class_id"])
        self.class_names = pd.read_csv(
            class_names_txt, sep=" ", names=["class_id", "class_name"]
        )
        self.class_names["class_name"] = self.class_names["class_name"].apply(
            lambda s: s.split(".", 1)[1]
        )
        self.attribute_names = pd.read_csv(
            attributes_names_txt, sep=" ", header=None, names=["attr_id", "attr_name"]
        )

        self.class_attributes = pd.read_csv(class_attributes_txt, sep=" ", header=None)
        df = images.merge(split_flags, on="img_id").merge(labels, on="img_id")

        # official train or test partition
        if split in ("train", "val"):
            df = df[df["is_train"] == 1].reset_index(drop=True)
        else:
            df = df[df["is_train"] == 0].reset_index(drop=True)

        # 2 If we need a train/val sub‐split, do it here
        if split in ("train", "val"):
            # stratify by class_id
            train_idx, val_idx = train_test_split(
                df.index.tolist(),
                test_size=val_split,
                stratify=df["class_id"].tolist(),
                random_state=seed,
            )
            if split == "train":
                chosen = train_idx
            else:
                chosen = val_idx
        else:
            # test split: use all
            chosen = df.index.tolist()

        # store only the rows we will actually serve
        self.data = df.loc[chosen].reset_index(drop=True)
        # keep a mapping so that __getitem__ can map new idx -> original position in df
        self.indices = chosen

        # 3 build attribute‐presence matrix exactly as before
        attr_names_txt = os.path.join(root, "attributes", "attributes.txt")
        self.attr_map = {}
        with open(attr_names_txt, "r") as f:
            for line in f:
                aid, name = line.strip().split(" ", 1)
                self.attr_map[int(aid)] = name

        attr_labels_txt = os.path.join(root, "attributes", "image_attribute_labels.txt")
        records = []
        with open(attr_labels_txt, "r") as f:
            reader = csv.reader(f, delimiter=" ", skipinitialspace=True)
            for row in reader:
                if len(row) != 5:
                    continue
                img_id, a_id, present, _, _ = row
                records.append((int(img_id), self.attr_map[int(a_id)], int(present)))
        attr_df = pd.DataFrame(
            records, columns=["img_id", "attribute_name", "is_present"]
        )
        self.all_info_df = df.merge(attr_df, on="img_id")
        # drop filepath and is_train for the all_info_df
        self.all_info_df = self.all_info_df.drop(columns=["filepath", "is_train"])
        # replace class_id with class_name from class_names df
        mapping = self.class_names.set_index("class_id")["class_name"]
        self.all_info_df["class_name"] = self.all_info_df["class_id"].map(mapping)

        matrix = attr_df.pivot(
            index="img_id", columns="attribute_name", values="is_present"
        ).fillna(0)
        self.attr_matrix = matrix

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # support both single int and list of ints
        if isinstance(idx, (list, tuple)):
            return [self[i] for i in idx]

        # map dataset‐local idx back to the original row in self.data
        row = self.data.iloc[idx]
        img_id = row["img_id"]
        img_path = os.path.join(self.root, "images", row["filepath"])

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        # attributes vector
        attrs = torch.tensor(self.attr_matrix.loc[img_id].values, dtype=torch.float32)

        # one‐hot class label
        label = torch.zeros(200, dtype=torch.float32)
        label[row["class_id"] - 1] = 1.0

        if self.return_segmentation:
            seg_file = row["filepath"].replace(".jpg", ".png")
            seg_path = os.path.join(self.root, "segmentations", seg_file)
            seg = Image.open(seg_path).convert("L")
            seg = transforms.ToTensor()(seg)
            return img, label, attrs, seg, idx

        return img, label, attrs, idx


    def label_to_class_name(self, label_tensor):
        class_idx = torch.argmax(label_tensor).item() + 1
        class_name = self.class_names[self.class_names["class_id"] == class_idx][
            "class_name"
        ].values[0]
        return class_name

    def attributes_to_names(self, attrs_tensor, threshold=0.5):
        # Find indices where attribute is present (value == 1) and map to names
        present_idx = (attrs_tensor == 1).nonzero(as_tuple=True)[0].tolist()
        return [
            self.attribute_names[self.attribute_names["attr_id"] == i + 1]["attr_name"].values[0]
            for i in present_idx
        ]


import numpy as np

def _jaccard(u, v, eps=1e-9):
    inter = np.logical_and(u == 1, v == 1).sum()
    union = np.logical_or(u == 1, v == 1).sum()
    return float(inter) / float(union + eps)

def get_prototypical_images_for_class(
    CUB_dataset,
    class_id,
    top_k=10,
    top_m_attrs=40,
    require_top_r=0,
):
    df = CUB_dataset.data
    df_c = df[df["class_id"] == class_id].reset_index(drop=False)

    # class prototype p_j in [0,1]
    p = (CUB_dataset.class_attributes.iloc[class_id - 1].values.astype(float)) / 100.0

    # rank attribute IDs by class likelihood
    attr_ids_sorted = np.argsort(-p) + 1   # now 1..312

    # anchor attribute (ID and name)
    anchor_attr_id = int(attr_ids_sorted[0])
    anchor_attr = CUB_dataset.attr_map[anchor_attr_id]

    # relevant attribute IDs (exclude anchor from similarity set)
    relevant_attr_ids = [
        int(aid) for aid in attr_ids_sorted[:top_m_attrs]
        if aid != anchor_attr_id
    ]

    # optionally require additional top attributes
    required_attr_ids = [anchor_attr_id] + list(attr_ids_sorted[1:1 + require_top_r])

    relevant_attr_names = [CUB_dataset.attr_map[aid] for aid in relevant_attr_ids]
    required_attr_names = [CUB_dataset.attr_map[aid] for aid in required_attr_ids]

    # collect candidates
    candidates = []
    vectors = []

    for j in range(len(df_c)):
        img_id = int(df_c.loc[j, "img_id"])
        a = CUB_dataset.attr_matrix.loc[img_id]

        # enforce required attributes
        if any(int(a[name]) != 1 for name in required_attr_names):
            continue

        vec = a[relevant_attr_names].values.astype(np.uint8)
        candidates.append((j, img_id))
        vectors.append(vec)

    if len(candidates) == 0:
        return [], anchor_attr, relevant_attr_names, required_attr_names

    X = np.stack(vectors, axis=0)

    # centroid as per-attribute mode
    centroid = (X.mean(axis=0) >= 0.5).astype(np.uint8)

    # medoid initialization
    sims = np.array([_jaccard(X[i], centroid) for i in range(X.shape[0])])
    seed = int(sims.argmax())

    selected = [seed]
    selected_set = {seed}

    # greedy cohesion maximization
    while len(selected) < min(top_k, X.shape[0]):
        best_i = None
        best_score = None
        for i in range(X.shape[0]):
            if i in selected_set:
                continue
            score = sum(_jaccard(X[i], X[s]) for s in selected) / len(selected)
            if best_score is None or score > best_score:
                best_score = score
                best_i = i
        selected.append(best_i)
        selected_set.add(best_i)

    # return dataset indices
    results = []
    for i in selected:
        j, img_id = candidates[i]
        dataset_idx = int(df_c.loc[j, "index"])
        cohesion = float(sum(_jaccard(X[i], X[s]) for s in selected) / len(selected))
        results.append((dataset_idx, img_id, cohesion))

    return results, anchor_attr, relevant_attr_names, required_attr_names


from pathlib import Path
import tempfile
from PIL import Image

def to_pil(img):
    # If CUB_dataset returns a PIL.Image already, this is a no-op.
    if isinstance(img, Image.Image):
        return img
    # If it is a torch tensor in CHW, convert it.
    if torch.is_tensor(img):
        x = img.detach().cpu()
        if x.ndim == 3 and x.shape[0] in (1, 3):  # CHW
            x = x.permute(1, 2, 0)
        # assume [0,1] float
        x = (x.clamp(0, 1) * 255).to(torch.uint8).numpy()
        return Image.fromarray(x)
    raise TypeError(f"Unsupported image type: {type(img)}")


def generate_image(
    model_name: str = "flux.2-dev",
    prompt: str | None = None,
    debug_mode: bool = False,
    cpu_offloading: bool = True,
    torch_device=None,
    mistral=None,
    model=None,
    ae=None,
    **overwrite,):


    # API client will be initialized lazily when needed

    cfg = DEFAULTS.copy()
    changes = [f"{key}={value}" for key, value in overwrite.items()]
    updates = parse_key_values(" ".join(changes))
    apply_updates(cfg, updates)
    if prompt is not None:
        cfg.prompt = prompt
    print_config(cfg)
    img = None

    # Load input images first to potentially match dimensions
    img_ctx = [Image.open(input_image) for input_image in cfg.input_images]

    # Apply match_image_size if specified
    width = cfg.width
    height = cfg.height
    if cfg.match_image_size is not None:
        if cfg.match_image_size < 0 or cfg.match_image_size >= len(img_ctx):
            print(
                f"  ! match_image_size={cfg.match_image_size} is out of range (0-{len(img_ctx)-1})",
                file=sys.stderr,
            )
            print(f"  ! Using default dimensions: {width}x{height}", file=sys.stderr)
        else:
            ref_img = img_ctx[cfg.match_image_size]
            width, height = ref_img.size
            print(f"  Matched dimensions from image {cfg.match_image_size}: {width}x{height}")

    seed = cfg.seed if cfg.seed is not None else random.randrange(2**31)
    dir = Path("output")
    dir.mkdir(exist_ok=True)
    output_name = dir / f"sample_{len(list(dir.glob('*')))}.png"

    with torch.no_grad():
        ref_tokens, ref_ids = encode_image_refs(ae, img_ctx)

        if cfg.upsample_prompt_mode == "local":
            # Use local model for upsampling
            upsampled_prompts = mistral.upsample_prompt(
                [cfg.prompt], img=[img_ctx] if img_ctx else None
            )
            prompt = upsampled_prompts[0] if upsampled_prompts else cfg.prompt
        else:
            # upsample_prompt_mode == "none" or invalid value
            prompt = cfg.prompt

        print("Generating with prompt: ", prompt)

        ctx = mistral([prompt]).to(torch.bfloat16)
        ctx, ctx_ids = batched_prc_txt(ctx)

        if cpu_offloading:
            mistral = mistral.cpu()
            torch.cuda.empty_cache()
            model = model.to(torch_device)

        # Create noise
        shape = (1, 128, height // 16, width // 16)
        generator = torch.Generator(device="cuda").manual_seed(seed)
        randn = torch.randn(shape, generator=generator, dtype=torch.bfloat16, device="cuda")
        x, x_ids = batched_prc_img(randn)

        timesteps = get_schedule(cfg.num_steps, x.shape[1])
        x = denoise(
            model,
            x,
            x_ids,
            ctx,
            ctx_ids,
            timesteps=timesteps,
            guidance=cfg.guidance,
            img_cond_seq=ref_tokens,
            img_cond_seq_ids=ref_ids,
        )
        x = torch.cat(scatter_ids(x, x_ids)).squeeze(2)
        x = ae.decode(x).float()
        # x = embed_watermark(x)

        if cpu_offloading:
            model = model.cpu()
            torch.cuda.empty_cache()
            mistral = mistral.to(torch_device)

    x = x.clamp(-1, 1)
    x = rearrange(x[0], "c h w -> h w c")

    img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
    return img


def iterate_cub():


    data_dir = "/data/jonas/CUB"
    CUB_dataset = CUBDataset(
        data_dir,
        split="train",
        transform=None,
        return_segmentation=False,
    )

    model_name: str = "flux.2-dev"
    debug_mode: bool = False
    cpu_offloading: bool = True
    assert model_name.lower() in FLUX2_MODEL_INFO, (
        f"{model_name} is not available, choose from {FLUX2_MODEL_INFO.keys()}"
    )

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

    out_dir = Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Minimal verbalization to make prompts clear and grounded.
    # For unknown values, fall back to the raw attribute string.


    for class_id in range(1, 201):
        results, anchor_attr, relevant_attr_names, required_attr_names = get_prototypical_images_for_class(
            CUB_dataset,
            class_id=class_id,
            top_k=5,
            top_m_attrs=50,
            require_top_r=0,
        )

        class_name = CUB_dataset.class_names[CUB_dataset.class_names["class_id"] == class_id]["class_name"].values[0]
        print("class_id:", class_id)
        print("class_name:", class_name)
        print("anchor_attr:", anchor_attr)
        print("n_selected:", len(results))

        # Replace anchor with its opposite; skip if not mappable.
        if anchor_attr not in OPPOSITE_ATTR:
            print("Skipping class: anchor_attr has no opposite mapping:", anchor_attr)
            continue

        target_attr = OPPOSITE_ATTR[anchor_attr]
        prompt = build_prompt(anchor_attr, target_attr)
        print("target_attr:", target_attr)
        print("prompt:", prompt)

        for dataset_idx, img_id, cohesion in results:
            row = CUB_dataset.data.iloc[dataset_idx]
            print(f"  img_id: {img_id}, filepath: {row['filepath']}, cohesion: {cohesion:.4f}")

            img, label, attrs, idx = CUB_dataset[dataset_idx]
            pil = to_pil(img).convert("RGB")

            with tempfile.TemporaryDirectory() as td:
                p = Path(td) / f"cub_{dataset_idx}.png"
                pil.save(p)

                # Use a deterministic per-image seed for reproducibility.
                seed = int(img_id)

                gen_img = generate_image(
                    prompt=prompt,
                    input_images=str(p),
                    match_image_size=0,
                    num_steps=40,
                    guidance=3.0,
                    torch_device=torch_device,
                    mistral=mistral,
                    model=model,
                    ae=ae,
                    seed=seed,
                )

            torch.cuda.empty_cache()

            # Optional quick visual sanity check.
            # pil.show()
            # gen_img.show()

            safe_class = str(class_name).replace("/", "_")
            out_orig = out_dir / f"cub_class{class_id}_{safe_class}_img{img_id}_{anchor_attr.replace('::','_')}_orig.png"
            out_gen = out_dir / f"cub_class{class_id}_{safe_class}_img{img_id}_{anchor_attr.replace('::','_')}__TO__{target_attr.replace('::','_')}_gen.png"

            pil.save(out_orig, quality=95, subsampling=0)
            gen_img.save(out_gen, quality=95, subsampling=0)

            print(f"Saved {out_orig}")
            print(f"Saved {out_gen}")




if __name__ == "__main__":
    iterate_cub()