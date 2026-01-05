import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from einops import rearrange

from flux2.sampling import encode_image_refs, batched_prc_txt, batched_prc_img, get_schedule, denoise, scatter_ids
from flux2.util import FLUX2_MODEL_INFO, load_mistral_small_embedder, load_flow_model, load_ae
from local_datasets.cub_dataset import CUBDataset
from scripts.cli import DEFAULTS, parse_key_values, apply_updates, print_config

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
    k, v = a.replace("--","::").split("::", 1)
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
    prompt: str | None = None,
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


def _jaccard(u, v, eps=1e-9):
    inter = np.logical_and(u == 1, v == 1).sum()
    union = np.logical_or(u == 1, v == 1).sum()
    return float(inter) / float(union + eps)
