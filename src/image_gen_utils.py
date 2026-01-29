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
    return f"'{k_txt} is {v_txt}'"

def build_prompt(anchor_attr: str, target_attr: str, class_name: str, family:str) -> str:
    return (
        "Task: counterfactual single-attribute edit.\n"
        "Inputs: Image 1 is the base photograph to be edited. Image 2 is a reference exemplar for the target attribute only.\n"
        f"Base subject: a {class_name} bird in Image 1.\n\n"

        "Primary rule (hard priority): Preserve Image 1 exactly. Treat Image 1 as the ground-truth for identity and all pixels except the one permitted attribute.\n"
        "Reference rule (strictly limited): Use Image 2 only to understand the visual appearance of the target attribute. "
        "Do not copy any other details from Image 2.\n\n"

        "Allowed change (exactly one): "
        f"Replace {describe_attr(anchor_attr)} with {describe_attr(target_attr)} on the bird in Image 1. "
        "The change must be unambiguous, localized to the attribute region, and physically plausible.\n\n"

        "Invariances (must not change):\n"
        "1. Bird identity and species-specific appearance from Image 1.\n"
        "2. Pose, body shape, size, camera viewpoint, framing, and scale.\n"
        "3. Background, environment, and all non-bird objects.\n"
        "4. Lighting direction, intensity, shadows, and overall color grading.\n"
        "5. All other colors, patterns, textures, and markings on the bird, including those adjacent to the attribute.\n"
        "6. No additions, removals, or hallucinated objects.\n"
        # if family does not contain color attributes, we can add a prohibition against changing colors
        +
        ("" if "color" in family else "7. Do not change any colors on the bird, "
        "e.g. when we change the breast pattern to spotted or striped we want to keep the breast color from the bird in Image 1\n\n")
        +
        "Prohibitions (explicit negatives):\n"
        "1. Do not change the bird’s head shape, beak shape, eye shape, feather layout, or any markings unrelated to the specified attribute.\n"
        "2. Do not change the background or introduce new scenery.\n"
        "3. Do not import colors, patterns, or species traits from Image 2 except the target attribute appearance.\n\n"

        "Output requirement: one realistic photograph that matches Image 1 except for the single specified attribute change."
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
    # print_config(cfg)
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

        # print("Generating with prompt: ", prompt)

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

def _jaccard(u, v, eps=1e-9):
    inter = np.logical_and(u == 1, v == 1).sum()
    union = np.logical_or(u == 1, v == 1).sum()
    return float(inter) / float(union + eps)
