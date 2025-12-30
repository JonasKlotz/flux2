import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, Optional

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from pandas import DataFrame

from local_datasets.cub_dataset import CUBDataset


# -----------------------------
# helpers (given)
# -----------------------------
def family(attr: str) -> str:
    # "has_eye_color::black" -> "has_eye_color"
    return attr.replace("--","::").split("::", 1)[0] if isinstance(attr, str) else ""


def remove_family(attrs, banned_family: str):
    return [a for a in attrs if family(a[0]) != banned_family]


def keep_topk(attrs, k: int):
    return list(attrs[:k])


def save_bird_labels(bird_labels: Dict[Any, Any], path: Union[str, Path] = "bird_labels.pkl") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(bird_labels, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_bird_labels(path: Union[str, Path] = "bird_labels.pkl") -> Dict[Any, Any]:
    path = Path(path)
    if not path.exists():
        dataset = load_dataset("Jessica-bader/SUB")
        test_data = dataset["test"]
        features = dataset["test"].features
        bird_labels = {name:[] for name in features["bird_label"].names}
        for sample in test_data:
            attr_label = dataset["test"].features["attr_label"].int2str(sample["attr_label"])
            bird_label = dataset["test"].features["bird_label"].int2str(sample["bird_label"])
            if attr_label not in bird_labels[bird_label]:
                bird_labels[bird_label].append(attr_label)
        save_bird_labels(bird_labels)
        return bird_labels
    with path.open("rb") as f:
        return pickle.load(f)


def get_class_attrs(CUB_dataset: CUBDataset) -> DataFrame:
    class_attributes_df = CUB_dataset.class_attributes  # rows classes, cols attributes (after renaming)
    attributes_df = CUB_dataset.attribute_names  # attr_id attr_name - id starts at 1
    class_df = CUB_dataset.class_names  # class_id class_name - id starts at 1

    attr_names = attributes_df["attr_name"].to_numpy()
    class_names = class_df["class_name"].to_numpy()

    class_attributes_df.columns = attr_names
    class_attributes_df["class_name"] = class_names
    class_attributes_df.columns.name = "attribute"
    return class_attributes_df


# -----------------------------
# IO utilities
# -----------------------------
def ensure_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_df_if_exists(path: Union[str, Path]) -> Optional[pd.DataFrame]:
    p = Path(path)
    if not p.exists():
        return None
    try:
        if p.suffix.lower() in [".parquet"]:
            return pd.read_parquet(p)
        if p.suffix.lower() in [".csv"]:
            return pd.read_csv(p)
        if p.suffix.lower() in [".pkl", ".pickle"]:
            return pd.read_pickle(p)
        raise ValueError(f"Unsupported dataframe format: {p.suffix}")
    except Exception as e:
        raise RuntimeError(f"Failed to load cached dataframe at {p}: {e}") from e


def save_df(df: pd.DataFrame, path: Union[str, Path]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() == ".parquet":
        df.to_parquet(p, index=False)
    elif p.suffix.lower() == ".csv":
        df.to_csv(p, index=False)
    elif p.suffix.lower() in [".pkl", ".pickle"]:
        df.to_pickle(p)
    else:
        raise ValueError(f"Unsupported dataframe format: {p.suffix}")




def filter_cub_df_to_sub_birds(df: pd.DataFrame, birds_to_consider: List[str]) -> pd.DataFrame:
    return df[df["class_name"].isin(birds_to_consider)].copy()


def compute_topk_attributes_per_class(
    df: pd.DataFrame,
    topk: int = 5,
    class_col: str = "class_name",
) -> Dict[str, List[str]]:
    attribute_cols = [c for c in df.columns if c != class_col]

    class_to_top_attrs: Dict[str, List[str]] = {}
    for _, row in df.iterrows():
        class_name = row[class_col]
        values = row[attribute_cols].to_numpy(dtype=float)
        top_idx = np.argsort(values)[-topk:][::-1]
        class_to_top_attrs[class_name] = [(attribute_cols[i], values[i]) for i in top_idx]

    return class_to_top_attrs


def clean_top_attrs(
    class_to_top_attrs: Dict[str, List[str]],
    banned_family: str = "has_eye_color",
    keep_k: int = 3,
) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for cls, attrs in class_to_top_attrs.items():
        attrs2 = keep_topk(remove_family(attrs, banned_family), keep_k)
        if attrs2:
            out[cls] = attrs2
    return out


def _flatten_loaded_attrs(loaded_val: Any) -> List[str]:
    if loaded_val is None:
        return []
    if isinstance(loaded_val, list):
        return loaded_val
    if isinstance(loaded_val, dict):
        flat: List[str] = []
        for v in loaded_val.values():
            if isinstance(v, list):
                flat.extend(v)
        return flat
    return []


def match_loaded_attrs_to_topk_families(
    class_to_top_attrs_clean: Dict[str, List[str]],
    loaded: Dict[Any, Any],
) -> pd.DataFrame:
    # match (deduplicated):
    # dedup criterion: within each class, each loaded_attr should appear at most once.
    # if it matches multiple topk ranks (rare but possible if families repeat), keep best (lowest rank).
    rows = []
    seen_best_rank = {}  # key: (class_name, loaded_attr) -> best_rank

    for cls, top_attrs in class_to_top_attrs_clean.items():
        top_fams = [family(a[0]) for a in top_attrs]

        loaded_attrs = _flatten_loaded_attrs(loaded.get(cls, []))

        # remove duplicates already in loaded list (keeps order)
        loaded_attrs = list(dict.fromkeys(loaded_attrs))

        for la in loaded_attrs:
            laf = family(la)

            for rank, (ta, tf) in enumerate(zip(top_attrs, top_fams), start=1):
                if laf != tf:
                    continue

                key = (cls, la)
                prev_rank = seen_best_rank.get(key)

                # keep only the best (lowest) rank per (class, loaded_attr)
                if (prev_rank is None) or (rank < prev_rank):
                    seen_best_rank[key] = rank
                    rows.append(
                        {
                            "class_name": cls,
                            "loaded_attr": la,
                            "loaded_family": laf,
                            "topk_attr": ta,
                            "topk_family": tf,
                            "topk_rank": rank,
                        }
                    )

    matches_df = pd.DataFrame(
        rows,
        columns=["class_name", "loaded_attr", "loaded_family", "topk_attr", "topk_family", "topk_rank"],
    )

    # final dedup pass (defensive): keep best rank if duplicates remain
    if not matches_df.empty:
        matches_df = (
            matches_df.sort_values(["class_name", "loaded_attr", "topk_rank"])
            .drop_duplicates(subset=["class_name", "loaded_attr"], keep="first")
            .sort_values(["class_name", "topk_rank", "loaded_attr"])
            .reset_index(drop=True)
        )

    return matches_df


def compute_or_load_matches_df(
    cache_path: Union[str, Path],
    *,
    cub,
    bird_labels_path: Union[str, Path],
    sub,
    topk_raw: int = 5,
    banned_family: str = "has_eye_color",
    topk_clean: int = 3,
) -> pd.DataFrame:
    cache_path = Path(cache_path)
    # cached = load_df_if_exists(cache_path)
    # if cached is not None:
    #     return cached

    df_cub = get_class_attrs(cub)

    birds_to_consider = sub["test"].features["bird_label"].names
    df_cub = filter_cub_df_to_sub_birds(df_cub, birds_to_consider)

    class_to_top_attrs = compute_topk_attributes_per_class(df_cub, topk=topk_raw)
    class_to_top_attrs_clean = clean_top_attrs(
        class_to_top_attrs,
        banned_family=banned_family,
        keep_k=topk_clean,
    )

    loaded = load_bird_labels(bird_labels_path)
    matches_df = match_loaded_attrs_to_topk_families(class_to_top_attrs_clean, loaded)

    return matches_df



import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import pandas as pd


def save_reference_images_for_topk_attrs(
    matches_df: pd.DataFrame,
    cub_dataset: Any,
    out_root: Union[str, Path],
    max_per_attr: int = 100,
) -> pd.DataFrame:
    """
    Iterates over CUB dataset and saves images for each (class_name, topk_attr) pair
    into: out_root/class_name/reference_attribute/

    Returns a dataframe with columns:
      [class_name, topk_attr, image_src_path, image_dst_path]
    """
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    required_pairs = set(
        matches_df[["class_name", "topk_attr"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )

    counts: Dict[Tuple[str, str], int] = defaultdict(int)
    rows = []

    def all_done() -> bool:
        return all(counts[p] >= max_per_attr for p in required_pairs)

    for i in range(len(cub_dataset)):
        if all_done():
            break

        img, label, attrs, idx = cub_dataset[i]

        class_name = cub_dataset.label_to_class_name(label)

        needed_attrs = [a for (c, a) in required_pairs if c == class_name]
        if not needed_attrs:
            continue

        # recover image path exactly as in __getitem__
        row = cub_dataset.data.iloc[idx]
        image_path = Path(cub_dataset.root) / "images" / row["filepath"]
        if not image_path.exists():
            continue

        attribute_names = cub_dataset.attributes_to_names(attrs)

        for needed_attr in needed_attrs:
            key = (class_name, needed_attr)
            if counts[key] >= max_per_attr:
                continue

            if needed_attr[0] not in attribute_names:
                continue

            dst_dir = out_root / class_name / f"{needed_attr[0]}_{needed_attr[1]}"
            dst_dir.mkdir(parents=True, exist_ok=True)

            dst_path = dst_dir / f"{counts[key]:04d}_{image_path.name}"
            if not dst_path.exists():
                shutil.copy2(image_path, dst_path)

            counts[key] += 1
            rows.append(
                {
                    "class_name": class_name,
                    "topk_attr": needed_attr,
                    "image_src_path": str(image_path),
                    "image_dst_path": str(dst_path),
                }
            )

    return pd.DataFrame(
        rows,
        columns=["class_name", "topk_attr", "image_src_path", "image_dst_path"],
    )

def export_reference_images(
    matches_df: pd.DataFrame,
    cub_dataset: Any,
    assets_dir: Union[str, Path],

) -> pd.DataFrame:
    """
    Creates:
      assets_dir/reference_images/<class_name>/<topk_attr>/*.jpg
    Also saves an index dataframe for reproducibility.
    """
    assets_dir = Path(assets_dir)
    out_root = assets_dir / "reference_images"
    out_root.mkdir(parents=True, exist_ok=True)

    index_df = save_reference_images_for_topk_attrs(
        matches_df=matches_df,
        cub_dataset=cub_dataset,
        out_root=out_root,
    )

    return index_df

def print_reference_image_stats(ref_index_df: pd.DataFrame) -> None:
    if ref_index_df.empty:
        print("Reference index is empty.")
        return

    n_images = len(ref_index_df)
    n_classes = ref_index_df["class_name"].nunique()
    n_attrs = ref_index_df["topk_attr"].nunique()
    n_pairs = (
        ref_index_df[["class_name", "topk_attr"]]
        .drop_duplicates()
        .shape[0]
    )

    per_pair = (
        ref_index_df
        .groupby(["class_name", "topk_attr"])
        .size()
    )

    per_class = (
        ref_index_df
        .groupby("class_name")
        .size()
    )

    per_attr = (
        ref_index_df
        .groupby("topk_attr")
        .size()
    )

    print("Reference image statistics")
    print("--------------------------")
    print(f"Total images saved           : {n_images}")
    print(f"Unique classes               : {n_classes}")
    print(f"Unique reference attributes  : {n_attrs}")
    print(f"Class–attribute pairs        : {n_pairs}")
    print()
    print("Images per (class, attribute)")
    print(f"  min / mean / max            : "
          f"{per_pair.min()} / {per_pair.mean():.2f} / {per_pair.max()}")
    print()
    print("Images per class")
    print(f"  min / mean / max            : "
          f"{per_class.min()} / {per_class.mean():.2f} / {per_class.max()}")
    print()
    print("Images per attribute")
    print(f"  min / mean / max            : "
          f"{per_attr.min()} / {per_attr.mean():.2f} / {per_attr.max()}")
    print()
    print("Top 5 most frequent attributes:")
    print(per_attr.sort_values(ascending=False).head(5))
    print()
    print("Top 5 classes with most reference images:")
    print(per_class.sort_values(ascending=False).head(5))


# usage

#############################################################


def main():
    bird_labels_path = "bird_labels.pkl"

    out_dir = ensure_dir("/home/jonas/PycharmProjects/flux2/assets/")
    cache_path = out_dir / "matches_df.parquet"  # change suffix to .csv or .pkl if you prefer
    cub_data_dir = "/data/jonas/CUB"
    CUB_dataset = CUBDataset(
        cub_data_dir,
        split="train",
        transform=None,
        return_segmentation=False,
    )
    SUB_dataset = load_dataset("Jessica-bader/SUB")



    matches_df = compute_or_load_matches_df(
        cache_path=cache_path,
        cub=CUB_dataset,
        sub=SUB_dataset,
        bird_labels_path=bird_labels_path,
        topk_raw=5,
        topk_clean=3,

    )

    # after you computed/loaded matches_df (and still have CUB_dataset available)
    assets_dir = "/home/jonas/PycharmProjects/flux2/assets/"
    ref_index_df = export_reference_images(
        matches_df=matches_df,
        cub_dataset=CUB_dataset,
        assets_dir=assets_dir,
    )
    print(ref_index_df.head(10))




if __name__ == "__main__":
    main()

