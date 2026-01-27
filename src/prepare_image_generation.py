import glob
import os
import pickle
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Tuple, Union
from typing import List

import numpy as np
import pandas as pd
from datasets import load_dataset, tqdm, DatasetDict
from pandas import DataFrame

from io_utils import ensure_dir, save_df
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
    class_df = CUB_dataset.class_names  # class_id class_name - id starts at 1

    class_names = class_df["class_name"].to_numpy()

    class_attributes_df.columns = CUB_dataset.attribute_names
    class_attributes_df["class_name"] = class_names
    class_attributes_df.columns.name = "attribute"
    return class_attributes_df


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
    cub,
    bird_labels_path: Union[str, Path],
    sub,
    topk_raw: int = 5,
    banned_family: str = "has_eye_color",
    topk_clean: int = 3,
) -> pd.DataFrame:
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





def save_base_images_for_topk_attrs(
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
        .itertuples(index=False, name=None)
    )

    counts: Dict[Tuple[str, str], int] = defaultdict(int)
    rows = []

    def all_done() -> bool:
        return all(counts[p] >= max_per_attr for p in required_pairs)

    for i in tqdm(range(len(cub_dataset))):
        if all_done():
            print("All done!")
            break

        img, label, attrs, idx = cub_dataset[i]

        class_name = cub_dataset.label_to_class_name(label)

        needed_attrs = [a for (c, a) in required_pairs if c == class_name]
        if not needed_attrs:
            continue

        image_path = Path(cub_dataset.get_image_path(idx))
        if not image_path.exists():
            continue

        attribute_names = cub_dataset.attributes_to_names(attrs)

        for needed_attr in needed_attrs:
            key = (class_name, needed_attr)
            if counts[key] >= max_per_attr:
                continue

            if needed_attr[0] not in attribute_names:
                continue

            dst_dir = out_root / class_name / f"{needed_attr[0]}"
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



def write_replacement_attrs(base_img_out_dir: Path):
    dirs = glob.glob(str(base_img_out_dir / "*"))
    class_to_attrs = {d: glob.glob(str(Path(d) / "*")) for d in sorted(dirs)}

    # remove everything except dirname
    class_to_attrs = {Path(k).name: [Path(p).name for p in v] for k, v in class_to_attrs.items()}

    print(class_to_attrs)
    # save to txt
    with open(base_img_out_dir / "replacement_attrs.txt", "w") as f:
        for k, v in class_to_attrs.items():
            f.write(f"{k};{v}\n")



def write_base_images(CUB_dataset: CUBDataset, SUB_dataset: DatasetDict, base_img_out_dir: Path, bird_labels_path: str):
    matches_df = compute_or_load_matches_df(
        cub=CUB_dataset,
        sub=SUB_dataset,
        bird_labels_path=bird_labels_path,
        topk_raw=5,
        topk_clean=3,

    )

    ref_index_df = save_base_images_for_topk_attrs(
        matches_df=matches_df,
        cub_dataset=CUB_dataset,
        out_root=base_img_out_dir,
    )

    # save  df
    save_df(
        ref_index_df,
        Path(base_img_out_dir) / "base_images_index.csv"
    )
    print(ref_index_df.head(10))


def extract_label_information(cub_dataset: Any, out_dir:Path):
    out_images_root = ensure_dir(out_dir / "images")
    out_attr_root = ensure_dir(out_dir / "attributes")

    ref_df = pd.read_csv(out_images_root / "base_images_index.csv")
    # filter 'image_src_path' solumn for unique entries
    ref_df = ref_df.drop_duplicates(subset=["image_src_path"]).reset_index(drop=True)

    images_txt = []
    labels_txt = []
    split_txt = []
    image_attr_rows = []

    next_img_id = 1

    # attr_id 1..312 in the same order as cub.attribute_names / cub.attr_matrix columns
    num_attrs = len(cub_dataset.attribute_names)

    for _, row in ref_df.iterrows():
        src_path = row["image_src_path"]
        dst_path = row["image_dst_path"]

        info = cub_dataset.get_info_from_image_path(src_path)

        # destination filepath to store in images.txt: relative to new dataset images/
        dst_path = os.path.abspath(dst_path)
        dst_rel = os.path.relpath(dst_path, out_images_root)

        # ids and split
        images_txt.append((next_img_id, dst_rel))
        labels_txt.append((next_img_id, int(info["class_id"])))
        split_txt.append((next_img_id, 1))  # mark as train by convention

        # write per-image attribute rows in CUB format:
        # img_id attr_id is_present certainty time
        # we set certainty=1, time=0.0 (placeholders, but valid numeric fields)
        attrs = info["attributes"]  # float tensor of 0/1
        for a in range(num_attrs):
            present = int(attrs[a].item() >= 0.5)
            image_attr_rows.append((next_img_id, a + 1, present, 1, 0.0))

        next_img_id += 1

    # write core files
    pd.DataFrame(images_txt).to_csv(
        out_dir / "images.txt", sep=" ", index=False, header=False
    )
    pd.DataFrame(labels_txt).to_csv(
        out_dir / "image_class_labels.txt", sep=" ", index=False, header=False
    )
    pd.DataFrame(split_txt).to_csv(
        out_dir / "train_test_split.txt", sep=" ", index=False, header=False
    )

    # write attributes file
    pd.DataFrame(image_attr_rows).to_csv(
        out_attr_root / "image_attribute_labels.txt", sep=" ", index=False, header=False
    )

    print(f"Wrote {next_img_id - 1} samples.")
    print(f"Wrote {len(image_attr_rows)} attribute rows.")



#############################################################


def main():
    bird_labels_path = "local_datasets/bird_labels.pkl"
    assets_dir = "/home/jonas/PycharmProjects/flux2/assets"

    out_dir = ensure_dir("/home/jonas/PycharmProjects/flux2/outputs/syn_cub_dataset")
    base_img_out_dir = ensure_dir(out_dir / "images")

    cub_data_dir = "/data/jonas/CUB"
    CUB_dataset = CUBDataset(
        cub_data_dir,
        split="all",
        transform=None,
        return_segmentation=False,
    )
    SUB_dataset = load_dataset("Jessica-bader/SUB")

    # write_base_images(CUB_dataset, SUB_dataset, base_img_out_dir, bird_labels_path)
    classes_path = CUB_dataset.class_names_txt
    shutil.copy2(classes_path, out_dir / "classes.txt")
    attributes_names_txt = CUB_dataset.attributes_names_txt
    shutil.copy2(attributes_names_txt, out_dir / "attributes" / "attributes.txt")
    class_attributes_txt = CUB_dataset.class_attributes_txt
    shutil.copy2(class_attributes_txt, out_dir / "attributes" / "class_attribute_labels_continuous.txt")

    # write_replacement_attrs(base_img_out_dir)
    #
    # extract_label_information(cub_dataset=CUB_dataset, out_dir=out_dir)



if __name__ == "__main__":
    main()

