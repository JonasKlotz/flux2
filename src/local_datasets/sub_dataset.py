import pandas as pd
from datasets import load_dataset
from pandas import DataFrame

from local_datasets.cub_dataset import CUBDataset
# python
import pickle
from pathlib import Path
from typing import Any, Dict, Union
import numpy as np
# -----------------------------
# helpers
# -----------------------------
def family(attr: str) -> str:
    # "has_eye_color::black" -> "has_eye_color"
    return attr.split("::", 1)[0] if isinstance(attr, str) else ""

def remove_family(attrs, banned_family: str):
    return [a for a in attrs if family(a) != banned_family]

def keep_topk(attrs, k: int):
    return list(attrs[:k])

def save_bird_labels(bird_labels: Dict[Any, Any], path: Union[str, Path] = 'bird_labels.pkl') -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as f:
        pickle.dump(bird_labels, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_bird_labels(path: Union[str, Path] = 'bird_labels.pkl') -> Dict[Any, Any]:
    path = Path(path)
    with path.open('rb') as f:
        return pickle.load(f)


def get_class_attrs(CUB_dataset: CUBDataset) -> DataFrame:
    class_attributes_df = CUB_dataset.class_attributes  # rows attributes, cols classes
    attributes_df = CUB_dataset.attribute_names  # attr_id attr_name  - id starts at 1
    class_df = CUB_dataset.class_names  # class_id, class_name - id starts at 1

    # rename rows and col inde to corresponding class and attr
    # assign names by position when possible, otherwise map using 1-based ids
    attr_names = attributes_df['attr_name'].to_numpy()
    class_names = class_df['class_name'].to_numpy()

    class_attributes_df.columns = attr_names
    class_attributes_df['class_name'] = class_names
    class_attributes_df.columns.name = 'attribute'
    return class_attributes_df


def main():
    data_dir = "/data/jonas/CUB"
    CUB_dataset = CUBDataset(
        data_dir,
        split="train",
        transform=None,
        return_segmentation=False,
    )

    df = get_class_attrs(CUB_dataset)

    dataset = load_dataset("Jessica-bader/SUB")
    features = dataset["test"].features
    bird_labels = {name:[] for name in features["bird_label"].names}
    test_data = dataset["test"]
    # for sample in test_data:
    #     attr_label = dataset["test"].features["attr_label"].int2str(sample["attr_label"]).replace("--", "::")
    #     bird_label = dataset["test"].features["bird_label"].int2str(sample["bird_label"])
    #
    #     if attr_label not in bird_labels[bird_label]:
    #         bird_labels[bird_label].append(attr_label)

    # save_bird_labels(bird_labels)          # saves to `bird_labels.pkl`
    sub_perturbation_loaded = load_bird_labels("bird_labels.pkl")

    birds_to_consider = features["bird_label"].names
    df = df[df["class_name"].isin(birds_to_consider)]

    # top-k attributes per class (from CUB)
    attribute_cols = [c for c in df.columns if c != "class_name"]

    topk = 5
    class_to_top_attrs = {}
    for _, row in df.iterrows():
        class_name = row["class_name"]
        values = row[attribute_cols].to_numpy(dtype=float)
        top_idx = np.argsort(values)[-topk:][::-1]
        class_to_top_attrs[class_name] = [attribute_cols[i] for i in top_idx]

    # clean:
    BANNED_FAMILY = "has_eye_color"
    TOPK = 3

    class_to_top_attrs_clean = {}
    for cls, attrs in class_to_top_attrs.items():
        attrs2 = keep_topk(remove_family(attrs, BANNED_FAMILY), TOPK)
        if attrs2:
            class_to_top_attrs_clean[cls] = attrs2

    # match (deduplicated):
    # dedup criterion: within each class, each loaded_attr should appear at most once.
    # if it matches multiple topk ranks (rare but possible if families repeat), keep best (lowest rank).
    rows = []
    seen_best_rank = {}  # key: (class_name, loaded_attr) -> best_rank

    for cls, top_attrs in class_to_top_attrs_clean.items():
        top_fams = [family(a) for a in top_attrs]

        loaded_attrs = sub_perturbation_loaded.get(cls, [])
        if isinstance(loaded_attrs, dict):
            flat = []
            for v in loaded_attrs.values():
                if isinstance(v, list):
                    flat.extend(v)
            loaded_attrs = flat

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

    print(matches_df.head(20))

    # save
    path = Path("/home/jonas/PycharmProjects/flux2/assets/")


if __name__ == '__main__':
    main()