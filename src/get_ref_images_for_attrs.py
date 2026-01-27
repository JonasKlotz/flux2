import os
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt

from create_syn_images import load_replacement_attrs, ATTR_FALLBACK
from local_datasets.cub_dataset import CUBDataset
from prepare_image_generation import family


def main():
    replacement_attrs = Path(
        "/home/jonas/PycharmProjects/flux2/outputs/syn_cub_dataset/images/replacement_attrs.txt"
    )
    replacement_attr_dict = load_replacement_attrs(replacement_attrs)
    sub = load_dataset("Jessica-bader/SUB")
    sub_attr_names = sub["test"].features["attr_label"].names
    sub_attr_names = [name.replace("--", "::") for name in sub_attr_names]

    outputs = Path("/home/jonas/PycharmProjects/flux2/outputs/syn_cub_dataset/reference_images/confidence>3")

    attrs_to_replace = list(replacement_attr_dict.values())
    # concatenate the list of lists in attrs_to_replace
    attrs_to_replace = set([item for sublist in attrs_to_replace for item in sublist])
    all_candidate_attr_names = []
    for attr_to_replace in attrs_to_replace:

        fam = family(attr_to_replace)
        candidate_attr_names = [
            a for a in sub_attr_names
            if family(a) == fam and a != attr_to_replace
        ]

        if len(candidate_attr_names) == 0:
            candidate_attr_names = [ATTR_FALLBACK.get(attr_to_replace)]
        all_candidate_attr_names += candidate_attr_names

    match_counter = Counter({a: 0 for a in all_candidate_attr_names})

    cub_root = Path("/data/jonas/CUB")

    cub = CUBDataset(
        cub_root,
        split="all",
        transform=None,
        return_segmentation=False,
        only_attributes_with_high_certainty=True,
    )


    for idx in tqdm(range(len(cub))):
        img, label, attrs, key = cub[idx]
        class_name = cub.label_to_class_name(label)
        attr_names = cub.attributes_to_names(attrs)
        img_path = cub.get_image_path(idx)
        filename = os.path.basename(img_path)

        matches = [
            name for name in attr_names
            if name in all_candidate_attr_names
        ]
        for match in matches:
            match_counter[match] += 1
            tmp_dir = outputs / match
            os.makedirs(tmp_dir, exist_ok=True)
            img.save(tmp_dir / filename)

        # if len(matches) > 0:
        #     plt.imshow(img)
        #     plt.axis("off")
        #     plt.title(f"Class: {class_name}\nAttrs: {', '.join(matches)}")
        #     plt.show()


    print("Match counts per attribute:")
    for attr, count in match_counter.items():
        print(f"{attr}: {count}")

    zero_matches = [a for a, c in match_counter.items() if c == 0]
    print(f"Attributes with zero matches: {len(zero_matches)}")

    total_matches = sum(match_counter.values())
    print(f"Total matches found: {total_matches}")


if __name__ == "__main__":
    main()