import os
import pandas as pd
from io_utils import ensure_dir
from local_datasets.cub_dataset import CUBDataset


def main():
    assets_dir = ensure_dir("/home/jonas/PycharmProjects/flux2/assets/")
    cub_root = "/data/jonas/CUB"

    cub = CUBDataset(
        cub_root,
        split="train",
        transform=None,
        return_segmentation=False,
    )

    ref_df = pd.read_csv(assets_dir / "reference_images_index.csv")

    # filter 'image_src_path' solumn for unique entries
    ref_df = ref_df.drop_duplicates(subset=["image_src_path"]).reset_index(drop=True)

    out_root = ensure_dir(assets_dir / "cub_reference_dataset")
    out_images_root = ensure_dir(out_root / "images")
    out_attr_root = ensure_dir(out_root / "attributes")

    images_txt = []
    labels_txt = []
    split_txt = []
    image_attr_rows = []

    next_img_id = 1

    # attr_id 1..312 in the same order as cub.attribute_names / cub.attr_matrix columns
    num_attrs = len(cub.attribute_names)

    for _, row in ref_df.iterrows():
        src_path = row["image_src_path"]
        dst_path = row["image_dst_path"]

        info = cub.get_info_from_image_path(src_path)

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
        out_root / "images.txt", sep=" ", index=False, header=False
    )
    pd.DataFrame(labels_txt).to_csv(
        out_root / "image_class_labels.txt", sep=" ", index=False, header=False
    )
    pd.DataFrame(split_txt).to_csv(
        out_root / "train_test_split.txt", sep=" ", index=False, header=False
    )

    # write attributes file
    pd.DataFrame(image_attr_rows).to_csv(
        out_attr_root / "image_attribute_labels.txt", sep=" ", index=False, header=False
    )

    print(f"Wrote {next_img_id - 1} samples.")
    print(f"Wrote {len(image_attr_rows)} attribute rows.")


if __name__ == "__main__":
    main()
