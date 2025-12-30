import os
import numpy as np
import torch
from tqdm import tqdm

from local_datasets.cub_dataset import CUBDataset


def recompute_class_attribute_matrix(dataset, out_path):
    """
    dataset: CUBDataset instance
    out_path: path to save txt file
    """

    # infer dimensions from first sample
    _, label, attrs, _ = dataset[0]
    num_classes = label.shape[0]
    num_attrs = attrs.shape[0]

    class_attr_sum = np.zeros((num_classes, num_attrs), dtype=np.float64)
    class_count = np.zeros(num_classes, dtype=np.int64)

    for i in tqdm(range(len(dataset))):
        _, label, attrs, _ = dataset[i]

        # ensure numpy
        if torch.is_tensor(label):
            label = label.cpu().numpy()
        if torch.is_tensor(attrs):
            attrs = attrs.cpu().numpy()

        active_classes = np.where(label == 1)[0]

        for c in active_classes:
            class_attr_sum[c] += attrs
            class_count[c] += 1



    # save
    np.savetxt(out_path, class_attr_sum)
    print(f"Saved: {out_path}")
    print(f"Class counts (min/mean/max): "
          f"{class_count.min()} / {class_count.mean():.1f} / {class_count.max()}")

def main():
    data_dir = "/data/jonas/CUB"
    out_dir = os.path.join(data_dir, "attributes")
    os.makedirs(out_dir, exist_ok=True)

    for split in ["train", "val", "test"]:
        dataset = CUBDataset(
            data_dir,
            split=split,
            transform=None,
            return_segmentation=False,
        )

        out_path = os.path.join(
            out_dir,
            f"class_attribute_labels_continuous_recal_{split}.txt",
        )

        recompute_class_attribute_matrix(dataset, out_path)

if __name__ == "__main__":
    main()
