import csv
import os

import pandas as pd
import rootutils
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms


# Set up project root
project_root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)


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
            root, "attributes", f"class_attribute_labels_continuous_recal_{split}.txt"
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
