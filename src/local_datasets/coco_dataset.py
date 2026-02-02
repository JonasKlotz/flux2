import os

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from torch.utils.data import Dataset
import torchvision

from create_syn_images import main

COCO_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]
COCO_IDX2NAME = {i: name for i, name in enumerate(COCO_CLASSES)}


def collate_fn(batch):
    images, bboxes, masks, category_ids, mlc_vectors, idx = zip(*batch)
    images = torch.stack(images)
    mlc_vectors = torch.stack(mlc_vectors)  # fixed-size so can stack
    return images, bboxes, masks, category_ids, mlc_vectors, idx


class COCODataset(Dataset):
    num_classes = 80

    def __init__(self, root_dir, annotation_file, transform=None, normalize=True):
        self.root_dir = root_dir
        self.transform = transform
        self.normalize = normalize
        self.coco = COCO(annotation_file)
        self.image_ids = list(self.coco.imgs.keys())

        # keep only images that have at least one valid annotation
        self.image_ids = [
            i for i in self.image_ids if len(self.coco.getAnnIds(imgIds=i)) > 0
        ]

        # build contiguous label space
        self.cat_ids = sorted(self.coco.getCatIds())
        self.catid2contig = {cid: i for i, cid in enumerate(self.cat_ids)}
        self.num_classes = len(self.cat_ids)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        path = os.path.join(self.root_dir, image_info["file_name"])

        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        H, W = image.shape[:2]

        ann_ids = self.coco.getAnnIds(imgIds=image_id, iscrowd=None)
        anns = self.coco.loadAnns(ann_ids)

        bboxes, masks, category_ids = [], [], []
        mlc_vector = np.zeros(self.num_classes, dtype=np.float32)

        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue

            segm = ann.get("segmentation", None)

            # Try segmentation first
            mask = None
            if segm is not None and len(segm) > 0:
                try:
                    mask = self.coco.annToMask(ann)
                except TypeError:
                    # Defensive fallback when pycocotools hits the bbox path
                    mask = None

            # Fallback to a rectangular mask from bbox
            if mask is None:
                x1 = max(0, int(np.floor(x)))
                y1 = max(0, int(np.floor(y)))
                x2 = min(W, int(np.ceil(x + w)))
                y2 = min(H, int(np.ceil(y + h)))
                if x2 <= x1 or y2 <= y1:
                    continue
                mask = np.zeros((H, W), dtype=np.uint8)
                mask[y1:y2, x1:x2] = 1

            if mask.sum() == 0:
                continue

            cat = self.catid2contig[ann["category_id"]]
            bboxes.append([x, y, x + w, y + h])
            masks.append(mask)
            category_ids.append(cat)
            mlc_vector[cat] = 1

        if self.transform is not None:
            image = torchvision.transforms.ToPILImage()(image)
            image = self.transform(image)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            image = image / 255.0

        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        if masks:
            masks = torch.as_tensor(np.stack(masks, axis=0), dtype=torch.float32)
        else:
            # empty mask tensor with correct spatial size
            if isinstance(image, torch.Tensor):
                h_t, w_t = image.shape[1], image.shape[2]
            else:
                # if transforms produced PIL, not expected here, but guard anyway
                h_t, w_t = H, W
            masks = torch.zeros((0, h_t, w_t), dtype=torch.float32)

        category_ids = torch.as_tensor(category_ids, dtype=torch.int64)
        mlc_vector = torch.as_tensor(mlc_vector, dtype=torch.float32)
        return image, bboxes, masks, category_ids, mlc_vector, anns, idx

    def get_class_name(self, class_idx):
        class_dict = self.coco.cats[self.cat_ids[class_idx]]
        return class_dict["name"]



def main():

    coco_path = "/data/jonas/COCO"
    coco_train_path = os.path.join(coco_path, "train2017")
    annotation_file = os.path.join(
        coco_path, "annotations_trainval2017/annotations/instances_train2017.json"
    )

    dataset = COCODataset(
        root_dir=coco_train_path,
        annotation_file=annotation_file,
        transform=None,
        normalize=True,
    )
    for i in range(15):
        sample = dataset[i]
        # plot
        img, bboxes, masks, category_ids, mlc_vector, anns, idx = sample

        plot_coco(bboxes, category_ids, idx, img, mlc_vector)


def plot_coco(bboxes, category_ids, idx, img, mlc_vector):
    import matplotlib.pyplot as plt

    img_np = img.permute(1, 2, 0).numpy()
    plt.imshow(img_np)
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        rect = plt.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, fill=False, color="red", linewidth=2
        )
        plt.gca().add_patch(rect)
    plt.show()
    print("Sampled image index:", idx)
    print("Number of objects:", bboxes.shape[0])
    print("Category IDs:", category_ids)
    print("MLC vector:", mlc_vector)


if __name__ == "__main__":
    main()
