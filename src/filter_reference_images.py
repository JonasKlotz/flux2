from collections import defaultdict
from pathlib import Path


def main():
    ref_images_path = Path("/home/jonas/PycharmProjects/flux2/outputs/syn_cub_dataset/reference_images")

    # glob all dirs in ref_images_path
    dirs = [d for d in ref_images_path.iterdir() if d.is_dir()]
    attribute_dict = {}

    for d in dirs:
        dir_name = d.name
        attribute_dict[dir_name] = defaultdict(int)

        # get all jpg files in dir
        jpg_files = list(d.glob("*.jpg"))
        for jpg_file in jpg_files:
            # split at _
            parts = jpg_file.stem.split("_")
            # remove last 2 parts
            class_name = "_".join(parts[:-2])
            attribute_dict[dir_name][class_name] += 1

    # only keep top 3 classes per dir or fewer if less than 3 exist
    for dir_name, class_counts in attribute_dict.items():
        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        top_classes = {cls: count for cls, count in sorted_classes[:3]}
        top_classes = {cls: count for cls, count in top_classes.items() if count > 2}
        attribute_dict[dir_name] = top_classes

    print("Top classes per directory:")
    print(attribute_dict)

    # remove all jpg files that do not belong to the top classes per dir
    files_removed = 0
    for d in dirs:
        dir_name = d.name
        top_classes = attribute_dict[dir_name]

        jpg_files = list(d.glob("*.jpg"))
        for jpg_file in jpg_files:
            parts = jpg_file.stem.split("_")
            class_name = "_".join(parts[:-2])

            # Remove file if class is not in top classes
            if class_name not in top_classes:
                jpg_file.unlink()
                files_removed += 1
                print(f"Removed: {jpg_file}")

    print(f"\nTotal files removed: {files_removed}")


if __name__ == "__main__":
    main()