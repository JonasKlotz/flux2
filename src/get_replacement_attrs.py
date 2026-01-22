from pathlib import Path
import glob
from datasets import load_dataset


def main():
    SUB_dataset = load_dataset("Jessica-bader/SUB")


    to_change_path = Path("/home/jonas/PycharmProjects/flux2/assets/cub_reference_dataset/images")
    dirs = glob.glob(str(to_change_path / "*"))
    class_to_attrs = {d:glob.glob(str(Path(d) / "*")) for d in sorted(dirs)}

    # remove everything except dirname
    class_to_attrs = {Path(k).name: [Path(p).name for p in v] for k, v in class_to_attrs.items()}

    # remove everything behin last _ in values
    class_to_attrs = {k: [p.rsplit("_", 1)[0] for p in v][0] for k, v in class_to_attrs.items()}
    print(class_to_attrs)
    # save to txt
    with open("/home/jonas/PycharmProjects/flux2/assets/cub_reference_dataset/replacement_attrs.txt", "w") as f:
        for k, v in class_to_attrs.items():
            f.write(f"{k} {v}\n")

if __name__ == "__main__":
    main()