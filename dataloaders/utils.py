import os
import numpy as np

IMG_EXTENSIONS = ['.h5', ]


def is_image_file(filename):
    """Check whether this file has specified extensions (".h5") or not.

    Args:
        filename (str): Image filename

    Returns:
        bool: Whether this file's extension is in IMG_EXTENSIONS or not.
    """
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, data_type, val_split_rate=0.01):
    """Obtain an image-path list in the dataset dir, and Split them into train, val, and test. 

    Args:
        dir (str): Dataset root directory path
        data_type: Select data type from train/val/test
        val_split_rate: Ratio (0~1) of validation data in entire dataset

    Returns:
        images: Image list of specified type of dataset.
    """
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, target)

                    if data_type == "train":
                        if np.random.random() >= val_split_rate:
                            images.append(item)
                    elif data_type == "val":
                        if np.random.random() < val_split_rate:
                            images.append(item)
                    elif data_type == "test":
                        images.append(item)
                    else:
                        raise (RuntimeError("Invalid dataset type: " + data_type + "\n"
                                            "Supported dataset types are: train, val, test"))

    # load checker
    print("Found {} images in {} folder.".format(len(images), dir))

    return images
