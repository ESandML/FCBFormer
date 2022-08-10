import os
import glob
import argparse
import numpy as np

from sklearn.metrics import jaccard_score, f1_score, precision_score, recall_score
from skimage.io import imread
from skimage.transform import resize

from Data.dataloaders import split_ids


def eval(args):

    if args.test_dataset == "Kvasir":
        prediction_files = sorted(
            glob.glob(
                "./Predictions/Trained on {}/Tested on {}/*".format(
                    args.train_dataset, args.test_dataset
                )
            )
        )
        depth_path = args.root + "masks/*"
        target_paths = sorted(glob.glob(depth_path))
    elif args.test_dataset == "CVC":
        prediction_files = sorted(
            glob.glob(
                "./Predictions/Trained on {}/Tested on {}/*".format(
                    args.train_dataset, args.test_dataset
                )
            )
        )
        depth_path = args.root + "Ground Truth/*"
        target_paths = sorted(glob.glob(depth_path))

    _, test_indices, _ = split_ids(len(target_paths))

    test_files = sorted(
        [target_paths[test_indices[i]] for i in range(len(test_indices))]
    )

    dice = []
    IoU = []
    precision = []
    recall = []

    for i in range(len(test_files)):
        pred = np.ndarray.flatten(imread(prediction_files[i]) / 255) > 0.5
        gt = (
            resize(imread(test_files[i]), (int(352), int(352)), anti_aliasing=False)
            > 0.5
        )

        if len(gt.shape) == 3:
            gt = np.mean(gt, axis=2)
        gt = np.ndarray.flatten(gt)

        dice.append(f1_score(gt, pred))
        IoU.append(jaccard_score(gt, pred))
        precision.append(precision_score(gt, pred))
        recall.append(recall_score(gt, pred))

        if i + 1 < len(test_files):
            print(
                "\rTest: [{}/{} ({:.1f}%)]\tModel scores: Dice={:.6f}, mIoU={:.6f}, precision={:.6f}, recall={:.6f}".format(
                    i + 1,
                    len(test_files),
                    100.0 * (i + 1) / len(test_files),
                    np.mean(dice),
                    np.mean(IoU),
                    np.mean(precision),
                    np.mean(recall),
                ),
                end="",
            )
        else:
            print(
                "\rTest: [{}/{} ({:.1f}%)]\tModel scores: Dice={:.6f}, mIoU={:.6f}, precision={:.6f}, recall={:.6f}".format(
                    i + 1,
                    len(test_files),
                    100.0 * (i + 1) / len(test_files),
                    np.mean(dice),
                    np.mean(IoU),
                    np.mean(precision),
                    np.mean(recall),
                )
            )


def get_args():
    parser = argparse.ArgumentParser(
        description="Make predictions on specified dataset"
    )
    parser.add_argument(
        "--train-dataset", type=str, required=True, choices=["Kvasir", "CVC"]
    )
    parser.add_argument(
        "--test-dataset", type=str, required=True, choices=["Kvasir", "CVC"]
    )
    parser.add_argument("--data-root", type=str, required=True, dest="root")

    return parser.parse_args()


def main():
    args = get_args()
    eval(args)


if __name__ == "__main__":
    main()

