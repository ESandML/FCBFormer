import torch
import os
import argparse
import time
import numpy as np
import glob
import cv2

import torch
import torch.nn as nn

from Data import dataloaders
from Models import models
from Metrics import performance_metrics


def build(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.test_dataset == "Kvasir":
        img_path = args.root + "images/*"
        input_paths = sorted(glob.glob(img_path))
        depth_path = args.root + "masks/*"
        target_paths = sorted(glob.glob(depth_path))
    elif args.test_dataset == "CVC":
        img_path = args.root + "Original/*"
        input_paths = sorted(glob.glob(img_path))
        depth_path = args.root + "Ground Truth/*"
        target_paths = sorted(glob.glob(depth_path))
    _, test_dataloader, _ = dataloaders.get_dataloaders(
        input_paths, target_paths, batch_size=1
    )

    _, test_indices, _ = dataloaders.split_ids(len(target_paths))
    target_paths = [target_paths[test_indices[i]] for i in range(len(test_indices))]

    perf = performance_metrics.DiceScore()

    model = models.FCBFormer()

    state_dict = torch.load(
        "./Trained models/FCBFormer_{}.pt".format(args.train_dataset)
    )
    model.load_state_dict(state_dict["model_state_dict"])

    model.to(device)

    return device, test_dataloader, perf, model, target_paths


@torch.no_grad()
def predict(args):
    device, test_dataloader, perf_measure, model, target_paths = build(args)

    if not os.path.exists("./Predictions"):
        os.makedirs("./Predictions")
    if not os.path.exists("./Predictions/Trained on {}".format(args.train_dataset)):
        os.makedirs("./Predictions/Trained on {}".format(args.train_dataset))
    if not os.path.exists(
        "./Predictions/Trained on {}/Tested on {}".format(
            args.train_dataset, args.test_dataset
        )
    ):
        os.makedirs(
            "./Predictions/Trained on {}/Tested on {}".format(
                args.train_dataset, args.test_dataset
            )
        )

    t = time.time()
    model.eval()
    perf_accumulator = []
    for i, (data, target) in enumerate(test_dataloader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        perf_accumulator.append(perf_measure(output, target).item())
        predicted_map = np.array(output.cpu())
        predicted_map = np.squeeze(predicted_map)
        predicted_map = predicted_map > 0
        cv2.imwrite(
            "./Predictions/Trained on {}/Tested on {}/{}".format(
                args.train_dataset, args.test_dataset, os.path.basename(target_paths[i])
            ),
            predicted_map * 255,
        )
        if i + 1 < len(test_dataloader):
            print(
                "\rTest: [{}/{} ({:.1f}%)]\tAverage performance: {:.6f}\tTime: {:.6f}".format(
                    i + 1,
                    len(test_dataloader),
                    100.0 * (i + 1) / len(test_dataloader),
                    np.mean(perf_accumulator),
                    time.time() - t,
                ),
                end="",
            )
        else:
            print(
                "\rTest: [{}/{} ({:.1f}%)]\tAverage performance: {:.6f}\tTime: {:.6f}".format(
                    i + 1,
                    len(test_dataloader),
                    100.0 * (i + 1) / len(test_dataloader),
                    np.mean(perf_accumulator),
                    time.time() - t,
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
    predict(args)


if __name__ == "__main__":
    main()

