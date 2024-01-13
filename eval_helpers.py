import os
import torch
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as trn
from torch.autograd import Variable as V
import numpy as np
import calibration_tools
import logging

from tqdm import tqdm
from distutils.dir_util import copy_tree

logging.basicConfig(level=logging.INFO, filename="onepeace_eval.log", filemode="w")


def concat(x):
    np.concatenate(x, axis=0)


def to_np(x):
    return x.data.to("cpu").numpy()


# Symlink does not work on colab, just copy files
def create_symlinks_to_imagenet(imagenet_folder, folder_to_scan, imagenet1k_path):
    if not os.path.exists(imagenet_folder):
        os.makedirs(imagenet_folder)
        folders_of_interest = os.listdir(folder_to_scan)
        path_prefix = imagenet1k_path
        for folder in folders_of_interest:
            copy_tree(
                os.path.join(path_prefix, folder),
                os.path.join(imagenet_folder, folder),
            )


def get_predictions(loader, net=None, mask=None):
    confidence = []
    correct = []
    num_correct = 0
    with torch.no_grad():
        for data, target in tqdm(loader):
            data, target = data.cuda(), target.cuda()
            target = torch.tensor(target, dtype=torch.int64)
            mask = torch.tensor(mask, dtype=torch.bool)
            output = net(data)[:, mask]

            # accuracy
            pred = output.data.max(1)[1]
            num_correct += pred.eq(target.data).sum().item()

            confidence.extend(
                to_np(F.softmax(output, dim=1).max(1)[0]).squeeze().tolist()
            )
            pred = output.data.max(1)[1]
            correct.extend(pred.eq(target).to("cpu").numpy().squeeze().tolist())

    return np.array(confidence), np.array(correct), num_correct


def get_imagenet_a_results(loader, net, mask):
    confidence, correct, num_correct = get_predictions(loader, net, mask)
    acc = num_correct / len(loader.dataset)
    print("Accuracy (%):", round(100 * acc, 4))
    calibration_tools.show_calibration_results(confidence, correct)


def get_imagenet_o_results(in_loader, out_loader, net, mask):
    confidence_in, correct, num_correct = get_predictions(in_loader, net=net, mask=mask)
    in_score = -confidence_in
    confidence_out, correct_out, num_correct_out = get_predictions(
        out_loader, net=net, mask=mask
    )
    out_score = -confidence_out

    aurocs, auprs, fprs = [], [], []
    measures = calibration_tools.get_measures(out_score, in_score)
    aurocs = measures[0]
    auprs = measures[1]
    fprs = measures[2]

    calibration_tools.print_measures_old(aurocs, auprs, fprs, method_name="MSP")


def show_performance_imagenet_c(
    distortion_name, data_path, model, batch_size, num_workers, input_size
):
    errs = []
    logging.info(
        f"Evaluation for {distortion_name} ##########################################"
    )
    for severity in range(1, 6, 2):
        logging.info(f"Severity {severity} ##########################################")

        distorted_dataset = dset.ImageFolder(
            root=data_path + distortion_name + "/" + str(severity),
            transform=trn.Compose(
                [trn.Resize((input_size, input_size)), trn.ToTensor()]
            ),
        )

        distorted_dataset_loader = torch.utils.data.DataLoader(
            distorted_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        correct = 0
        for batch_idx, (data, target) in enumerate(tqdm(distorted_dataset_loader)):
            data = V(data.cuda(), volatile=True)
            output = model(data)
            pred = output.data.max(1)[1]
            correct += pred.eq(target.cuda()).sum()

        errs.append(1 - 1.0 * correct / len(distorted_dataset))

    errs = [err.cpu() for err in errs]
    logging.info("\n=Average", tuple(errs))
    return np.mean(errs)
