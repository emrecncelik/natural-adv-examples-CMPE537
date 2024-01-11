import torch
import torch.nn.functional as F

import numpy as np
import calibration_tools

from tqdm import tqdm


def concat(x):
    np.concatenate(x, axis=0)


def to_np(x):
    return x.data.to("cpu").numpy()


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
