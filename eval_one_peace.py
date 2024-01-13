import calibration_tools
import numpy as np
import os
import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as trn
import torchvision.transforms.functional as trnF
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import models_vit
import logging

# timm is needed for deit models
import timm

# assert timm.__version__ == "0.3.2"
import shutil
import argparse
from copy import copy
from tqdm import tqdm
import torch.backends.cudnn as cudnn

# MODEL_DIR = "/content/drive/MyDrive/CMPE537_Project/models"
# DATA_DIR = "/content/drive/MyDrive/CMPE537_Project/data"
# MODEL_PATH_SMALL = f"{MODEL_DIR}/onepeace_ft_21kto1k_384.pth"
# MODEL_PATH_LARGE = f"{MODEL_DIR}/onepeace_ft_21kto1k_512.pth"
# # PATH_TO_IMAGENET_A = f"{DATA_DIR}/imagenet-a/val"
# PATH_TO_IMAGENET_A = f"/content/sketch"
# PATH_TO_IMAGENET_O = f"{DATA_DIR}/imagenet-o"
# PATH_TO_IMAGENET_VAL = f"/content/imagenet-1k-fuck/val"
# TORCH_HOME_DIR = "~/.models/"


def get_args():
    parser = argparse.ArgumentParser(
        "fine-tuning for image classification", add_help=False
    )
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    parser.add_argument(
        "--model_path",
        default="/content/models/onepeace_ft_21kto1k_384.pth",
        type=str,
        help="Path of model to eval",
    )
    parser.add_argument(
        "--model_name",
        default="one_piece_g_384",
        type=str,
        help="Name of model to eval",
    )
    parser.add_argument("--input_size", default=384, type=int, help="images input size")
    parser.add_argument(
        "--drop_path",
        type=float,
        default=0.1,
        metavar="PCT",
        help="Drop path rate (default: 0.1)",
    )
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--global_pool", action="store_true")
    parser.set_defaults(global_pool=True)
    parser.add_argument(
        "--data_path",
        default="/content/imagenet-a",
        type=str,
        help="dataset path",
    )
    parser.add_argument(
        "--data_name",
        default="imagenet-a",
        choices=[
            "imagenet-a",
            "imagenet-c",
            "imagenet-p",
            "imagenet-r",
            "imagenet-sketch",
        ],
        type=str,
        help="ImageNet dataset path",
    )
    parser.add_argument(
        "--imagenet1k_path",
        default="/content/imagenet-1k",
        type=str,
        help="imagenet1k val path",
    )
    parser.add_argument(
        "--nb_classes",
        default=1000,
        type=int,
        help="number of the classification types",
    )
    parser.add_argument(
        "--output_dir",
        default="./output_dir",
        help="path where to save, empty for no saving",
    )
    parser.add_argument(
        "--log_dir", default="./output_dir", help="path where to tensorboard log"
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    return parser.parse_args()


def main(args):
    logging.basicConfig(level=logging.INFO, filename="onepeace_eval.log", filemode="w")

    # Set seed
    torch.manual_seed(1)
    np.random.seed(1)
    torch.cuda.manual_seed(1)
    cudnn.benchmark = True

    # test_transform = trn.Compose(
    #     [trn.Resize((args.input_size, args.input_size)), trn.ToTensor()]
    # )

    # naes = dset.ImageFolder(root=PATH_TO_IMAGENET_A, transform=test_transform)
    # nae_loader = torch.utils.data.DataLoader(
    #     naes, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
    # )

    # noes = dset.ImageFolder(root=PATH_TO_IMAGENET_O, transform=test_transform)
    # noe_loader = torch.utils.data.DataLoader(
    #     noes, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
    # )

    # imagenet_o_folder = "imagenet_val_for_imagenet_o_ood/"

    # create_symlinks_to_imagenet(
    #     imagenet_o_folder, PATH_TO_IMAGENET_O, args.imagenet1k_path
    # )

    # val_examples_imagenet_o = dset.ImageFolder(
    #     root=imagenet_o_folder, transform=test_transform
    # )
    # val_loader_imagenet_o = torch.utils.data.DataLoader(
    #     val_examples_imagenet_o,
    #     batch_size=32,
    #     shuffle=False,
    #     num_workers=4,
    #     pin_memory=True,
    # )

    # val_imagenet = dset.ImageFolder(root=PATH_TO_IMAGENET_VAL, transform=test_transform)
    # val_imagenet_loader = torch.utils.data.DataLoader(
    #     val_imagenet, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
    # )

    # Load & configure model
    model = models_vit.__dict__[args.model_name](
        num_classes=args.nbclasses,
        drop_path_rate=args.drop_path,
        dropout=args.dropout,
        global_pool=args.global_pool,
        use_checkpoint=True,
    )

    device = torch.device("cuda")
    model.to(device)
    checkpoint = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])

    model.cuda()
    model.eval()

    if args.data_name == "imagenet-c":
        from calibration_tools import show_performance_imagenet_c

        distortions = [
            "gaussian_noise",
            "shot_noise",
            "impulse_noise",
            "defocus_blur",
            "glass_blur",
            "motion_blur",
            "zoom_blur",
            "snow",
            "frost",
            "fog",
            "brightness",
            "contrast",
            "elastic_transform",
            "pixelate",
            "jpeg_compression",
            "speckle_noise",
            "gaussian_blur",
            "spatter",
            "saturate",
        ]
        error_rates = []
        for distortion_name in distortions:
            rate = show_performance_imagenet_c(
                distortion_name,
                args.data_path,
                model,
                args.batch_size,
                args.num_workers,
            )
            error_rates.append(rate)
            logging.info(
                "Distortion: {:15s}  | CE (unnormalized) (%): {:.2f}".format(
                    distortion_name, 100 * rate
                )
            )
        logging.info(
            "mCE (unnormalized by AlexNet errors) (%): {:.2f}".format(
                100 * np.mean(error_rates)
            )
        )
    elif args.data_name == "imagenet-p":
        pass
    # print("ImageNet-A Results")
    # get_imagenet_a_results(nae_loader, net=net, mask=imagenet_a_mask)

    # print("\n\n\n")


if __name__ == "__main__":
    args = get_args()
    main(args)
