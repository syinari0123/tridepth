import argparse
import torch
import torch.backends.cudnn as cudnn

from auxiliary import str2bool, fix_random_seed, prepare_logdir, save_arguments
from dataloaders import prepare_dataloader, NYUCamMat
from models import Model
from trainer import TriDepthTrainer


def main(args):
    # CUDA settings
    fix_random_seed(seed=46)
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Logger
    log_dir = prepare_logdir(log_path=args.log_path, descript=args.theme)
    save_arguments(args, log_dir)

    # Model
    print("=> Creating model")
    cudnn.benchmark = True
    model = Model(cam_mat=NYUCamMat(), model_type=args.model_type,
                  loss_type="l1", normal_weight=0.5).to(device)

    # Optimizer
    print("=> Preparing optimizer")
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # Dataset
    print("=> Preparing dataloader")
    dataloader_dic = prepare_dataloader(args.data_path,
                                        datatype_list=["train", "val", "test"],
                                        batchsize=args.batchsize,
                                        workers=args.workers,
                                        img_size=(228, 304),
                                        val_split_rate=0.01)
    print('Train set\t', len(dataloader_dic["train"]))
    print('Validation set\t', len(dataloader_dic["val"]))
    print('Test set\t', len(dataloader_dic["test"]))

    # Start training
    print("=> Start training!!!")
    trainer = TriDepthTrainer(model, optimizer, dataloader_dic,
                              trainer_args={"log_root": log_dir,
                                            "nepoch": args.nepoch,
                                            "print_freq": args.print_freq,
                                            "img_print_freq": args.img_print_freq,
                                            "print_progress": args.print_progress},
                              device=device)
    trainer.load_checkpoint(args.pretrained_path)
    trainer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Basic settings
    parser.add_argument('--theme', type=str, default="test", help='Theme description of this experiment')
    parser.add_argument('--log-path', type=str, default="log")

    # Basic settings
    parser.add_argument('--data-path', type=str, default="~/datasets/nyudepthv2")
    parser.add_argument('--model-type', type=str, default="simple", choices=["simple", "upconv"])
    parser.add_argument('--pretrained-path', type=str, default="")

    # Optimizer setting
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=1e-4)

    # Train settings
    parser.add_argument('--batchsize', type=int, default=2)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--nepoch', type=int, default=3)
    parser.add_argument('--print-freq', type=int, default=3)
    parser.add_argument('--img-print-freq', type=int, default=3)
    parser.add_argument('--print-progress', type=str2bool, default="true")

    args = parser.parse_args()
    print(args)
    main(args)
