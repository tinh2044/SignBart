import os
import argparse
import random
import logging
import torch
import numpy as np
import yaml
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from torch.utils.data import DataLoader
from pathlib import Path

from dataset import Datasets
from model import SignBart
from utils import train_epoch, evaluate
from utils import body_idx, lefthand_idx, righthand_idx, save_checkpoints, load_checkpoints

def get_default_args():
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("--experiment_name", type=str, default="",
                        help="Name of the experiment after which the logs and plots will be named")
    parser.add_argument("--config_path", type=str, default="",
                        help="Path to the config model to be used")
    parser.add_argument("--pretrained_path", type=str, default="",
                        help="Path to the config model to be used")
    parser.add_argument("--seed", type=int, default=379,
                        help="Seed with which to initialize all the random components of the training")
    parser.add_argument("--task", type=str, default=False, choices=["train", "eval"],
                        help="Whether to train or evaluate the model")

    parser.add_argument("--data_path", type=str, default="",
                        help="Path to the training dataset")

    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs to train the model for")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate for the model training")

    parser.add_argument("--resume_checkpoints", type=str, default="",
                        help="Path to the checkpoints to be used for resuming training")

    # Scheduler
    parser.add_argument("--scheduler_factor", type=float, default=0.1,
                        help="Factor for the ReduceLROnPlateau scheduler")
    parser.add_argument("--scheduler_patience", type=int, default=5,
                        help="Patience for the ReduceLROnPlateau scheduler")

    return parser

def setup_logging(experiment_name):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(experiment_name + ".log")
        ]
    )

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    g = torch.Generator()
    g.manual_seed(seed)
    return g

def load_model(config_path, pretrained_path, device):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    model = SignBart(config)
    model.train(True)
    model.to(device)
    if pretrained_path:
        print(f"Load checkpoint from file : {pretrained_path}")
        state_dict = torch.load(pretrained_path)
        ret = model.load_state_dict(state_dict, strict=False)
        
        print("Missing keys: ", ret.missing_keys)
        print("Unexpected keys: ", ret.unexpected_keys)
        
    return model

def prepare_data_loaders(data_path, joint_idx, generator):
    train_datasets = Datasets(data_path, "train", shuffle=True, joint_idxs=joint_idx, augment=True)
    val_datasets = Datasets(data_path, "test", shuffle=True, joint_idxs=joint_idx)
    train_loader = DataLoader(train_datasets, shuffle=True, generator=generator,
                              batch_size=1, collate_fn=train_datasets.data_collator)
    val_loader = DataLoader(val_datasets, shuffle=True, generator=generator,
                            batch_size=1, collate_fn=val_datasets.data_collator)
    return train_loader, val_loader

def main(args):
    g = set_random_seed(args.seed)
    setup_logging(args.experiment_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    joint_idx = [body_idx, lefthand_idx, righthand_idx]
    checkpoint_dir = "checkpoints_" + args.experiment_name

    train_loader, val_loader = prepare_data_loaders(args.data_path, joint_idx, g)
    model = load_model(args.config_path, args.pretrained_path, device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.scheduler_factor,
                                                     patience=args.scheduler_patience)

    list_train_loss, list_train_acc, list_val_loss, list_val_acc = [], [], [], []
    top_train_acc, top_val_acc = 0, 0
    lr_progress = []
    epochs = args.epochs

    if args.resume_checkpoints:
        print(f"Resume training from file : {args.resume_checkpoints}")
        resume_epoch = load_checkpoints(model, optimizer, args.resume_checkpoints, resume=True)
    else:
        resume_epoch = 0

    if args.task == "eval":
        print("Evaluate model..!")
        model.train(False)
        val_loss, val_acc, _ = evaluate(model, val_loader, epoch=0, epochs=0)
        print(f"[1] Valuation  loss: {val_loss} acc: {val_acc}")
        return

    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path("out-imgs/" + args.experiment_name + "/").mkdir(parents=True, exist_ok=True)
    for epoch in range(resume_epoch, epochs):
        train_loss, train_acc, _ = train_epoch(model, train_loader, optimizer, scheduler, epoch=epoch, epochs=epochs)
        list_train_loss.append(train_loss)
        list_train_acc.append(train_acc)

        model.train(False)
        val_loss, val_acc, _ = evaluate(model, val_loader, epoch=epoch, epochs=epochs)
        model.train(True)

        list_val_loss.append(val_loss)
        list_val_acc.append(val_acc)

        if train_acc > top_train_acc:
            top_train_acc = train_acc
            save_checkpoints(model, optimizer, checkpoint_dir, epoch)

        if val_acc > top_val_acc:
            top_val_acc = val_acc
            save_checkpoints(model, optimizer, checkpoint_dir, epoch)

        print(f"[{epoch + 1}] TRAIN  loss: {train_loss} acc: {train_acc}")
        logging.info(f"[{epoch + 1}] TRAIN  loss: {train_loss} acc: {train_acc}")
        print(f"[{epoch + 1}] Valuation  loss: {val_loss} acc: {val_acc}")
        logging.info(f"[{epoch + 1}] Valuation  loss: {val_loss} acc: {val_acc}")
        print("")
        logging.info("")

        lr_progress.append(optimizer.param_groups[0]["lr"])

    fig, ax = plt.subplots()
    ax.plot(range(1, len(list_train_loss) + 1), list_train_loss, c="#D64436", label="Training loss")
    ax.plot(range(1, len(list_train_acc) + 1), list_train_acc, c="#00B09B", label="Training accuracy")
    if val_loader:
        ax.plot(range(1, len(list_val_acc) + 1), list_val_acc, c="#E0A938", label="Validation accuracy")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set(xlabel="Epoch", ylabel="Accuracy / Loss", title="")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=4, fancybox=True, shadow=True, fontsize="xx-small")
    ax.grid()
    fig.savefig("out-imgs/" + args.experiment_name + "_loss.png")

    fig1, ax1 = plt.subplots()
    ax1.plot(range(1, len(lr_progress) + 1), lr_progress, label="LR")
    ax1.set(xlabel="Epoch", ylabel="LR", title="")
    ax1.grid()
    fig1.savefig("out-imgs/" + args.experiment_name + "_lr.png")

    print("\nAny desired statistics have been plotted.\nThe experiment is finished.")
    logging.info("\nAny desired statistics have been plotted.\nThe experiment is finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser("", parents=[get_default_args()], add_help=False)
    args = parser.parse_args()
    main(args)
