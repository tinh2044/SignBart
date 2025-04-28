import os

import cv2
import numpy as np
import torch
from tqdm import tqdm

total_body_idx = 33
total_hand = 42

body_idx = list(range(11, 17))
lefthand_idx = [x + total_body_idx for x in range(0, 21)]

righthand_idx = [x + 21 for x in lefthand_idx]

total_idx = body_idx + lefthand_idx + righthand_idx

def accuracy(logits, labels):
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    return correct / total


def top_k_accuracy(logits, labels, k=5):
    top_k_preds = torch.topk(logits, k, dim=1).indices
    correct = (top_k_preds == labels.unsqueeze(1)).any(dim=1).sum().item()
    total = labels.size(0)
    return correct / total


def save_checkpoints(model, optimizer, path_dir, epoch, name=None):
    if not os.path.exists(path_dir):
        print(f"Making directory {path_dir}")
        os.makedirs(path_dir)
    if name is None:
        filename = f'{path_dir}/checkpoints_{epoch}.pth'
    else:
        filename = f'{path_dir}/checkpoints_{epoch}_{name}.pth'
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch
    }, filename)


def load_checkpoints(model, optimizer, path, resume=True):
    if not os.path.exists(path):
        raise FileNotFoundError
    if os.path.isdir(path):
        epoch = max([int(x[x.index("_") + 1:len(x) - 4]) for x in os.listdir(path)])
        filename = f'{path}/checkpoints_{epoch}.pth'
        print(f'Loaded latest checkpoint: {epoch}')

        checkpoints = torch.load(filename)

    else:
        print(f"Load checkpoint from file : {path}")
        checkpoints = torch.load(path)

    model.load_state_dict(checkpoints['model'])
    optimizer.load_state_dict(checkpoints['optimizer'])
    if resume:
        return checkpoints['epoch'] + 1
    else:
        return 1


def train_epoch(model, dataloader, optimizer, scheduler=None, epoch=0, epochs=0):
    all_loss, all_acc, all_top_5_acc = 0.0, 0.0, 0.0
    loop = tqdm(enumerate(dataloader), total=len(dataloader), leave=True, desc=f"Training epoch {epoch + 1}/{epochs}: ")
    for i, data in loop:
        labels = data["labels"]
        optimizer.zero_grad()
        loss, logits = model(**data)
        loss.backward()
        optimizer.step()
        all_loss += loss.item()

        acc = accuracy(logits, labels)
        top_5_acc = top_k_accuracy(logits, labels, k=5)

        all_acc += acc
        all_top_5_acc += top_5_acc

        loop.set_postfix_str(f"Loss: {loss.item():.3f}, Acc: {acc:.3f}, Top 5 Acc: {top_5_acc:.3f}")

    if scheduler:
        scheduler.step(loss)

    all_loss /= len(dataloader)
    all_acc /= len(dataloader)
    all_top_5_acc /= len(dataloader)

    return all_loss, all_acc, all_top_5_acc


def evaluate(model, dataloader, epoch=0, epochs=0):
    all_loss, all_acc, all_top_5_acc = 0.0, 0.0, 0.0
    loop = tqdm(enumerate(dataloader), total=len(dataloader), leave=True,
                desc=f"Evaluation epoch {epoch + 1}/{epochs}: ")

    for i, data in loop:
        labels = data["labels"]
        loss, logits = model(**data)
    
        all_loss += loss.item()
        acc = accuracy(logits, labels)
        top_5_acc = top_k_accuracy(logits, labels, k=5)

        all_acc += acc
        all_top_5_acc += top_5_acc

        loop.set_postfix_str(f"Loss: {loss.item():.3f}, Acc: {acc:.3f}, Top 5 Acc: {top_5_acc:.3f}")

    all_loss /= len(dataloader)
    all_acc /= len(dataloader)
    all_top_5_acc /= len(dataloader)

    return all_loss, all_acc, all_top_5_acc

def create_attention_mask(mask, dtype, tgt_len = None):

        bsz, src_len = mask.size()
        tgt_len = tgt_len if tgt_len is not None else src_len

        expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

        inverted_mask = 1.0 - expanded_mask

        return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)
    
    
def create_causal_attention_mask(attention_mask, input_shape, inputs_embeds):
    
    batch_size, query_length = input_shape[0], input_shape[1]

    expanded_mask = attention_mask[:, None, None, :].expand(batch_size, 1, query_length, query_length).to(
        dtype=inputs_embeds.dtype
    )
    inverted_mask = 1.0 - expanded_mask
    expanded_mask = inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(inputs_embeds.dtype).min)
    
    causal_mask = torch.tril(torch.ones((query_length, query_length), device=inputs_embeds.device, dtype=inputs_embeds.dtype))
    expanded_mask += causal_mask[None, None, :, :]

    return expanded_mask


def count_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total_params, "trainable": trainable_params}


if __name__ == "__main__":
    mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]])
    input_embeds = torch.randn(2, 5, 768)
    expand_mask = create_causal_attention_mask(mask, (2, 5), input_embeds)
    print(expand_mask)