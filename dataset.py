import json
import pickle

import numpy as np
import torch
import os
from augmentations import rotate_keypoints, noise_injection, clip_frame, time_warp_uniform, flip_keypoints

class Datasets(torch.utils.data.Dataset):
    def __init__(self, root, split, shuffle=True, joint_idxs=None, augment=False):
        self.root = root
        self.split = split
        self.joint_idxs = joint_idxs
        self.augment = augment
        with open(f"{self.root}/label2id.json", 'r') as f:
            self.label2id = json.load(f)

        with open(f"{self.root}/id2label.json", 'r') as f:
            self.id2label = json.load(f)

        self.data_dir = f"{root}/{split}"

        self.list_key = [f"{self.data_dir}/{x}/{y}"
                         for x in os.listdir(self.data_dir)
                         for y in os.listdir(f"{self.data_dir}/{x}")
                         ]

        if shuffle:
            np.random.shuffle(self.list_key)

    def __getitem__(self, i):
        key = self.list_key[i]
        with open(key, "rb") as f:
            sample = pickle.load(f)

        keypoints = np.array(sample['keypoints'])[:, :, :2]
        keypoints = keypoints[:, :, :2]
        assert sample['class'] == key.split("/")[-2], f"{sample['class']} != {key.split('/')[-2]}"
        label = self.label2id[sample['class']]

        keypoints = np.clip(keypoints, 0, 1)
        if keypoints.shape[0] > 64:
            keypoints = clip_frame(keypoints, 64, True)

        if self.augment and np.random.uniform(0, 1) < 0.4:
            keypoints = self.apply_augment(keypoints)

        keypoints = self.normalize_keypoints(keypoints)

        keypoints = torch.from_numpy(keypoints).float()
        label = torch.tensor(label, dtype=torch.long)

        return keypoints, label

    def apply_augment(self, keypoints):
        aug = False
        while not aug:
            if np.random.uniform(0, 1) < 0.5:
                angle = np.random.uniform(-15, 15)
                keypoints = rotate_keypoints(keypoints, (0, 0), angle)
                aug = True
            if np.random.uniform(0, 1) < 0.5:
                random_noise = np.random.uniform(0.01, 0.2)
                keypoints = noise_injection(keypoints, random_noise)
                aug = True
            if np.random.uniform(0, 1) < 0.5:
                n_f = keypoints.shape[0]
                tgt = np.random.randint(n_f // 2, n_f)
                is_uniform = np.random.uniform(0, 1) < 0.5
                keypoints = clip_frame(keypoints, tgt, is_uniform)
                aug = True
            if np.random.uniform(0, 1) < 0.5:
                speed = np.random.uniform(1.1, 1.5)
                keypoints = time_warp_uniform(keypoints, speed)
                aug = True

            if np.random.uniform(0, 1) < 0.5:
                keypoints = flip_keypoints(keypoints)
                aug = True

        return keypoints

    def query_class(self, class_name, _max=5):
        key_querys = [self.list_key.index(x) for x in self.list_key if class_name in x][:_max]
        return [self[i] for i in key_querys]

    def __len__(self):
        return len(self.list_key)

    def normalize_keypoints(self, keypoints):
        for i in range(keypoints.shape[0]):
            for idx in self.joint_idxs:
                keypoints[i, idx, :] = self._normalize_part(keypoints[i, idx, :])
        return keypoints

    @staticmethod
    def _normalize_part(keypoint):
        assert keypoint.shape[-1] == 2, "Keypoints must have x, y"
        x_coords = keypoint[:, 0]
        y_coords = keypoint[:, 1]
        min_x, min_y = min(x_coords), min(y_coords)
        max_x, max_y = max(x_coords), max(y_coords)
        w = max_x - min_x
        h = max_y - min_y

        if w > h:
            delta_x = 0.05 * w
            delta_y = delta_x + ((w - h) / 2)
        else:
            delta_y = 0.05 * h
            delta_x = delta_y + ((h - w) / 2)

        s_point = [max(0, min(min_x - delta_x, 1)), max(0, min(min_y - delta_y, 1))]
        e_point = [max(0, min(max_x + delta_x, 1)), max(0, min(max_y + delta_y, 1))]

        assert s_point[0] >= 0, f"Strating point[0] < 0 is {s_point[0]}"
        assert s_point[1] >= 0, f"Strating point[1] < 0 is {s_point[1]}"
        assert e_point[0] >= 0, f"Ending_point point[0] < 0 is {e_point[0]}"
        assert e_point[1] >= 0, f"Ending_point point[1] < 0 is {e_point[1]}"

        if (e_point[0] - s_point[0]) != 0.0:
            keypoint[:, 0] = (keypoint[:, 0] - s_point[0]) / (e_point[0] - s_point[0])
        if (e_point[1] - s_point[1]):
            keypoint[:, 1] = (keypoint[:, 1] - s_point[1]) / (e_point[1] - s_point[1])

        return keypoint


    def padding_keypoint(self, keypoint, max_len):
        T, K, C = keypoint.shape
        padding = torch.zeros((max_len - T, K, C))
        kp_padd = torch.cat((keypoint, padding), dim=0)

        return kp_padd

    def data_collator(self, batch):
        keypoints = [x[0] for x in batch]
        labels = [x[1] for x in batch]
        lengths = [x.shape[0] for x in keypoints]
        max_len = max(lengths)

        attention_masks = []
        keypoint_paddings = []
        for keypoint in keypoints:
            T, K, C = keypoint.shape
            if T < max_len:
                mask = torch.cat((torch.ones(T), torch.zeros(max_len - T)), dim=-1)
                kp_padd = self.padding_keypoint(keypoint, max_len)
            else:
                mask = torch.ones(T)
                kp_padd = keypoint
            attention_masks.append(mask)
            keypoint_paddings.append(kp_padd)

        keypoints = torch.stack(keypoint_paddings)
        attention_mask = torch.stack(attention_masks)

        labels = torch.stack(labels)

        return {
            "keypoints": keypoints,
            "attention_mask": attention_mask,
            "labels": labels,
        }


if __name__ == "__main__":
    x = np.random.randint(0, 3)
    print(type(x))