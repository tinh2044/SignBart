import numpy as np


def rotate_keypoints(frames, origin, angle_degrees):
    angle_radians = np.radians(angle_degrees)

    cos_angle = np.cos(angle_radians)
    sin_angle = np.sin(angle_radians)

    rotation_matrix = np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])

    shifted_points = frames - np.array(origin)

    rotated_points = np.einsum('ij,klj->kli', rotation_matrix, shifted_points)

    rotated_frames = rotated_points + np.array(origin)

    return rotated_frames


def noise_injection(frames, noise_level):
    noise = np.random.normal(0, noise_level, frames.shape)

    noisy_frames = frames + noise
    return noisy_frames


def clip_frame(frames, tgt_frame, is_uniform):
    t = frames.shape[0]
    if is_uniform:
        indices = np.clip(np.linspace(0, t - 1, tgt_frame).astype(int), 0, t - 1)
    else:
        indices = np.sort(np.random.choice(np.arange(t), size=tgt_frame, replace=True))
    selected_frames = frames[indices]
    
    return selected_frames


def time_warp_uniform(frames, scale):
    t = frames.shape[0]
    new_t = int(t * scale)

    indices = np.clip(np.linspace(0, t - 1, new_t).astype(int), 0, t - 1)
    
    warped_frames = frames[indices]

    return warped_frames


def flip_keypoints(keypoints):
    flipped_keypoints = keypoints.copy()

    flipped_keypoints[..., 0] = 1 - flipped_keypoints[..., 0]

    return flipped_keypoints
