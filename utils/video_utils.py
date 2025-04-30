import os
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
import imageio
from einops import rearrange
import torchvision.transforms as transforms


def save_videos_from_pil(pil_images, path, fps=15, crf=23):

    save_fmt = Path(path).suffix
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if save_fmt == ".mp4":
        with imageio.get_writer(path, fps=fps) as writer:
            for img in pil_images:
                img_array = np.array(img)  # Convert PIL Image to numpy array
                writer.append_data(img_array)

    elif save_fmt == ".gif":
        pil_images[0].save(
            fp=path,
            format="GIF",
            append_images=pil_images[1:],
            save_all=True,
            duration=(1 / fps * 1000),
            loop=0,
        )
    else:
        raise ValueError("Unsupported file type. Use .mp4 or .gif.")


def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=6, fps=24):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    height, width = videos.shape[-2:]
    outputs = []

    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)  # (c h w)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)  # (h w c)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        x = Image.fromarray(x)

        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)

    save_videos_from_pil(outputs, path, fps)


def resize_tensor_frames(video_tensor, new_size):
    B, C, video_length, H, W = video_tensor.shape
    # Reshape video tensor to combine batch and frame dimensions: (B*F, C, H, W)
    video_tensor_reshaped = video_tensor.reshape(-1, C, H, W)
    # Resize using interpolate
    resized_frames = F.interpolate(
        video_tensor_reshaped, size=new_size, mode="bilinear", align_corners=False
    )
    resized_video = resized_frames.reshape(B, C, video_length, new_size[0], new_size[1])

    return resized_video


def pil_list_to_tensor(image_list, size=None):
    to_tensor = transforms.ToTensor()
    if size is not None:
        tensor_list = [to_tensor(img.resize(size[::-1])) for img in image_list]
    else:
        tensor_list = [to_tensor(img) for img in image_list]
    stacked_tensor = torch.stack(tensor_list, dim=0)
    tensor = stacked_tensor.permute(1, 0, 2, 3)
    return tensor


def conatenate_into_video():
    gt_list = []
    gt_root = '/home/mengting/projects/champ_abls/no_exp_coeff/results/output_images'
    imgs = sorted(os.listdir(gt_root))
    for img in imgs:
        cur_img_path = os.path.join(gt_root, img)
        tmp_img = Image.open(cur_img_path)
        tmp_img = tmp_img.resize((512, 512))
        tmp_img = transforms.ToTensor()(tmp_img)
        gt_list.append(tmp_img)
    gt_list = torch.stack(gt_list, dim=1).unsqueeze(0)
    print(gt_list.shape)

    ref_image_path = '/home/mengting/Desktop/frames_1500_updated/4Z7qKXu9Sck_2/images/frame_0000.jpg'
    ref_image_pil = Image.open(ref_image_path)
    ref_image_w, ref_image_h = ref_image_pil.size
    video_length = len(imgs)
    ref_video_tensor = transforms.ToTensor()(ref_image_pil)[None, :, None, ...].repeat(
        1, 1, video_length, 1, 1
    )

    drive_list = []
    guidance_path = '/home/mengting/Desktop/frames_new_1500/2yj1P52T1X8_4/images'
    imgs = sorted(os.listdir(guidance_path))
    for i, img in enumerate(imgs):
        cur_img_path = os.path.join(guidance_path, img)
        tmp_img = Image.open(cur_img_path)
        tmp_img = transforms.ToTensor()(tmp_img)
        drive_list.append(tmp_img)
        if len(drive_list) == video_length:
            break
    drive_list = torch.stack(drive_list, dim=1).unsqueeze(0)
    print(drive_list.shape, ref_video_tensor.shape)

    save_dir = '/home/mengting/projects/champ_abls/no_exp_coeff/results/comparison'
    grid_video = torch.cat([drive_list, ref_video_tensor, gt_list], dim=0)
    save_videos_grid(grid_video, os.path.join(save_dir, "grid_wdrive_aniportrait.mp4"))

if __name__ == '__main__':
    conatenate_into_video()