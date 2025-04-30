import os
import sys

import torch as th
from torchvision.utils import save_image
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))


from decalib.deca_with_smirk import DECA
from decalib.utils.config import cfg as deca_cfg
from data_utils.transfer_utils import get_image_dict

# Build DECA
deca_cfg.model.use_tex = True
deca_cfg.model.tex_path = "./decalib/data/FLAME_texture.npz"
deca_cfg.model.tex_type = "FLAME"
deca = DECA(config=deca_cfg, device="cuda")



def get_render(source, target, save_file):

    src_dict = get_image_dict(source, 512, True)
    tar_dict = get_image_dict(target, 512, True)
    # ===================get DECA codes of the target image===============================
    tar_cropped = tar_dict["image"].unsqueeze(0).to("cuda")
    imgname = tar_dict["imagename"]

    with th.no_grad():
        tar_code = deca.encode(tar_cropped)
    tar_image = tar_dict["original_image"].unsqueeze(0).to("cuda")
    # ===================get DECA codes of the source image===============================
    src_cropped = src_dict["image"].unsqueeze(0).to("cuda")
    with th.no_grad():
        src_code = deca.encode(src_cropped)
    # To align the face when the pose is changing
    src_ffhq_center = deca.decode(src_code, return_ffhq_center=True)
    tar_ffhq_center = deca.decode(tar_code, return_ffhq_center=True)

    src_tform = src_dict["tform"].unsqueeze(0)
    src_tform = th.inverse(src_tform).transpose(1, 2).to("cuda")
    src_code["tform"] = src_tform

    tar_tform = tar_dict["tform"].unsqueeze(0)
    tar_tform = th.inverse(tar_tform).transpose(1, 2).to("cuda")
    tar_code["tform"] = tar_tform

    src_image = src_dict["original_image"].unsqueeze(0).to("cuda")  # 平均的参数
    tar_image = tar_dict["original_image"].unsqueeze(0).to("cuda")

    # code 1 means source code, code 2 means target code
    code1, code2 = {}, {}
    for k in src_code:
        code1[k] = src_code[k].clone()

    for k in tar_code:
        code2[k] = tar_code[k].clone()

    code1["pose"][:, :3] = code2["pose"][:, :3]
    code1['exp'] = code2['exp']
    code1['pose'][:, 3:] = tar_code['pose'][:, 3:]

    opdict, _ = deca.decode(
        code1,
        render_orig=True,
        original_image=tar_image,
        tform=src_code["tform"],
        align_ffhq=False,
        ffhq_center=src_ffhq_center,
        imgpath=target
    )

    depth = opdict["depth_images"].detach()
    normal = opdict["normal_images"].detach()
    render = opdict["rendered_images"].detach()
    os.makedirs(f'./transfers/{save_file}/depth', exist_ok=True)
    os.makedirs(f'./transfers/{save_file}/normal', exist_ok=True)
    os.makedirs(f'./transfers/{save_file}/render', exist_ok=True)

    save_image(depth[0], f"./transfers/{save_file}/depth/{imgname}")
    save_image(normal[0], f"./transfers/{save_file}/normal/{imgname}")
    save_image(render[0], f"./transfers/{save_file}/render/{imgname}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sor_img",
        type=str,
        default='/home/mengting/projects/diffusionRig/myscripts/papers/example42/boy2_cropped.jpg',
        required=False
    )
    parser.add_argument(
        "--driving_path",
        type=str,
        default='/home/mengting/Desktop/frames_1500_updated/1fsFQ2gF4oE_0/images',
        required=False
    )
    parser.add_argument(
        "--save_name",
        type=str,
        default='example1',
        required=False
    )

    args = parser.parse_args()
    images = sorted(os.listdir(args.driving_path))
    for image in images:
        cur_image_path = os.path.join(args.driving_path, image)
        get_render(args.sor_img, cur_image_path, args.save_name)

    print('done')