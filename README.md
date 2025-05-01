# MagicPortrait

**MagicPortrait: Temporally Consistent Face Reenactment with 3D Geometric Guidance**
<br>
[Mengting Wei](),
[Yante Li](),
[Tuomas Varanka](),
[Yan Jiang](),
[Licai Sun](),
[Guoying Zhao]()
<br>

_[arXiv](https://arxiv.org/abs/2504.21497) | [Model](https://huggingface.co/mengtingwei/MagicPortrait)_

This repository contains the example inference script for the MagicPortrait-preview model.

https://github.com/user-attachments/assets/e7f1c4fd-e817-4940-ab0a-a440bf71183c

## Installation

```bash
conda create -n mgp python=3.10 -y
conda activate mgp
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt241/download.html
```

## Inference of the Model


### Step 1: Download pre-trained models

Download our models from Huggingface.

```bash
huggingface-cli download --resume-download mengtingwei/MagicPortrait --local-dir ./pre_trained
```
### Step 2: Setup necessary libraries for face motion transfer
0. Put the downloaded `third_party_files` in the last step under the project directory `./`.
1. Visit [DECA Github](https://github.com/yfeng95/DECA?tab=readme-ov-file) to download the pretrained `deca_model.tar`. 
2. Visit [FLAME website](https://flame.is.tue.mpg.de/download.php) to download `FLAME 2020` and extract `generic_model.pkl`.
3. Visit [FLAME website](https://flame.is.tue.mpg.de/download.php) to download `FLAME texture space` and extract `FLAME_texture.npz`.
4. Visit [DECA' data page](https://github.com/yfeng95/DECA/tree/master/data) and download all files.
5. Visit [SMIRK website](https://github.com/georgeretsi/smirk) to download `SMIRK_em1.pt`.
6. Place the files in their corresponding locations as specified below.

```plaintext
decalib
    data/
      deca_model.tar
      generic_model.pkl
      FLAME_texture.npz
      fixed_displacement_256.npy
      head_template.obj
      landmark_embedding.npy
      mean_texture.jpg
      texture_data_256.npy
      uv_face_eye_mask.png
      uv_face_mask.png
    ...
    smirk/
      pretrained_models/
        SMIRK_em1.pt
      ...
    ... 
```

### Step 3: Process the identity image and driving video

> As our model is designed to focus only on the face, 
> you should crop the face from your images or videos if they are full-body shots.
>  However, **if your images or videos already contain only the face and the aspect ratio is approximately 1:1, 
> you can simply resize them without doing the following pre-processing steps.**.

1. Crop the face from an image:

```python
python crop_process.py --sign image --img_path './assets/boy.jpeg' --save_path './assets/boy_cropped.jpg'
```

2. Crop the faces sequence from the driving video.

   * If you have a video
      ```bash
       mkdir ./assets/driving_images
       ffmpeg -i ./assets/driving.mp4 ./assets/driving_images/frame_%04d.jpg
      ```
 Crop face from the driving images.
```python
python crop_process.py --sign video --video_path './assets/driving_images' --video_imgs_dir './assets/driving_images_cropped'
```
   
3. Retrieve guidance images using DECA and SMIRK models.

```python
python render_and_transfer.py --sor_img './assets/boy_cropped.jpg' --driving_path './assets/driving_images_cropped' --save_name example1
```
The guidance will be saved in the `./transfers` directory.

### Step 4: Inference

Update the model and image directories in `./configs/inference/inference.yaml` to match your own file locations.

Then run:
```python
python inference.py
```

## Acknowledgement

Our work is made possible thanks to open-source pioneering
3D face reconstruction works (including [DECA](https://github.com/yfeng95/DECA?tab=readme-ov-file) 
and [SMIRK](https://github.com/georgeretsi/smirk)) and
a high-quality talking-video dataset [CelebV-HQ](https://celebv-hq.github.io).

## Contact
Open an issue here or email [mengting.wei@oulu.fi]().