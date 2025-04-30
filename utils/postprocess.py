from moviepy import VideoFileClip, concatenate_videoclips, clips_array, ImageClip, ColorClip, concatenate_videoclips, vfx
import os
import cv2
from pyarrow import duration


def print_video_info(video, name):
    fps = video.fps
    duration = video.duration  # 秒数
    frame_count = int(fps * duration)

    print(f"视频{name}:")
    print(f"  帧率: {fps} fps")
    print(f"  时长: {duration:.2f} 秒")
    print(f"  总帧数: {frame_count} 帧")
    print("")

def images_to_video(image_dir, output_path, fps=24):
    # 获取图像文件并按自然顺序排序
    image_files = sorted([
        f for f in os.listdir(image_dir) if f.endswith((".jpg", ".jpeg", ".png"))])


    # 读取第一张图像获取尺寸
    first_image = cv2.imread(os.path.join(image_dir, image_files[0]))
    if first_image is None:
        raise ValueError("❌ 无法读取第一张图像。")
    height, width = first_image.shape[:2]

    # 设置视频编码器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # .mp4 输出
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for img_name in image_files:
        img_path = os.path.join(image_dir, img_name)
        frame = cv2.imread(img_path)

        # 自动 resize 不一致的尺寸
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))

        video_writer.write(frame)

    video_writer.release()
    print(f"✅ 视频保存成功: {output_path}")

def concatenate_videos():
    # 加载三个视频
    clip1 = VideoFileClip("/home/mengting/Desktop/tmp/champ/gpu_1/cut1.mp4")
    clip2 = VideoFileClip("/home/mengting/Desktop/tmp/champ/gpu_1/cut2.mp4")
    # clip3 = VideoFileClip("/home/mengting/Desktop/tmp/champ/gpu_1/cut3.mp4")

    # 拼接视频
    final_clip = concatenate_videoclips([clip1, clip2])

    # 输出到新文件
    final_clip.write_videofile("/home/mengting/Desktop/tmp/champ/gpu_1/output_combined.mp4", codec="libx264")


def array_videos():
    # 加载两个视频
    clip1 = VideoFileClip("/home/mengting/Desktop/tmp/champ/gpu_1/driving.mp4")
    clip2 = VideoFileClip("/home/mengting/Desktop/tmp/champ/gpu_1/animation.mp4")

    # 横向拼接（按帧对齐）
    final_clip = clips_array([[clip1, clip2]])  # 一行两列，clip1 左，clip2 右

    # 导出结果
    final_clip.write_videofile("/home/mengting/Desktop/tmp/champ/gpu_1/side_by_side.mp4", codec="libx264")

def set_fps():
    clip = VideoFileClip("/home/mengting/Desktop/tmp/champ/gpu_1/combined_with_transitions.mp4")

    # 设定新帧率（不会插帧）
    clip = clip.with_fps(24)

    # 写出时指定新帧率（帧数不变，播放速度加快）
    clip.write_videofile("/home/mengting/Desktop/tmp/champ/gpu_1/combined_with_transitions_24.mp4", fps=24, codec="libx264")


def add_static_image():
    # 加载 side-by-side 视频
    video = VideoFileClip("/home/mengting/Desktop/tmp/champ/gpu_1/output_fast_30fps.mp4")

    # 加载静态图像，并设置和视频一样的高度、持续整个视频时长
    image = ImageClip("/home/mengting/Desktop/tmp/champ/gpu_1/boy2_cropped.jpg").resized(height=video.h).with_duration(video.duration)

    # 横向拼接（静止图像 + 视频）
    final = clips_array([[image, video]])

    # 导出结果
    final.write_videofile("/home/mengting/Desktop/tmp/champ/gpu_1/final_output.mp4", codec="libx264")
# 179 拼接， 360-179 = 181
# 181和117拼接，181-117 = 64
# 64和150拼接，150-64 = 86
# video 7 and video 8
# video 1 is 275, video2 is 96, video 3 is 360, video5 is 117, video6 is 150, video7 is 98, video8 is 102
def print_info():
    # 加载两个视频
    video1 = VideoFileClip("/home/mengting/Desktop/tmp/champ/gpu_1/person1/video_part1.mp4")
    video2 = VideoFileClip("/home/mengting/Desktop/tmp/champ/gpu_1/person2/final_output.mp4")

    fps = video1.fps
    duration = video1.duration  # 秒
    frame_count = int(fps * duration)
    print(f"  帧率 (fps): {fps}")
    print(f"  总时长 (秒): {duration:.2f}")
    print(f"  总帧数: {frame_count}")
    print("")

def split_video():
    video = VideoFileClip("/home/mengting/Desktop/tmp/champ/gpu_1/person8/final_output.mp4")

    # 获取帧率和总帧数
    fps = video.fps
    total_frames = int(video.duration * fps)  # 或直接写 total_frames = 275

    # 设定拆分点
    first_part_frames = 98
    first_part_duration = first_part_frames / fps  # 以秒为单位
    total_duration = video.duration

    # 截取第一段（前96帧）
    clip1 = video.subclipped(0, first_part_duration)

    # 截取第二段（剩下的帧）
    clip2 = video.subclipped(first_part_duration, total_duration)

    # 保存
    clip1.write_videofile("/home/mengting/Desktop/tmp/champ/gpu_1/person8/video_part_1.mp4", codec="libx264", fps=fps)
    clip2.write_videofile("/home/mengting/Desktop/tmp/champ/gpu_1/person8/video_part_2.mp4", codec="libx264", fps=fps)


def up_and_down():
    # 加载两个视频
    clip1 = VideoFileClip("/home/mengting/Desktop/tmp/champ/gpu_1/person7/final_output.mp4")
    clip2 = VideoFileClip("/home/mengting/Desktop/tmp/champ/gpu_1/person8/video_part_1.mp4")


    # 上下拼接（垂直堆叠）
    final = clips_array([[clip1], [clip2]])

    # 输出拼接后的视频
    final.write_videofile("/home/mengting/Desktop/tmp/champ/gpu_1/stacked_vertical_5.mp4", codec="libx264", fps=clip1.fps)


def concate_all():
    # 视频文件名列表
    video_files = ["/home/mengting/Desktop/tmp/champ/gpu_1/stacked_vertical_1.mp4",
                   "/home/mengting/Desktop/tmp/champ/gpu_1/stacked_vertical_2.mp4",
                   "/home/mengting/Desktop/tmp/champ/gpu_1/stacked_vertical_3.mp4",
                   "/home/mengting/Desktop/tmp/champ/gpu_1/stacked_vertical_4.mp4",
                   "/home/mengting/Desktop/tmp/champ/gpu_1/stacked_vertical_5.mp4"]

    # 设置淡入/淡出时长（秒）
    fade_duration = 1.0
    black_duration = 0.5

    # 加载视频，添加淡入淡出
    clips = []
    for idx, file in enumerate(video_files):
        clip = VideoFileClip(file)
        clip = clip.with_effects([vfx.FadeIn(duration=fade_duration)])
        clip = clip.with_effects([vfx.FadeOut(duration=black_duration)])
        clips.append(clip)

        # 在中间插入黑色片段（不加最后一个）
        if idx < len(video_files) - 1:
            black = ColorClip(size=clip.size, color=(0, 0, 0), duration=black_duration)
            clips.append(black)

    # 拼接所有片段
    final_clip = concatenate_videoclips(clips, method="compose")

    # 导出视频
    final_clip.write_videofile("/home/mengting/Desktop/tmp/champ/gpu_1/combined_with_transitions.mp4", codec="libx264", fps=clips[0].fps)

if __name__ == '__main__':
    #concatenate_videos()
    #images_to_video('/home/mengting/Desktop/frames_1500_updated/1fsFQ2gF4oE_0/images', '/home/mengting/Desktop/tmp/champ/gpu_1/driving.mp4')
    #array_videos()
    # video1 = VideoFileClip("/home/mengting/Desktop/tmp/champ/gpu_1/driving.mp4")
    # print_video_info(video1, "driving")
    set_fps()
    #add_static_image()
    #print_info()
    #split_video()
    #up_and_down()
    #concate_all()