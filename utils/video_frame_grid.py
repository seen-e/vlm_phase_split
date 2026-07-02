"""
视频帧处理工具：resize、时间戳/帧号叠加、帧拼图网格化。

功能：
1. 读取视频并按指定宽高 resize 每一帧
2. 在每一帧左上角叠加黑底白字的时间戳或帧号（二选一）
3. 将所有帧按行列数分组拼接为网格图，帧序从上到下、从左到右递增
4. 视频较长时自动生成多张网格图，不足一页的末张用黑色帧补齐
5. 支持帧采样步长（frame_step），可跳帧处理
6. 将拼接图批量保存到指定目录

依赖：opencv-python, Pillow
"""

from __future__ import annotations

from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """加载字体，优先尝试等宽字体，回退到默认字体。"""
    font_paths = [
        # Windows
        "C:/Windows/Fonts/consola.ttf",
        "C:/Windows/Fonts/cour.ttf",
        # macOS
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/Courier.dfont",
        # Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ]
    for fp in font_paths:
        try:
            return ImageFont.truetype(fp, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def _frame_to_pil(frame: cv2.Mat) -> Image.Image:
    """将 OpenCV BGR 帧转换为 PIL RGB Image。"""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _draw_overlay(
    pil_img: Image.Image,
    text: str,
    *,
    font_size: int = 20,
    margin: int = 6,
    padding: int = 4,
    bg_color: tuple[int, int, int] = (0, 0, 0),
    fg_color: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """在图片左上角绘制黑底白字的文本叠加层。"""
    draw = ImageDraw.Draw(pil_img)
    font = _load_font(font_size)

    # 使用 textbbox 获取精确的文本边界
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # 背景矩形
    box_x0 = margin
    box_y0 = margin
    box_x1 = margin + text_w + padding * 2
    box_y1 = margin + text_h + padding * 2

    draw.rectangle([box_x0, box_y0, box_x1, box_y1], fill=bg_color)
    draw.text((box_x0 + padding, box_y0 + padding), text, fill=fg_color, font=font)

    return pil_img


def _format_timestamp(seconds: float) -> str:
    """将秒数格式化为 HH:MM:SS.mmm 时间戳字符串。"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def process_video_to_grid(
    video_path: str | Path,
    output_dir: str | Path,
    *,
    width: int,
    height: int,
    fps: float,
    rows: int,
    cols: int,
    overlay_mode: str = "frame_number",
    font_size: int = 20,
    frame_step: int = 1,
    output_format: str = "png",
) -> list[Path]:
    """
    遍历视频所有帧，逐帧 resize、叠加时间戳/帧号后，按行列数拼接为网格图。
    视频较长时会自动生成多张网格图，保存到指定目录。

    参数
    ----
    video_path : str | Path
        输入视频文件路径。
    output_dir : str | Path
        输出目录，网格图以 ``000.png``、``001.png`` … 命名保存。
    width : int
        每帧 resize 后的宽度（像素）。
    height : int
        每帧 resize 后的高度（像素）。
    fps : float
        视频帧率（用于计算时间戳），会优先使用视频元数据中的帧率。
    rows : int
        网格行数。
    cols : int
        网格列数（每张网格图包含 rows × cols 帧）。
    overlay_mode : str
        叠加模式，二选一：
        - ``"frame_number"``  — 显示帧序号（从 1 开始）
        - ``"timestamp"``     — 显示时间戳，格式 HH:MM:SS.mmm
    font_size : int
        叠加文本的字体大小，默认 20。
    frame_step : int
        帧采样步长，默认 1 表示不跳帧；设为 N 则每隔 N 帧取一帧。
    output_format : str
        输出图片格式（``"png"`` / ``"jpg"``），默认 ``"png"``。

    返回
    ----
    list[Path]
        生成的所有网格图文件路径列表。

    示例
    ----
    >>> paths = process_video_to_grid(
    ...     "input.mp4", "output_grids",
    ...     width=320, height=240, fps=30.0,
    ...     rows=4, cols=5,
    ...     overlay_mode="timestamp",
    ... )
    >>> print(paths)
    [PosixPath('output_grids/000.png'), PosixPath('output_grids/001.png')]
    """
    if overlay_mode not in ("frame_number", "timestamp"):
        raise ValueError(f"overlay_mode 必须为 'frame_number' 或 'timestamp'，实际为: {overlay_mode!r}")
    if frame_step < 1:
        raise ValueError(f"frame_step 必须 >= 1，实际为: {frame_step}")

    frames_per_grid = rows * cols
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频文件: {video_path}")

    actual_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS) or fps

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bucket: list[Image.Image] = []  # 当前拼图的帧缓存
    grid_index = 0
    global_frame_idx = 0  # 累计已处理的原始帧数（用于帧号和跳帧判断）
    saved_paths: list[Path] = []

    def _flush_bucket() -> None:
        """将 bucket 中的帧拼接为一张网格图并保存。不足部分用黑色帧补齐。"""
        nonlocal grid_index
        if not bucket:
            return

        # 补齐到 frames_per_grid
        while len(bucket) < frames_per_grid:
            bucket.append(Image.new("RGB", (width, height), (0, 0, 0)))

        grid = Image.new("RGB", (width * cols, height * rows))
        for i, frame_img in enumerate(bucket):
            r = i // cols
            c = i % cols
            grid.paste(frame_img, (c * width, r * height))

        filename = out_dir / f"{grid_index:03d}.{output_format}"
        grid.save(str(filename))
        saved_paths.append(filename)
        grid_index += 1
        bucket.clear()

    # 逐帧读取
    for idx in range(actual_total):
        # 按步长跳帧
        if idx % frame_step != 0:
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            global_frame_idx += 1
            continue

        # Resize
        resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        pil_img = _frame_to_pil(resized)

        # 叠加文本
        if overlay_mode == "timestamp":
            text = _format_timestamp(idx / actual_fps)
        else:
            text = str(global_frame_idx + 1)  # 帧号从 1 开始，按处理后计数

        pil_img = _draw_overlay(pil_img, text, font_size=font_size)
        bucket.append(pil_img)
        global_frame_idx += 1

        # 攒够一页就写出
        if len(bucket) >= frames_per_grid:
            _flush_bucket()

    cap.release()

    # 处理剩余的不足一页的帧
    _flush_bucket()

    return saved_paths


if __name__ == "__main__":
    # 简单自测（需本地有 test.mp4）
    paths = process_video_to_grid(
        r"C:\Users\34927\Desktop\robot_mind2\camera_extrinsics_intrinsics\franka\robogene_twoArm_franka_adjust_black_computer_stand\videos\chunk-000\observation.rgb_images.camera_front\episode_000000.mp4",
        "output_grids",
        width=320,
        height=240,
        fps=30.0,
        rows=4,
        cols=5,
        overlay_mode="frame_number",
    )
    print(f"生成 {len(paths)} 张网格图:", paths)
