import os
from pathlib import Path
import cv2
import numpy as np
from skimage.feature.texture import greycomatrix
from tqdm import tqdm

# データ元と保存先
root_dir = "data/CASIA2.0"      # 入力元
out_dir = "glcm"                # 出力先
os.makedirs(out_dir, exist_ok=True)

# GLCMパラメータ
T = 192                         # 論文で使用されたトリミング上限
angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # 0°, 45°, 90°, 135°

def extract_edge(img_channel):
    """ Scharrオペレータ + クリップ＆整数化（論文準拠） """
    sx = cv2.Scharr(img_channel, cv2.CV_32F, 1, 0)
    sy = cv2.Scharr(img_channel, cv2.CV_32F, 0, 1)
    magnitude = cv2.magnitude(sx, sy)
    edge = np.clip(magnitude, 0, T - 1).astype(np.uint8)
    return edge

# ラベルごとに処理
for label, subdir in enumerate(["Au", "Tp"]):
    files = sorted(
        p for ext in ("*.jpg", "*.jpeg", "*.png")
        for p in Path(root_dir, subdir).rglob(ext)
    )

    for path in tqdm(files, desc=f"[{subdir}]"):
        img = cv2.imread(str(path))
        if img is None:
            print(f"skip: {path}")
            continue

        # YCrCb変換 → Cr/Cb抽出
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        cr, cb = ycrcb[:, :, 1], ycrcb[:, :, 2]

        glcm_stack = []
        for comp in [cr, cb]:  # Cr と Cb それぞれ処理
            edge = extract_edge(comp)
            glcm = greycomatrix(edge, [1], angles, levels=T, symmetric=True, normed=True)
            for i in range(len(angles)):
                glcm_stack.append(glcm[:, :, 0, i])  # (T, T)

        glcm_all = np.stack(glcm_stack, axis=-1)  # → shape: (T, T, 8)

        np.savez_compressed(
            os.path.join(out_dir, path.stem + ".npz"),
            glcm=glcm_all.astype(np.float32),
            label=label,
        )
