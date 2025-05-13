from pathlib import Path
import cv2, numpy as np, os
from skimage.feature import graycomatrix
from tqdm import tqdm

root = "data/CASIA2.0"
out  = "glcm"
os.makedirs(out, exist_ok=True)

def scharr(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sx = cv2.Scharr(g, cv2.CV_32F, 1, 0)
    sy = cv2.Scharr(g, cv2.CV_32F, 0, 1)
    return cv2.convertScaleAbs(cv2.magnitude(sx, sy))

for label, sub in enumerate(["Au", "Tp"]):
    #  jpg / jpeg / png を再帰的に収集
    files = sorted(
        p for ext in ("*.jpg", "*.jpeg", "*.png")
        for p in Path(root, sub).rglob(ext)
    )

    for p in tqdm(files, desc=f"{sub} imgs"):
        img = cv2.imread(str(p))
        if img is None:               # 読めない or 壊れたファイルはスキップ
            print(f"skip {p}")
            continue
        edge = scharr(img)
        glcm_full = graycomatrix(edge, [1], [0], 256, symmetric=True, normed=True)
        glcm = glcm_full[..., 0, 0]  # (256,256,1,1) → (256,256)
        np.savez_compressed(
            os.path.join(out, p.stem + ".npz"),
            glcm=glcm.astype(np.float32),
            label=label,
        )
