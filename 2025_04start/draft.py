import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops

# 画像読み込み（グレースケール）
img = cv2.imread('Lounge1.jpg', cv2.IMREAD_GRAYSCALE)

# Scharrフィルタでエッジ検出
scharr_x = cv2.Scharr(img, cv2.CV_64F, 1, 0)
scharr_y = cv2.Scharr(img, cv2.CV_64F, 0, 1)
scharr = cv2.magnitude(scharr_x, scharr_y)

# エッジ画像を8bit化（GLCM計算に必要）
scharr_uint8 = cv2.normalize(scharr, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

# GLCM行列を作成
glcm = graycomatrix(
    scharr_uint8,  # 入力画像
    distances=[1],  # 隣接距離（ピクセル）
    angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],  # 方向（0度, 45度, 90度, 135度）
    symmetric=True,  # 対称性あり
    normed=True      # 正規化あり
)

# 特徴量を抽出
contrast = graycoprops(glcm, 'contrast')
homogeneity = graycoprops(glcm, 'homogeneity')
ASM = graycoprops(glcm, 'ASM')  # エネルギー
correlation = graycoprops(glcm, 'correlation')

# 表示
print("Contrast:", contrast)
print("Homogeneity:", homogeneity)
print("ASM:", ASM)
print("Correlation:", correlation)
