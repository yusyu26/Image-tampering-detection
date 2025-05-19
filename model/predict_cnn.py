import tensorflow as tf
import numpy as np
import cv2
from skimage.feature import greycomatrix
import os

# 入力画像パス
image_path = "data/Au_ani_00001.jpg"
# モデルチェックポイント
checkpoint_path = "checkpoints/model.ckpt"

# Scharrエッジ抽出
def scharr(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sx = cv2.Scharr(g, cv2.CV_32F, 1, 0)
    sy = cv2.Scharr(g, cv2.CV_32F, 0, 1)
    return cv2.convertScaleAbs(cv2.magnitude(sx, sy))

# GLCM特徴抽出
def extract_glcm_feature(img):
    edge = scharr(img)
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi,
              5*np.pi/4, 3*np.pi/2, 7*np.pi/4]
    glcm_channels = []
    for angle in angles:
        glcm = greycomatrix(edge, [1], [angle], 256, symmetric=True, normed=True)
        glcm_channels.append(glcm[..., 0, 0].astype(np.float32))  # shape: (256, 256)
    return np.stack(glcm_channels, axis=-1)  # shape: (256, 256, 8)
# モデル構築
import network  # 事前に network.py が正しく機能すること

def predict():
    # 画像読み込みとGLCM変換
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ 画像が読み込めません: {image_path}")
        return
    glcm = extract_glcm_feature(img)
    glcm = np.expand_dims(glcm, axis=0)

    tf.reset_default_graph()
    inputs = tf.placeholder(tf.float32, shape=[None, 256, 256, 8], name='input')
    logits, _ = network.DNNs(inputs, keep_prob=1.0, num_classes=2, is_training=False)
    preds = tf.nn.softmax(logits)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, checkpoint_path)
        probs = sess.run(preds, feed_dict={inputs: glcm})
        label = np.argmax(probs, axis=1)[0]
        confidence = probs[0][label] * 100

        label_str = "改ざん画像 (Tampered)" if label == 1 else "未改ざん画像 (Authentic)"
        print(f"✅ 結果: {label_str} [信頼度: {confidence:.2f}%]")

if __name__ == "__main__":
    predict()
