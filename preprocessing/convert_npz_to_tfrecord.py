import os
import numpy as np
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm

input_dir = "glcm"
output_path = "tfrecords/casia2.tfrecord"
os.makedirs("tfrecords", exist_ok=True)

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

with tf.io.TFRecordWriter(output_path) as writer:
    for path in tqdm(sorted(Path(input_dir).glob("*.npz"))):
        data = np.load(path)
        glcm = data["glcm"].astype(np.float32)
        label = int(data["label"])
        ex = tf.train.Example(features=tf.train.Features(feature={
            "glcm": _bytes_feature(glcm.tobytes()),
            "label": _int64_feature(label)
        }))
        writer.write(ex.SerializeToString())
