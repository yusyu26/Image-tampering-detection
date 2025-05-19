import tensorflow as tf
import network

# TFRecordの読み込み処理
def _parse(example_proto):
    features = {
        'glcm': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64)
    }
    parsed = tf.parse_single_example(example_proto, features)
    glcm = tf.decode_raw(parsed['glcm'], tf.float32)
    glcm = tf.reshape(glcm, [192, 192, 8])
    label = tf.cast(parsed['label'], tf.int32)
    return glcm, label

def input_fn(tfrecord_path, batch_size):
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(_parse)
    dataset = dataset.shuffle(buffer_size=200)
    dataset = dataset.batch(batch_size)
    return dataset

def train():
    batch_size = 2
    num_epochs = 10
    learning_rate = 0.0005

    dataset = input_fn("tfrecords/casia2.tfrecord", batch_size)
    iterator = dataset.make_one_shot_iterator()
    images, labels = iterator.get_next()

    logits, _ = network.DNNs(images, keep_prob=0.5, num_classes=2, is_training=True)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    correct = tf.equal(tf.argmax(logits, 1), tf.cast(labels, tf.int64))
    acc = tf.reduce_mean(tf.cast(correct, tf.float32))

    saver = tf.train.Saver()

    with tf.Session() as sess:
        print("Initializing variables...")
        sess.run(tf.global_variables_initializer())
        print("Initialization complete.")

        for epoch in range(num_epochs):
            print(f"\n[Epoch {epoch+1}] Starting...")
            avg_loss = 0
            avg_acc = 0
            steps = 0

            try:
                while True:
                    _, l, a = sess.run([train_op, loss, acc])
                    avg_loss += l
                    avg_acc += a
                    steps += 1

                    # ↓ ステップごとの進捗表示（10ステップごとに表示）
                    if steps % 10 == 0:
                        print(f"  Step {steps} - Loss: {l:.4f}, Acc: {a:.4f}")
            except tf.errors.OutOfRangeError:
                pass

            if steps > 0:
                print(f"[Epoch {epoch+1}] Finished - Avg Loss: {avg_loss/steps:.4f}, Avg Acc: {avg_acc/steps:.4f}")
            else:
                print(f"[Epoch {epoch+1}] Skipped: No data processed.")

            saver.save(sess, "checkpoints/model.ckpt")
            print(f"[Epoch {epoch+1}] Model saved to checkpoints/model.ckpt")

if __name__ == "__main__":
    train()
