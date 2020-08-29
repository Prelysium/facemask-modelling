import time
import os
import hashlib

from absl import app, flags, logging
from absl.flags import FLAGS
import tensorflow as tf
import lxml.etree
import tqdm
import pandas as pd

flags.DEFINE_string("output_file", "./data/facemask_val.tfrecord", "output dataset")
flags.DEFINE_string("classes", "./data/facemask.names", "classes file")
flags.DEFINE_string("data_path", "data/val", "image data directory path")
flags.DEFINE_string("csv_path", "data/val_labels.csv", "annotations csv file path")


def build_example_facemask(annotation, class_map):
    img_path = os.path.join(FLAGS.data_path, annotation["filename"].iloc[0])
    img_raw = open(img_path, "rb").read()
    key = hashlib.sha256(img_raw).hexdigest()

    width = int(annotation["width"].iloc[0])
    height = int(annotation["height"].iloc[0])

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    counter_face = 0
    counter_mask = 0
    for i in range(len(annotation)):
        row = annotation.iloc[i]
        xmin.append(float(row["xmin"]) / width)
        ymin.append(float(row["ymin"]) / height)
        xmax.append(float(row["xmax"]) / width)
        ymax.append(float(row["ymax"]) / height)
        classes_text.append(row["class"].encode("utf8"))
        classes.append(class_map[row["class"]])
        if row["class"] == "face":
            counter_face += 1
        if row["class"] == "face_mask":
            counter_face += 1
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                "image/height": tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[height])
                ),
                "image/width": tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[width])
                ),
                "image/filename": tf.train.Feature(
                    bytes_list=tf.train.BytesList(
                        value=[annotation["filename"].iloc[0].encode("utf8")]
                    )
                ),
                "image/source_id": tf.train.Feature(
                    bytes_list=tf.train.BytesList(
                        value=[annotation["filename"].iloc[0].encode("utf8")]
                    )
                ),
                "image/key/sha256": tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[key.encode("utf8")])
                ),
                "image/encoded": tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[img_raw])
                ),
                "image/format": tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=["jpeg".encode("utf8")])
                ),
                "image/object/bbox/xmin": tf.train.Feature(
                    float_list=tf.train.FloatList(value=xmin)
                ),
                "image/object/bbox/xmax": tf.train.Feature(
                    float_list=tf.train.FloatList(value=xmax)
                ),
                "image/object/bbox/ymin": tf.train.Feature(
                    float_list=tf.train.FloatList(value=ymin)
                ),
                "image/object/bbox/ymax": tf.train.Feature(
                    float_list=tf.train.FloatList(value=ymax)
                ),
                "image/object/class/text": tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=classes_text)
                ),
                "image/object/class/label": tf.train.Feature(
                    int64_list=tf.train.Int64List(value=classes)
                ),
            }
        )
    )
    return example


def main(_argv):
    class_map = {
        name: idx for idx, name in enumerate(open(FLAGS.classes).read().splitlines())
    }
    logging.info("Class mapping loaded: %s", class_map)

    writer = tf.io.TFRecordWriter(FLAGS.output_file)

    data = pd.read_csv(FLAGS.csv_path)
    for image in data.filename.unique():
        image_ann = data[data.filename == image]
        tf_example = build_example_facemask(image_ann, class_map)
        writer.write(tf_example.SerializeToString())

    writer.close()
    logging.info("Done")


if __name__ == "__main__":
    app.run(main)
