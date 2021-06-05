from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import random
import logging
from tqdm import tqdm

import io
import tensorflow as tf
from PIL import Image
from collections import namedtuple, OrderedDict

def download_original_data():
    if 'ShelfImages' not in os.listdir():
        #Downloading the train and test images
        print('\nCollecting ShelfImages: ')
        os.system('wget https://storage.googleapis.com/open_source_datasets/ShelfImages.tar.gz')
        os.system('tar -xvzf ShelfImages.tar.gz')
        os.system('rm -rf ShelfImages.tar.gz')
    else:
        print('\nFound existing ShelfImages. Continuing...')

    if 'grocerydataset' not in os.listdir():
        print('\n\nCollecting grocerydataset...')
        os.system('git clone https://github.com/gulvarol/grocerydataset')
    else:
        print('\nFound existing grocerydataset. Continuing...')

    if 'models' not in os.listdir():
        print('\n\nCollecting Tensorflow-models...')
        os.system('git clone https://github.com/tensorflow/models/')
    else:
        print('Found existing tf-models. Continuing...')

    if 'efficientdet_d0_coco17_tpu-32' not in os.listdir():
        print('\n\nCollecting EfficientDet D0 model...')
        os.system('wget http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d0_coco17_tpu-32.tar.gz')
        os.system('tar -xvzf efficientdet_d0_coco17_tpu-32.tar.gz')
        os.system('rm -rf efficientdet_d0_coco17_tpu-32.tar.gz')
    else:
        print('Found existing EffDet D0. Continuing...')

def install_dependencies():
    os.system('pip install -r packages.txt')
    import pycocotools, tf_slim, lvis

def get_image_filenames():
    path = 'ShelfImages/'
    img_train = os.listdir(path + 'train')
    img_test  = os.listdir(path + 'test')
    total_img = img_train + img_test
    print("Total Images:", len(total_img))
    print("Train images:", len(img_train), "(", round(len(img_train)/len(total_img) * 100, 1), "%)")
    print("Test images:", len(img_test), "(", round(len(img_test)/len(total_img) * 100, 1), "%)")
    return img_train, img_test

def get_annotations():
    f = open('grocerydataset/annotation.txt')
    img_to_annot = {}
    for i in tqdm(f):
        img_name = i.split()[0]
        annot = i.split()[2:]
        img_to_annot[img_name] = annot
    return img_to_annot

def get_pandas_data(img_to_annot, img_filenames):
    data_dict = {'filename': [], 'width': [], 'class': [], 'height': [], 'xmin': [], 'xmax': [], 'ymin': [], 'ymax': []}
    for fname in img_filenames:
        for i in range(0, len(img_to_annot[fname]), 5):
            x = int(img_to_annot[fname][i])
            y = int(img_to_annot[fname][i + 1])
            w = int(img_to_annot[fname][i + 2])
            h = int(img_to_annot[fname][i + 3])

            data_dict['filename'].append(fname)
            data_dict['width'].append(w)
            data_dict['class'].append('Product')
            data_dict['height'].append(h)
            data_dict['xmin'].append(x)
            data_dict['xmax'].append(x + w)
            data_dict['ymin'].append(y)
            data_dict['ymax'].append(y + h)

    df_data = pd.DataFrame(data_dict)
    return df_data

def class_text_to_int(row_label):
    if row_label == 'Product':
        return 1
    else:
        None

def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]

def create_tf_example(group, path):
    with tf.io.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    from models.research.object_detection.utils import dataset_util
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))

    return tf_example

def create_tfrecords():
    for split_name in ['test', 'train']:
        os.system('mkdir tfrecords')
        output_path = "tfrecords/{}.record".format(split_name)

        writer = tf.io.TFRecordWriter(output_path)

        images_path = 'ShelfImages/{}'.format(split_name)
        examples = pd.read_csv("csv_labels/{}_labels.csv".format(split_name))

        grouped = split(examples, 'filename')
        for group in grouped:
            tf_example = create_tf_example(group, images_path)
            writer.write(tf_example.SerializeToString())

        writer.close()
        output_path = os.path.join(os.getcwd(), output_path)
        print('Successfully created the TFRecords: {}'.format(output_path))

def check_config_files():
    assert os.path.isfile('training/labelmap.pbtxt') == True
    assert os.path.isfile('training/effdet.config') == True

def prepare_data():
    #Install and import the reuiqred packages and libraries
    install_dependencies()
    #Download the original github repo and image data
    download_original_data()
    img_train, img_test = get_image_filenames()
    img_to_annot = get_annotations()

    #Creating training and test dataset in pandas
    df_train = get_pandas_data(img_to_annot, img_train)
    df_test  = get_pandas_data(img_to_annot, img_test)
    os.system('mkdir csv_labels')
    df_train.to_csv('csv_labels/train_labels.csv')
    df_test.to_csv('csv_labels/test_labels.csv')

    #Create tfrecords from csv data
    create_tfrecords()
    os.system('mkdir training')
    check_config_files()

if __name__ == "__main__":
    prepare_data()
