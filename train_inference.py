import json
import os
import sys
import numpy as np
import pandas as pd
import six.moves.urllib as urllib
import tensorflow as tf
import tf_slim, pycocotools, lvis
import zipfile

from collections import defaultdict
from io import StringIO
from PIL import Image
from IPython.display import display
from data_prep import get_image_filenames, get_annotations
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from tqdm import tqdm

utils_ops.tf = tf.compat.v1
tf.gfile = tf.io.gfile

def train_and_eval():
    command = 'python models/research/object_detection/model_main_tf2.py '
    flag_1 = '--pipeline_config_path=training/effdet.config '
    flag_2 = '--model_dir=/training/effdet --alsologtostderr --num_train_steps=2000'
    os.system(command + flag_1 + flag_2)

    print('\n\n\nMODEL TRAINING COMPLETED. PERFORMING MODEL EVALUATIOn...\n\n')

    command = 'python models/research/object_detection/model_main_tf2.py '
    flag_1 = '--model_dir=training/effdet/ '
    flag_2 = '--pipeline_config_path=training/effdet.config '
    flag_3 = '--checkpoint_dir=training/effdet/ --eval_timeout=10'
    os.system(command + flag_1 + flag_2 + flag_3)

def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis]

    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key:value[0, :num_detections].numpy() for key,value in output_dict.items()}
    output_dict['num_detections'] = num_detections
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    return output_dict

def show_inference(model, image_path, category_index):
    image_np = np.array(Image.open(image_path))
    output_dict = run_inference_for_single_image(model, image_np)
    vis_util.visualize_boxes_and_labels_on_image_array(
    image_np,
    output_dict['detection_boxes'],
    output_dict['detection_classes'],
    output_dict['detection_scores'],
    category_index,
    max_boxes_to_draw=200,
    min_score_thresh=.5,
    use_normalized_coordinates=True,
    line_thickness=8)

    return image_np

def export_inference_model():
    category_index = label_map_util.create_category_index_from_labelmap(
                        'training/labelmap.pbtxt',
                        use_display_name=True)

    os.system('mkdir inference graph')

    command = 'python models/research/object_detection/exporter_main_v2.py '
    flag_1 = '--input_type=image_tensor '
    flag_2 = '--pipeline_config_path=training/effdet.config '
    flag_3 = '--trained_checkpoint_dir=training/effdet --output_directory=inference_graph'
    os.system(command + flag_1 + flag_2 + flag_3)
    detection_model = tf.saved_model.load('inference_graph/saved_model')

    return category_index, detection_model

def make_predicions(category_index, detection_model):
    _, img_test = get_image_filenames()
    img_to_annot = get_annotations()

    os.system('mkdir predictions')

    for image_name in tqdm(img_test):
        path = 'ShelfImages/test/' + image_name
        image = show_inference(detection_model, path, category_index)
        image = Image.fromarray(image)
        image.save('predictions/' + image_name)

    print("Test set predictions have been successfully saved in ./predictions/")
    return img_test, img_to_annot

def export_image2products_json(img_test, img_to_annot):
    img_to_prod = {}
    for image_name in tqdm(img_test):
        file = np.array(Image.open('ShelfImages/test/' + image))
        result = run_inference_for_single_image(detection_model, file)['detection_scores']
        #using a passing threshold of 0.5
        infer = sum(infer>0.5)
        img_to_prod[image] = int(infer)

    json_object = json.dumps(img_to_prod, indent = 4)
    with open("image2products_json.json", "w") as outfile:
        outfile.write(json_object)

    print('image2products.json successfully written...')

def train_inference():
    #Now the EffDet D0 model will be trained and evaluiated on the test set
    train_and_eval()
    #After the training the final model is loaded for making predictins on test set
    category_index, detection_model = port_inference_model()
    #Making final predictions on test set and saving modified images (with bounding boxes)
    img_test, img_to_annot = make_predicions(category_index, detection_model)
    #Export json files fole for evaluation purpose
    export_image2products_json(img_test, img_to_annot, detection_model)

if __name__ == '__main__':
    train_inference()
