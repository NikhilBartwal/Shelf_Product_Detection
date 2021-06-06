import numpy as np
import os
from PIL import Image
import streamlit as st
import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from train_inference import run_inference_for_single_image as run_inference
from train_inference import show_inference

def import_inference_model():
    try:
        model = tf.saved_model.load('inference_graph/saved_model')
    except:
        raise Exception('Please train the model first or import it from repo!')

    try:
        category_index = label_map_util.create_category_index_from_labelmap(
                            'training/labelmap.pbtxt',
                            use_display_name=True
        )
    except:
        raise Exception('Please install TF object_detection API first.')
    return category_index, detection_model

def deploy_model():
    model = import_inference_model()
    category_index, detection_model = import_inference_model()
    st.write('My First Web App: BoxtheShelf!')
    option = st.sidebar('Please choose the task:', ['Classification, Object Detection'])
    st.write('You chose:', option)

if __name__ == '__main__':
    deploy_model()
