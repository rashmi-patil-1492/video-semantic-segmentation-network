import cv2
import argparse
import time
import os
import six
import tensorflow as tf
from math import sqrt

import numpy as np
from tensorflow.python.platform import gfile
from tools.flow_utils import warp

from model.flownetmodel import FlowNets
from model.decisionmodel import Decision

from pathlib import Path
home = str(Path.home())
# VIDEO_FILE = home + '/data/video-segmentation/01TP_extract.avi'

VIDEO_FILE = './01TP_extract.avi'
process_original = 'True'

SEGNET_CHKPT = './resnet50_segnet_model/resnet50_segnet.pb'
DVS_FLOWNET_CHKPT = './dvs_net_flownets_checkpoints/finetune/'
DECISION_CHKPT = './decision_network_checkpoints/'

# target_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# Based on the experiments performed.
# The values will be picked based on the input target value
frames_table = [25, 25, 25, 25, 25, 24, 19, 16, 8, 2]


SAVE_DIR = './video-output/'
NUM_CLASSES = 11
TARGET = 80.0



seg_input_width = 608
seg_input_height = 416
seg_output_width = 304
seg_output_height = 208
input_size = [seg_input_height, seg_input_width]
original_width = 960
original_height = 720
original_size = [original_height, original_width]
camvid_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
cityscale_camvid_labels = [0, 1, 2, 4, 5, 6, 8, 10, 11, 13, 18]

decision_feature_size = [4, 5]

class DataLoaderError(Exception):
    pass


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Dynamic Video Segmentation Network")
    parser.add_argument("--video_file", type=str, default=VIDEO_FILE,
                        help="Path to the input video file.")
    parser.add_argument("--process_original", type=str, default=process_original,
                        help="if true, original file with have predicted annotations in rectangular form")
    parser.add_argument("--segnet_chkpt", type=str, default=SEGNET_CHKPT,
                        help="Where restore segnet model parameters from.")
    parser.add_argument("--dvs_flownet_chkpt", type=str, default=DVS_FLOWNET_CHKPT,
                        help="Where restore dvs flownet model parameters from.")
    parser.add_argument("--decision_chkpt", type=str, default=DECISION_CHKPT,
                        help="Where restore decision model parameters from.")
    parser.add_argument("--save_dir", type=str, default=SAVE_DIR,
                        help="Where to save segmented output.")
    parser.add_argument("--target", type=float, default=TARGET,
                        help="Confidence score threshold.")
    return parser.parse_args()

def get_image_array(image_input,
                    width, height,
                    imgNorm="sub_mean", ordering='channels_last'):
    """ Load image array from input """

    if type(image_input) is np.ndarray:
        # It is already an array, use it as it is
        img = image_input
    elif isinstance(image_input, six.string_types):
        if not os.path.isfile(image_input):
            raise DataLoaderError("get_image_array: path {0} doesn't exist"
                                  .format(image_input))
        img = cv2.imread(image_input, 1)
    else:
        raise DataLoaderError("get_image_array: Can't process input type {0}"
                              .format(str(type(image_input))))

    if imgNorm == "sub_and_divide":
        img = np.float32(cv2.resize(img, (width, height))) / 127.5 - 1
    elif imgNorm == "sub_mean":
        img = cv2.resize(img, (width, height))
        img = img.astype(np.float32)
        img[:, :, 0] -= 103.939
        img[:, :, 1] -= 116.779
        img[:, :, 2] -= 123.68
        img = img[:, :, ::-1]
    elif imgNorm == "divide":
        img = cv2.resize(img, (width, height))
        img = img.astype(np.float32)
        img = img/255.0

    if ordering == 'channels_first':
        img = np.rollaxis(img, 2, 0)
    return img


def to_bgr(image):
    img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=image)
    image_bgr = tf.concat(axis=2, values=[img_b, img_g, img_r])
    return image_bgr



def scale_fixed_size(key_image, current_image, output_shape):
    current_f = to_bgr(current_image)
    key_image = tf.cast(key_image, tf.float32) / 255.
    current_f = tf.cast(current_f, tf.float32) / 255.

    raw_height = tf.shape(key_image)[0]
    raw_width = tf.shape(key_image)[1]

    image_batch = tf.expand_dims(key_image, 0)
    current_f_batch = tf.expand_dims(current_f, 0)\

    raw_image_size = tf.shape(image_batch)
    image_f_size = tf.shape(current_f_batch)

    input_shape = tf.to_float(raw_image_size[1:3])

    scale_shape = output_shape / input_shape
    scale = tf.reduce_min(scale_shape)
    scaled_input_shape = tf.to_int32(tf.round(input_shape * scale))

    resized_image = tf.image.resize_nearest_neighbor(
        image_batch, scaled_input_shape)
    resized_current_f_image = tf.image.resize_nearest_neighbor(
        current_f_batch, scaled_input_shape)

    cropped_key_image = tf.image.resize_image_with_crop_or_pad(
        resized_image, output_shape[0] // 2, output_shape[1] // 2)

    cropped_current_f_image = tf.image.resize_image_with_crop_or_pad(
        resized_current_f_image, output_shape[0] // 2, output_shape[1] // 2)

    return cropped_key_image, cropped_current_f_image, resized_image


def scale_and_mask(key_image, current_image, input_size_to_rescale):
    cropped_key_image, cropped_current_image, resized_image = scale_fixed_size(key_image, current_image, input_size_to_rescale)
    return cropped_key_image, cropped_current_image, resized_image

def mask_channels(tensor, mask_indexes):
    short_list_tensors = []
    # mask_indexes contains indexes of output labels to consider
    # For camvid it would be 12
    for index in mask_indexes:
        short_list_tensors.append(tf.expand_dims(tensor[:,:,:,index], -1))
    return tf.concat(short_list_tensors, -1)

def load(saver, sess, ckpt_path):
    '''Load trained weights.

    Args:
      saver: TensorFlow saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    '''
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

def load_checkpoints(args, sess, variables_flownet, variables_decision):

    # Load segmentation model
    graph_def = tf.GraphDef()
    with gfile.FastGFile(args.segnet_chkpt, 'rb') as f:
        graph_def.ParseFromString(f.read())

    tf.import_graph_def(graph_def)
    print("Restored model parameters from {}".format(args.segnet_chkpt))

    ckpt = tf.train.get_checkpoint_state(args.dvs_flownet_chkpt)
    if ckpt and ckpt.model_checkpoint_path:
        loader = tf.train.Saver(var_list=variables_flownet)
        load(loader, sess, ckpt.model_checkpoint_path)
    else:
        print('No flownet checkpoint file found.')

    ckpt = tf.train.get_checkpoint_state(args.decision_chkpt)
    if ckpt and ckpt.model_checkpoint_path:
        loader = tf.train.Saver(var_list=variables_decision)
        load(loader, sess, ckpt.model_checkpoint_path)
    else:
        print('No decision net checkpoint file found.')

cmap = np.array([
    # sky
    [128, 128, 128],
    # building
    [128, 0, 0],
    # pole
    [192, 192, 128],
    # road
    [128, 64, 128],
    # pavement
    [60, 40, 222],
    # tree
    [128, 128, 0],
    # signsymbol
    [192, 128, 128],
    # fence
    [64, 64, 128],
    # car
    [64, 0, 128],
    # pedestrian
    [64, 64, 0],
    # bicyclist
    [0, 128, 192],
    # unlabelled
    [0, 0, 0]])
cmap_list = cmap.tolist()
class_names = np.array([
    'sky', 'building', 'pole', 'road', 'pavement', 'tree', 'signsymbol',
    'fence', 'car', 'pedestrian', 'bicyclist', 'unlabelled'])

class_names_enable_distance = ['pole', 'car', 'pedestrian', 'bicyclist']
class_names_list = class_names.tolist()

def laplacian(img):
    s = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    s = cv2.Laplacian(s, cv2.CV_16S, ksize=3)
    s = cv2.convertScaleAbs(s)
    return s

def add_rectangles_with_labels(img, original_img, enable_original='False'):
    s = laplacian(img)
    ret, binary = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        x1, y1, x2, y2 = x, y, x + w, y + h
        x_dist, y_dist = w, h
        if (x_dist < 20 and y_dist < 50) or (x_dist > original_width // 2 and y_dist > original_height // 2):
            continue

        region_x, region_y = (x1 + x2) // 2, (y1 + y2) // 2  # midpoint

        region = img[region_y, region_x]
        region_list = region.tolist()[::-1]
        if region_list in cmap_list:
            label = class_names_list[cmap_list.index(region_list)]
            """
            Distance conversion is based on following
            100 pixel = 0.000026458 kilometer
            https://www.translatorscafe.com/unit-converter/en-US/length/110-7/pixel-kilometer/
            """
            one_pixel_in_km = 0.000026458 / 100

            one_pixel_in_metres = 0.0002645833

            distance = sqrt(((original_width // 2) - region_x) * ((original_width // 2) - region_x) + (region_y - 0) * (region_y - 0))
            distance = 1.0 / (distance * one_pixel_in_metres)

            if label in class_names_enable_distance:
                label = label + " " + "{:.2f}".format(distance) + " m"

            if enable_original == 'True':
                # Draw black background rectangle
                cv2.rectangle(original_img, (x1, y1), (x1 + len(label) * 12, y1 - 20), (155, 155, 0), -1)
                cv2.putText(original_img, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
                cv2.rectangle(original_img, (x1, y1), (x2, y2), (155, 155, 0), 2)

            else:
                cv2.rectangle(img, (x1, y1), (x1 + len(label) * 12, y1 - 20), (155, 155, 0), -1)
                cv2.putText(img, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
                cv2.rectangle(img, (x1, y1), (x2, y2), (155, 155, 0), 2)


    if enable_original == 'True':
        return original_img

    return img


def vis_preparevideo(y, cmap):
    y = np.squeeze(y)
    r = y.copy()
    g = y.copy()
    b = y.copy()
    for l in range(0, len(cmap)):
        r[y == l] = cmap[l, 0]
        g[y == l] = cmap[l, 1]
        b[y == l] = cmap[l, 2]
    rgb = np.zeros((y.shape[0], y.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0

    rgb = cv2.resize(rgb, (original_width, original_height))

    rgb = (rgb * 255).astype(np.uint8)
    img = rgb #np.squeeze(rgb)
    return img


def main():
    args = get_arguments()
    print(args)

    key_image_raw_contents = tf.placeholder(tf.float32, [original_height, original_width, 3])
    current_image_raw_contents = key_image_raw_contents
    image_s, image_f, resized_image = scale_and_mask(key_image_raw_contents, current_image_raw_contents, input_size)

    image_s = tf.squeeze(image_s)
    image_f = tf.squeeze(image_f)
    image_s = tf.expand_dims(image_s, 0)
    image_f = tf.expand_dims(image_f, 0)
    current_frame = image_f
    image_current_frame_in = tf.placeholder(tf.float32, [seg_input_height, seg_input_width, 3])
    image_in_batch = tf.expand_dims(image_current_frame_in, 0)

    # Segmentation path.
    current_output_ph = tf.placeholder(tf.float32, [seg_output_height, seg_output_width, NUM_CLASSES + 1])

    key_image = tf.placeholder(tf.float32, [1, seg_input_height // 2, seg_input_width // 2, 3])
    key_frame = key_image
    flowNet = FlowNets(current_frame, key_frame)
    decisionNet = Decision(feature_size=decision_feature_size)

    raw_pred = tf.expand_dims(current_output_ph, dim=0)
    raw_pred = mask_channels(raw_pred, camvid_labels)

    seg_pred = raw_pred[0]

    # Estimation Flow and feature for decision network.
    flows = flowNet.inference()
    flow_feature = tf.image.resize_bilinear(flows['feature'], decision_feature_size)

    # Spatial warping path

    key_pred = tf.placeholder(tf.float32, [1, seg_output_height, seg_output_width, NUM_CLASSES])
    flow_field = tf.placeholder(tf.float32, [1, seg_input_height // 8, seg_input_width // 8, 2])
    scale_field = tf.placeholder(tf.float32, [1, seg_input_height // 8, seg_input_width // 8, NUM_CLASSES + 1 + 7])

    scale_field_masked = mask_channels(scale_field, cityscale_camvid_labels)
    raw_pred_resized = tf.image.resize_bilinear(key_pred, flow_field.get_shape()[1:3])
    warp_pred = warp(raw_pred_resized, flow_field)

    scale_pred = tf.multiply(warp_pred, scale_field_masked)
    wrap_output = tf.image.resize_bilinear(scale_pred, [seg_output_height, seg_output_width])
    wrap_output = wrap_output[0]

    # checkpoint variables
    variables_flownet = [var for var in tf.global_variables() if var.name.startswith('FlowNets')]
    variables_decision = [var for var in tf.global_variables() if var.name.startswith('decision')]

    # Set up tf session and initialize variables.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    load_checkpoints(args, sess, variables_flownet, variables_decision)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Register
    target_score = args.target
    key_outputs = None
    current_output = None
    seg_step = 0
    flow_step = 0
    avg_fps = 0
    cap = cv2.VideoCapture(args.video_file)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    outfilename = args.save_dir+'output-video-' + ('original.avi' if args.process_original == 'True' else 'segmented.avi')
    frame_index = (int(target_score) // len(frames_table)) - 1
    frame_index = frame_index if frame_index > 0 else 0
    FRAMES = frames_table[frame_index]
    print("output frames picked: ", FRAMES)
    out = cv2.VideoWriter(outfilename, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), FRAMES, (frame_width, frame_height))
    step = 0
    while (cap.isOpened()):
        ret, frame = cap.read()

        if step % 100 == 0:
            print('.', end='')
        if step % 1000 == 0:
            print('')
        start_time = time.time()
        current_frame_raw_image = get_image_array(frame, seg_input_width, seg_input_height)
        if step == 0:
            image_inputs, key_inputs, segmentation_input = sess.run([image_s, image_f, image_in_batch], feed_dict={
                image_current_frame_in: current_frame_raw_image,
                key_image_raw_contents: frame
            })
            # print("Initial region {}".format(i))
            segment_output_tensor = sess.graph.get_tensor_by_name('import/activation_49/truediv:0')
            segment_input_tensor = sess.graph.get_tensor_by_name('import/input_1:0')

            segment_output = sess.run(segment_output_tensor, {segment_input_tensor: segmentation_input})

            current_seg = segment_output.reshape((seg_output_height, seg_output_width, NUM_CLASSES + 1))
            key_outputs, pred = sess.run([raw_pred, seg_pred],
                                         feed_dict={current_output_ph: current_seg})
            pred = pred.argmax(axis=2)
            current_output = pred
            step = step + 1
        else:
            image_input, key_tmp, flow_features, flow_fields, scale_fields, segmentation_input = sess.run(
                [image_s, image_f, flow_feature, flows['flow'], flows['scale'], image_in_batch],
                feed_dict={
                    key_image: image_inputs,
                    image_current_frame_in: current_frame_raw_image,
                    key_image_raw_contents: frame
                })
            pred_scores = np.squeeze(decisionNet.pred(sess, flow_features))
            # print(
                # "step {} region {} predict score: {:.3}  target: {:.3}".format(step, i, pred_scores, targets[i]))
            if pred_scores < target_score:
                seg_step += 1
                # print("Segmentation Path")
                image_inputs = key_tmp
                segment_output_tensor = sess.graph.get_tensor_by_name('import/activation_49/truediv:0')
                segment_input_tensor = sess.graph.get_tensor_by_name('import/input_1:0')

                segment_output = sess.run(segment_output_tensor, {segment_input_tensor: segmentation_input})

                current_seg = segment_output.reshape((seg_output_height, seg_output_width, NUM_CLASSES + 1))
                key_outputs, pred = sess.run([raw_pred, seg_pred],
                                             feed_dict={current_output_ph: current_seg})
                pred = pred.argmax(axis=2)
                current_output = pred
            else:
                flow_step += 1
                # print("Spatial Warping Path")
                output_temp = sess.run([wrap_output],
                                feed_dict={flow_field: flow_fields,
                                           scale_field: scale_fields,
                                           key_pred: key_outputs})
                output_temp = output_temp[0]
                output_temp = output_temp.argmax(axis=2)
                current_output = output_temp
            step = step + 1

        # measure time
        total_time = time.time() - start_time
        fps = 1 / total_time
        avg_fps += fps
        # print("fps: {:.3}".format(1 / total_time))

        out_img = vis_preparevideo(current_output, cmap)
        out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
        out_img = add_rectangles_with_labels(out_img, frame, enable_original=args.process_original)
        out.write(out_img)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()