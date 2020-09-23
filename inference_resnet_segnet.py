import argparse
import os
import time
import cv2
import six
import tensorflow as tf
import numpy as np
from tensorflow.python.platform import gfile
from model.flownetmodel import FlowNets
from model.decisionmodel import Decision
from tools.flow_utils import warp


from tools.image_reader import read_image_from_filename, scale_and_mask

class DataLoaderError(Exception):
    pass

from pathlib import Path
home = str(Path.home())
# DATA_DIRECTORY = home + '/data/video-segmentation/camvid_30_fps/'
DATA_DIRECTORY = './camvid_30_fps_test_only/'
# DATA_LIST_PATH = './list/sample_video.txt'
DATA_LIST_PATH = './list/test_file_list.txt'

SEGNET_CHKPT = './resnet50_segnet_model/resnet50_segnet.pb'
DVS_FLOWNET_CHKPT = './dvs_net_flownets_checkpoints/finetune/'
DECISION_CHKPT = './decision_network_checkpoints/'


SAVE_DIR = './inference_output/'
NUM_CLASSES = 11
NUM_STEPS = 6959 # Number of images in the test set
TARGET = 80.0

seg_input_width = 608
seg_input_height = 416
seg_output_width = 304
seg_output_height = 208
input_size = [seg_input_height, seg_input_width]
original_width = 480
original_height = 360
original_size = [original_height, original_width]
camvid_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
cityscale_camvid_labels = [0, 1, 2, 4, 5, 6, 8, 10, 11, 13, 18]

decision_feature_size = [4, 5]

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

def get_segmentation_array(image_input, nClasses,
                           width, height, no_reshape=False, map_labels=True):
    """ Load segmentation array from input """

    seg_labels = np.zeros((height, width, nClasses))

    if type(image_input) is np.ndarray:
        # It is already an array, use it as it is
        img = image_input
    elif isinstance(image_input, six.string_types):
        if not os.path.isfile(image_input):
            raise DataLoaderError("get_segmentation_array: "
                                  "path {0} doesn't exist".format(image_input))
        img = cv2.imread(image_input, 1)
    else:
        raise DataLoaderError("get_segmentation_array: "
                              "Can't process input type {0}"
                              .format(str(type(image_input))))

    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)
    img = img[:, :, 0]

    for c in range(nClasses):
        seg_labels[:, :, c] = (img == c).astype(int)

    if not no_reshape:
        seg_labels = np.reshape(seg_labels, (width*height, nClasses))

    return seg_labels



def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Dynamic Video Segmentation Network")
    parser.add_argument("--data_dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the dataset.")
    parser.add_argument("--data_list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--segnet_chkpt", type=str, default=SEGNET_CHKPT,
                        help="Where restore segnet model parameters from.")
    parser.add_argument("--dvs_flownet_chkpt", type=str, default=DVS_FLOWNET_CHKPT,
                        help="Where restore dvs flownet model parameters from.")
    parser.add_argument("--decision_chkpt", type=str, default=DECISION_CHKPT,
                        help="Where restore decision model parameters from.")
    parser.add_argument("--save_dir", type=str, default=SAVE_DIR,
                        help="Where to save segmented output.")
    parser.add_argument("--num_steps", type=int, default=NUM_STEPS,
                        help="Number of images in the video.")
    parser.add_argument("--target", type=float, default=TARGET,
                        help="Confidence score threshold.")
    return parser.parse_args()


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

class_name = np.array([
    'sky', 'building', 'pole', 'road', 'pavement', 'tree', 'signsymbol',
    'fence', 'car', 'pedestrian', 'bicyclist', 'unlabelled'])

cmap = np.array([
    [128, 128, 128],
    [128, 0, 0],
    [192, 192, 128],
    [128, 64, 128],
    [60, 40, 222],
    [128, 128, 0],
    [192, 128, 128],
    [64, 64, 128],
    [64, 0, 128],
    [64, 64, 0],
    [0, 128, 192],
    [0, 0, 0]])

from PIL import Image
def save_img(img, dir_path, filename):
    img = np.squeeze(img) #.astype(np.uint8)
    img = Image.fromarray(img)
    _res_path = os.path.join(dir_path, filename)
    img.save(_res_path)
    return _res_path

def vis_semseg(y, cmap, inp_img, dir_path=None, filename=None):
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

    if inp_img is not None:
        orininal_h = original_height #inp_img.shape[0]
        orininal_w = original_width #inp_img.shape[1]
        rgb = cv2.resize(rgb, (orininal_w, orininal_h))

    if dir_path is None and filename is None:
        rgb = (rgb * 255).astype(np.uint8)
        return rgb
    else:
        rgb = (rgb * 255).astype(np.uint8)
        return save_img(rgb, dir_path, filename)


tp = np.zeros(NUM_CLASSES)
fp = np.zeros(NUM_CLASSES)
fn = np.zeros(NUM_CLASSES)
n_pixels = np.zeros(NUM_CLASSES)

def load_eval_metrics(pr, gt):
    gt = gt.argmax(-1)
    pr = pr.flatten()
    gt = gt.flatten()

    for cl_i in range(NUM_CLASSES):
        tp[cl_i] += np.sum((pr == cl_i) * (gt == cl_i))
        fp[cl_i] += np.sum((pr == cl_i) * ((gt != cl_i)))
        fn[cl_i] += np.sum((pr != cl_i) * ((gt == cl_i)))
        n_pixels[cl_i] += np.sum(gt == cl_i)

def compute_eval_metrics():
    cl_wise_score = tp / (tp + fp + fn + 0.000000000001)
    n_pixels_norm = n_pixels / np.sum(n_pixels)
    frequency_weighted_IU = np.sum(cl_wise_score * n_pixels_norm)
    mean_IU = np.mean(cl_wise_score)

    return {
        "frequency_weighted_IU": frequency_weighted_IU,
        "mean_IU": mean_IU,
        "class_wise_IU": cl_wise_score
    }


def main():
    """Create the model and start the evaluation process."""
    args = get_arguments()
    print(args)
    tf.reset_default_graph()

    key_image_raw, current_image_raw, raw_label = read_image_from_filename(args.data_dir, args.data_list, 1, input_size)
    image_s, image_f, resized_image, mask = scale_and_mask(key_image_raw, current_image_raw, raw_label, input_size)

    image_s = tf.squeeze(image_s)
    image_f = tf.squeeze(image_f)
    image_s = tf.expand_dims(image_s, 0)
    image_f = tf.expand_dims(image_f, 0)
    current_frame = image_f
    image_current_frame_in = tf.placeholder(tf.float32, [seg_input_height, seg_input_width, 3])
    image_in_batch = tf.expand_dims(image_current_frame_in, 0)

    # Segmentation path.
    current_output_ph = tf.placeholder(tf.float32, [seg_output_height, seg_output_width, NUM_CLASSES + 1])
    ground_truth_ph = tf.placeholder(tf.float32, [seg_output_height, seg_output_width, NUM_CLASSES + 1])


    key_image = tf.placeholder(tf.float32, [1, seg_input_height // 2, seg_input_width // 2, 3])
    key_frame = key_image
    flowNet = FlowNets(current_frame, key_frame)
    decisionNet = Decision(feature_size=decision_feature_size)

    raw_pred = tf.expand_dims(current_output_ph, dim=0)
    raw_pred = mask_channels(raw_pred, camvid_labels)

    raw_gt = tf.expand_dims(ground_truth_ph, dim=0)
    # raw_gt = mask_channels(raw_gt, camvid_labels) # Keep the original

    seg_pred = raw_pred[0]
    seg_gt = raw_gt[0]

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

    # Start queue threads.
    threads = tf.train.start_queue_runners(sess=sess)

    if not os.path.exists(args.save_dir) and args.save_dir != 'none':
        os.makedirs(args.save_dir)

    # Register
    target_score = args.target
    key_outputs = None
    current_output = None
    current_gt = None
    seg_step = 0
    flow_step = 0
    # Read input files
    list_file = open(args.data_list, 'r')
    avg_fps = 0

    for step in range(args.num_steps):
        f1, f2, ann = list_file.readline().split('\n')[0].split(' ')
        f2 = os.path.join(args.data_dir, f2)
        ann_file = os.path.join(args.data_dir, ann)

        current_frame_raw_image = get_image_array(f2, seg_input_width, seg_input_height)

        ann_img = get_segmentation_array(image_input= ann_file,
                                         nClasses=NUM_CLASSES + 1,
                                         width=seg_output_width,
                                         height=seg_output_height,
                                         no_reshape=True,
                                         map_labels=True)

        if step % 100 == 0:
            print('.', end='')
        if step % 1000 == 0:
            print('')
        start_time = time.time()
        if step == 0:
            image_inputs, key_inputs, segmentation_input = sess.run([image_s, image_f, image_in_batch], feed_dict={
                image_current_frame_in : current_frame_raw_image
            })
            segment_output_tensor = sess.graph.get_tensor_by_name('import/activation_49/truediv:0')
            segment_input_tensor = sess.graph.get_tensor_by_name('import/input_1:0')

            segment_output = sess.run(segment_output_tensor, {segment_input_tensor: segmentation_input})

            current_seg = segment_output.reshape((seg_output_height, seg_output_width, NUM_CLASSES + 1))
            key_outputs, pred, seg_img = sess.run([raw_pred, seg_pred, seg_gt],
                            feed_dict={
                                current_output_ph:current_seg,
                                ground_truth_ph: ann_img
                            })
            current_gt = seg_img
            pred = pred.argmax(axis=2)
            current_output = pred
        else:
            image_input, key_tmp, flow_features, flow_fields, scale_fields, segmentation_input = sess.run(
                [image_s, image_f, flow_feature, flows['flow'], flows['scale'], image_in_batch],
                feed_dict={
                    key_image: image_inputs,
                    image_current_frame_in: current_frame_raw_image,
                })
            pred_scores = np.squeeze(decisionNet.pred(sess, flow_features))

            if pred_scores < target_score:
                seg_step += 1
                # print("Segmentation Path")
                image_inputs = key_tmp
                segment_output_tensor = sess.graph.get_tensor_by_name('import/activation_49/truediv:0')
                segment_input_tensor = sess.graph.get_tensor_by_name('import/input_1:0')

                segment_output = sess.run(segment_output_tensor, {segment_input_tensor: segmentation_input})

                current_seg = segment_output.reshape((seg_output_height, seg_output_width, NUM_CLASSES + 1))
                key_outputs, pred, seg_img = sess.run([raw_pred, seg_pred, seg_gt],
                                             feed_dict={
                                                 current_output_ph: current_seg,
                                                 ground_truth_ph: ann_img
                                             })
                current_gt = seg_img
                pred = pred.argmax(axis=2)
                current_output = pred

            else:
                flow_step += 1
                # print("Spatial Warping Path")
                output_temp, seg_img = sess.run([wrap_output, seg_gt],
                                feed_dict={
                                    flow_field: flow_fields,
                                    scale_field: scale_fields,
                                    key_pred: key_outputs,
                                    ground_truth_ph: ann_img
                                           })
                current_gt = seg_img
                output_temp = output_temp.argmax(axis=2)
                current_output = output_temp


        # measure time
        total_time = time.time() - start_time
        fps = 1/ total_time
        avg_fps += fps
        # print("fps: {:.3}".format(1 / total_time))
        load_eval_metrics(pr=current_output, gt=current_gt)

        if args.save_dir != 'none':
            vis_semseg(current_output, cmap, current_frame_raw_image, args.save_dir, 'mask' + str(step) + '.png')

    print('\nAverage fps: ', avg_fps/args.num_steps)
    print('\nFinish!')
    print("segmentation steps:", seg_step, "flow steps:", flow_step)
    print("evaluation metrics:", compute_eval_metrics())

if __name__ == '__main__':
    main()