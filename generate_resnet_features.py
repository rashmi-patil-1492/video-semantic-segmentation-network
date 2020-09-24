import argparse
import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
import cv2
import six
import os

from model.flownetmodel import FlowNets
from tools.flow_utils import warp

class DataLoaderError(Exception):
    pass

# Read the following from resnet...config.json file
seg_input_width = 608
seg_input_height = 416
seg_output_width = 304
seg_output_height = 208
input_size = [seg_input_height, seg_input_width]
output_size = [seg_output_height, seg_output_width]


from pathlib import Path
home = str(Path.home())
# DATA_DIRECTORY = home + '/data/video-segmentation/camvid_30_fps/'
DATA_DIRECTORY = './camvid_30_fps/'

DATA_LIST_PATH = 'list/train_file_list.txt'
RESTORE_FROM_SEG = './resnet50_segnet_model/resnet50_segnet.pb'
RESTORE_FROM_FLOWNET = './dvs_net_flownets_checkpoints/finetune/'
SAVE_DIR = './generated_features_resnet_segnet/train/'

NUM_CLASSES = 11
NUM_STEPS = 11005  # Number of images in the dataset.

camvid_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
cityscale_camvid_labels = [0, 1, 2, 4, 5, 6, 8, 10, 11, 13, 18]

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

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Generate Testcases")
    parser.add_argument("--data_dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the dataset.")
    parser.add_argument("--data_list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--restore_from_seg", type=str, default=RESTORE_FROM_SEG,
                        help="Where restore segmentation model parameters from.")
    parser.add_argument("--restore_from_flownet", type=str, default=RESTORE_FROM_FLOWNET,
                        help="Where restore flownet model parameters from.")
    parser.add_argument("--save_dir", type=str, default=SAVE_DIR,
                        help="Where to save segmented output.")
    parser.add_argument("--num_steps", type=int, default=NUM_STEPS,
                        help="Number of images in the video.")
    parser.add_argument("--clip", type=float, default=0.0,
                        help="trim extreme confidence score")
    return parser.parse_args()


def mask_channels(tensor, mask_indexes):
    short_list_tensors = []
    # mask_indexes contains indexes of output labels to consider
    # For camvid it would be 12
    for index in mask_indexes:
        short_list_tensors.append(tf.expand_dims(tensor[:,:,:,index], -1))
    return tf.concat(short_list_tensors, -1)


def to_bgr(image):
    img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=image)
    image_bgr = tf.concat(axis=2, values=[img_b, img_g, img_r])
    return image_bgr


def scale_and_resize(image1, image2, input_sizes):
    image1 = to_bgr(image1)
    image2 = to_bgr(image2)
    image1 = tf.cast(image1, tf.float32) / 255.
    image2 = tf.cast(image2, tf.float32) / 255.

    image1 = tf.expand_dims(image1, 0)
    image2 = tf.expand_dims(image2, 0)

    image_shape = tf.shape(image2)
    image_input_shape = tf.to_float(image_shape[1:3])

    scale_shape = input_sizes / image_input_shape
    scale = tf.reduce_min(scale_shape)
    scaled_input_shape = tf.to_int32(tf.round(image_input_shape * scale))

    resized_image1 = tf.image.resize_nearest_neighbor(
        image1, scaled_input_shape)
    resized_image2 = tf.image.resize_nearest_neighbor(
        image2, scaled_input_shape)

    cropped_image1 = tf.image.resize_image_with_crop_or_pad(
        resized_image1, input_sizes[0] // 2, input_sizes[1] // 2)

    cropped_image2 = tf.image.resize_image_with_crop_or_pad(
        resized_image2, input_sizes[0] // 2, input_sizes[1] // 2)

    return cropped_image1, cropped_image2


def main():
    args = get_arguments()
    print(args)
    tf.reset_default_graph()

    # Read images and do other calculations

    # Set placeholder
    image1_filename = tf.placeholder(dtype=tf.string, name='filename1')
    image2_filename = tf.placeholder(dtype=tf.string, name='filename2')
    current_output_ph = tf.placeholder(tf.float32, [seg_output_height, seg_output_width, NUM_CLASSES + 1])

    # Read & Decode image
    image1 = tf.image.decode_image(tf.read_file(image1_filename), channels=3)
    image2 = tf.image.decode_image(tf.read_file(image2_filename), channels=3)

    image1, image2 = scale_and_resize(image1, image2, input_size)
    key_frame = image1
    current_frame = image2

    image_in = tf.placeholder(tf.float32, [seg_input_height, seg_input_width, 3])
    image_batch = tf.expand_dims(image_in, 0)

    # Get the image from session
    key_frame_input = tf.placeholder(shape=[seg_input_height // 2, seg_input_width // 2, 3], dtype=tf.float32, name='keyframe_input')
    key_frame_input_dim = tf.expand_dims(key_frame_input, dim=0)
    current_frame_input = tf.placeholder(shape=[seg_input_height // 2, seg_input_width // 2, 3], dtype=tf.float32, name='currentframe_input')
    current_frame_input_dim = tf.expand_dims(current_frame_input, dim=0)
    flowNet = FlowNets(current_frame_input_dim, key_frame_input_dim)

    flows = flowNet.inference()

    raw_pred = tf.expand_dims(current_output_ph, dim=0)
    raw_pred = mask_channels(raw_pred, camvid_labels)

    flows_scale = mask_channels(flows['scale'], cityscale_camvid_labels)
    flow_resized = tf.image.resize_bilinear(raw_pred, flows['flow'].get_shape()[1:3])
    warp_pred = warp(flow_resized, flows['flow'])
    scale_pred = tf.multiply(warp_pred, flows_scale)
    wrap_output = tf.image.resize_bilinear(scale_pred, output_size)
    output = tf.argmax(wrap_output, axis=3)

    current_output = tf.reshape(raw_pred, shape=[1, seg_output_height, seg_output_width, NUM_CLASSES])
    current_output = tf.argmax(current_output, axis=3)
    # Calculate confidence score.
    weight = tf.where(tf.equal(current_output, 255), tf.zeros_like(current_output), tf.ones_like(current_output))
    accuracy = tf.where(tf.equal(output, current_output), weight, tf.zeros_like(current_output))
    average = tf.divide(tf.reduce_sum(tf.contrib.layers.flatten(accuracy), 1),
                        tf.reduce_sum(tf.contrib.layers.flatten(weight), 1))

    variables_flownet = [var for var in tf.global_variables() if var.name.startswith('FlowNets')]
    # Set up tf session and initialize variables.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # Load segmentation model
    graph_def = tf.GraphDef()
    with gfile.FastGFile(args.restore_from_seg, 'rb') as f:
        graph_def.ParseFromString(f.read())

    tf.import_graph_def(graph_def)

    # Load flownet model
    ckpt = tf.train.get_checkpoint_state(args.restore_from_flownet)
    if ckpt and ckpt.model_checkpoint_path:
        loader = tf.train.Saver(var_list=variables_flownet)
        loader.restore(sess, ckpt.model_checkpoint_path)

    # Read input files
    list_file = open(args.data_list, 'r')
    score_list = []
    ft_list = []
    regions = 1
    for step in range(args.num_steps):
        f1, f2, f3 = list_file.readline().split('\n')[0].split(' ')
        f1 = os.path.join(args.data_dir, f1)
        f2 = os.path.join(args.data_dir, f2)
        image_2_raw = get_image_array(f2, seg_input_width, seg_input_height)

        segment_img_batches, current_img_batches, key_frame_img_batches = sess.run(
            [image_batch, current_frame, key_frame],
            feed_dict=
            {
                image1_filename: f1,
                image2_filename: f2,
                image_in: image_2_raw
            })

        for i in range(regions):
            segment_output_tensor = sess.graph.get_tensor_by_name('import/activation_49/truediv:0')
            segment_input_tensor = sess.graph.get_tensor_by_name('import/input_1:0')

            segment_output = sess.run(segment_output_tensor, {segment_input_tensor: segment_img_batches})
            segment_output = segment_output[0]
            current_seg = segment_output.reshape((seg_output_height, seg_output_width, NUM_CLASSES + 1))
            flow_feature, score, acc, w, curr_out = sess.run([flows['feature'], average, accuracy, weight, current_output],
                                           feed_dict={key_frame_input: key_frame_img_batches[i],
                                                      current_frame_input: current_img_batches[i],
                                                      current_output_ph: current_seg})
            if score > args.clip:
                ft_list.append(flow_feature)
                score_list.append(score * 100)
        if step % 100 == 0:
            print(step, 'finished ...', score_list[len(score_list) - 1])
    # save confidence score and feature
    np.save(args.save_dir + "X", ft_list)
    np.save(args.save_dir + "Y", score_list)
    print("Generate finish!")

if __name__ == '__main__':
    main()


