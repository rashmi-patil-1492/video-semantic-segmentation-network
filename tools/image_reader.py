import tensorflow as tf
import numpy as np

IMG_MEAN = np.array((103.939, 116.779, 123.68), dtype=np.float32)


def read_image_label_list(data_dir, data_list):
    """Reads txt file containing paths to images and ground truth masks.

        Args:
          data_dir: path to the directory with images and masks.
          data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.

        Returns:
          Two lists with all file names for images and masks, respectively.
        """
    f = open(data_list, 'r')
    key_images = []
    current_images = []
    labels = []
    for line in f:
        try:
            image_line = line[:-1].split('\n')[0]
        except ValueError:  # Adhoc for test.
            image_line = line.strip("\n")
        if image_line == '':
            continue
        if len(image_line.split(' ')) == 3:
            key_image_path, current_image_path, label_path = image_line.split(' ')
            key_image = data_dir + key_image_path
            current_image = data_dir + current_image_path
            label = data_dir + label_path
            if not tf.gfile.Exists(key_image):
                raise ValueError('Failed to find file: ' + key_image)
            if not tf.gfile.Exists(label):
                raise ValueError('Failed to find file: ' + label)

            key_images.append(key_image)
            current_images.append(current_image)
            labels.append(label)
        else:
            key_image_path = image_line.split(' ')
            key_image = data_dir + key_image_path
            if not tf.gfile.Exists(key_image):
                raise ValueError('Failed to find file: ' + key_image)
            key_images.append(key_image)

    f.close()
    return key_images, current_images, labels


def read_labeled_image_list(data_dir, data_list):
    """Reads txt file containing paths to images and ground truth masks.
    
    Args:
      data_dir: path to the directory with images and masks.
      data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.
       
    Returns:
      Two lists with all file names for images and masks, respectively.
    """
    f = open(data_list, 'r')
    images = []
    for line in f:
        try:
            image = line[:-1].split('\n')[0]
        except ValueError: # Adhoc for test.
            image = line.strip("\n")

        image = data_dir+image
        if not tf.gfile.Exists(image):
            raise ValueError('Failed to find file: ' + image)

        images.append(image)
    f.close()
    return images

def resizer(raw_image, input_size):
    return tf.image.resize_image_with_crop_or_pad(raw_image, input_size[0], input_size[1])


def read_images_from_disk(input_queue, input_size, overlap, img_mean=IMG_MEAN):
    """Consumes a single filename and label as a ' '-delimited string.
    Args:
        filename_tensor: A scalar string tensor.
    Returns:
        Three tensors: the decoded images and flos.
    """
    height = input_size[0]//2
    height_overlap = height+overlap
    width = input_size[1]//2
    width_overlap = width+overlap

    image_file = tf.read_file(input_queue[0])
    image = tf.image.decode_image(image_file)

    image = tf.cast(image,tf.float32)

    img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=image)
    image_bgr = tf.concat(axis=2, values=[img_b, img_g, img_r])
    image_bgr.set_shape((None, None, 3))
    image_bgr = tf.expand_dims(tf.image.resize_images(image_bgr, input_size), 0)
    print(' before spliting ', image_bgr.shape)
    images = tf.concat([image_bgr[:, :height+overlap, :width+overlap, :],
                    image_bgr[:, :height+overlap, width-overlap:, :],
                    image_bgr[:, height-overlap:, :width+overlap, :],
                    image_bgr[:, height-overlap:, width-overlap:, :]],0)

    print(' after spliting ', images.shape)

    # Preprocess.
    image_s = images-img_mean
    image_f = tf.image.resize_images(images/255.0, [(height_overlap)//2, (width_overlap)//2])

    return image_s, image_f


def to_bgr(image):
    img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=image)
    image_bgr = tf.concat(axis=2, values=[img_b, img_g, img_r])
    return image_bgr

def crop_and_upsample(prob, resized_image, raw_image, mask, num_classes):
    resized_h = tf.shape(resized_image)[1]
    resized_w = tf.shape(resized_image)[2]
    resized_shape = tf.stack([1, resized_h, resized_w, num_classes ])
    raw_shape = tf.shape(raw_image)[:2]
    cropped_prob = tf.boolean_mask(
        tf.squeeze(prob), tf.squeeze(tf.equal(mask, 0)))
    reshaped_prob = tf.reshape(cropped_prob, resized_shape)
    upsampled_prob = tf.image.resize_bilinear(reshaped_prob, raw_shape)
    return tf.squeeze(tf.cast(tf.argmax(upsampled_prob, axis=-1), tf.int32))


def read_image_from_filename(data_dir, data_list, batch_size, input_size_to_rescale):
    key_image_list, current_image_list, label_list = read_image_label_list(data_dir, data_list)
    key_image_tensor = tf.convert_to_tensor(key_image_list, dtype=tf.string)
    current_image_tensor = tf.convert_to_tensor(current_image_list, dtype=tf.string)
    label_tensor = tf.convert_to_tensor(label_list, dtype=tf.string)
    queue = tf.train.slice_input_producer(
        [key_image_tensor, current_image_tensor, label_tensor], shuffle=False)

    key_image_contents = tf.read_file(queue[0])
    current_image_contents = tf.read_file(queue[1])
    label_contents = tf.read_file(queue[2])

    key_images = tf.image.decode_png(key_image_contents, channels=3)
    current_images = tf.image.decode_png(current_image_contents, channels=3)
    labels = tf.image.decode_png(label_contents, channels=1)
    return key_images, current_images, labels

def scale_and_mask(key_image, current_image, labels, input_size_to_rescale):

    cropped_key_image, cropped_current_image, resized_image, mask = scale_fixed_size(key_image, current_image, labels, input_size_to_rescale)

    return cropped_key_image, cropped_current_image, resized_image, mask
    # return _generate_image_and_label_batch_with_mask(cropped_image, cropped_f_image, mask, batch_size)


def read_segment_flownet_images(input_queue, input_size, overlap):
    height = input_size[0]
    width = input_size[1]

    image_file = tf.read_file(input_queue[0])
    image = tf.image.decode_image(image_file)
    image = resizer(image, [height, width])
    image_s = image
    image_f = to_bgr(image)

    image_s = tf.cast(image_s, tf.float32)
    image_f = tf.cast(image_f, tf.float32)

    image_s.set_shape([None, None, 3])
    image_f.set_shape([None, None, 3])

    height = height + overlap
    width = width + overlap

    image_s = tf.image.resize_images((image_s) / 255.0, (height // 1, width // 1))
    image_f = tf.image.resize_images((image_f) / 255.0, (height // 2, width // 2))
    return image_s, image_f


def _generate_image_and_label_batch_with_mask(image_s, image_f, mask,  batch_size):
    """Construct a queued batch of images and labels.

    Args:
        image_s, image_f: 3-D Tensor of input image of type.float32.
        batch_size: Number of images per batch.

    Returns:
        bimages: Images. 4D tensor of [batch_size, height, width, 3] size.
        bflo: Flos. 4D tensor of [batch_size, height, width, 2] size.
    """
    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.
    num_preprocess_threads = 4
    bimage_s, bimage_f, bi_mask = tf.train.batch(
            [image_s, image_f, mask],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=1)

    return bimage_s, bimage_f, bi_mask


def _generate_image_and_label_batch(image_s, image_f, batch_size):
    """Construct a queued batch of images and labels.

    Args:
        image_s, image_f: 3-D Tensor of input image of type.float32.
        batch_size: Number of images per batch.

    Returns:
        bimages: Images. 4D tensor of [batch_size, height, width, 3] size.
        bflo: Flos. 4D tensor of [batch_size, height, width, 2] size.
    """
    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.
    num_preprocess_threads = 4
    bimage_s, bimage_f = tf.train.batch(
            [image_s, image_f],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=1)

    return bimage_s, bimage_f


def scale_fixed_size(key_image, current_image, raw_label, output_shape, ignore_label=255):

    current_f = to_bgr(current_image)
    key_image = tf.cast(key_image, tf.float32) / 255.
    current_f = tf.cast(current_f, tf.float32) / 255.

    raw_label = tf.cast(raw_label, tf.int32)
    raw_height = tf.shape(key_image)[0]
    raw_width = tf.shape(key_image)[1]

    image_batch = tf.expand_dims(key_image, 0)
    current_f_batch = tf.expand_dims(current_f, 0)
    label_batch = tf.expand_dims(raw_label, 0)

    raw_label_size = tf.shape(image_batch)
    raw_image_size = tf.shape(label_batch)
    image_f_size = tf.shape(current_f_batch)

    input_shape = tf.to_float(raw_image_size[1:3])

    scale_shape = output_shape / input_shape
    scale = tf.reduce_min(scale_shape)
    scaled_input_shape = tf.to_int32(tf.round(input_shape * scale))

    resized_image = tf.image.resize_nearest_neighbor(
        image_batch, scaled_input_shape)
    resized_current_f_image = tf.image.resize_nearest_neighbor(
        current_f_batch, scaled_input_shape)
    resized_label = tf.image.resize_nearest_neighbor(
        label_batch, scaled_input_shape)

    shifted_classes = resized_label + 1

    cropped_key_image = tf.image.resize_image_with_crop_or_pad(
        resized_image, output_shape[0] // 2, output_shape[1] // 2)

    cropped_current_f_image = tf.image.resize_image_with_crop_or_pad(
        resized_current_f_image, output_shape[0] // 2, output_shape[1] // 2)

    cropped_label = tf.image.resize_image_with_crop_or_pad(
        shifted_classes, output_shape[0], output_shape[1])

    mask = tf.to_int32(tf.equal(cropped_label, 0)) * (ignore_label + 1)
    cropped_label = cropped_label + mask - 1

    return cropped_key_image, cropped_current_f_image, resized_image, mask

def input_images(data_dir, data_list, batch_size, input_size, overlap):
    image_list = read_labeled_image_list(data_dir, data_list)

    images = tf.convert_to_tensor(image_list, dtype=tf.string)

    input_queue = tf.train.slice_input_producer([images], shuffle=False)

    image_s, image_f = read_segment_flownet_images(input_queue=input_queue, input_size=input_size, overlap=overlap)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(image_s, image_f, batch_size)


def inputs(data_dir, data_list, batch_size, input_size, overlap, img_mean=IMG_MEAN):
    """Construct input for CIFAR evaluation using the Reader ops.

    Args:
        data_dir: Path to the FlowNet data directory.
        batch_size: Number of images per batch.

    Returns:
        image1, image2: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    """
    image_list = read_labeled_image_list(data_dir, data_list)

    images = tf.convert_to_tensor(image_list, dtype=tf.string)

    input_queue = tf.train.slice_input_producer([images], shuffle=False)
    image_s, image_f = read_images_from_disk(input_queue, input_size, overlap, img_mean)
    
    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(image_s, image_f, batch_size)
