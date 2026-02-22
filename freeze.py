from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import argparse
import os
import json

graph_util = tf.compat.v1.graph_util

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_dir", default="model/export", help="directory containing the exported checkpoint and options.json")
parser.add_argument("--output_pb", default="model/export/frozen_model.pb", help="path for the output frozen .pb file")
a = parser.parse_args()

CROP_SIZE = 256


def preprocess(image):
    with tf.name_scope("preprocess"):
        return image * 2 - 1


def deprocess(image):
    with tf.name_scope("deprocess"):
        return (image + 1) / 2


def lrelu(x, leak):
    with tf.name_scope("lrelu"):
        x = tf.identity(x)
        return (0.5 * (1 + leak)) * x + (0.5 * (1 - leak)) * tf.abs(x)


def batchnorm(inputs):
    # Matches tf.layers.batch_normalization(axis=3, epsilon=1e-5, momentum=0.1,
    #   training=True, gamma_initializer=random_normal(1.0, 0.02))
    # Variable names must match the checkpoint: batch_normalization/{gamma,beta,moving_mean,moving_variance}
    with tf.variable_scope("batch_normalization"):
        n_out = inputs.get_shape().as_list()[-1]
        gamma = tf.get_variable("gamma", [n_out],
                                initializer=tf.random_normal_initializer(1.0, 0.02))
        beta = tf.get_variable("beta", [n_out],
                               initializer=tf.zeros_initializer())
        # moving_mean / moving_variance exist in the checkpoint (created by tf.layers
        # even with training=True) but are not used in the forward pass.
        tf.get_variable("moving_mean", [n_out],
                        initializer=tf.zeros_initializer(), trainable=False)
        tf.get_variable("moving_variance", [n_out],
                        initializer=tf.ones_initializer(), trainable=False)
        # training=True: normalise with batch statistics
        mean, variance = tf.nn.moments(inputs, axes=[0, 1, 2])
        return tf.nn.batch_normalization(inputs, mean, variance, beta, gamma,
                                         variance_epsilon=1e-5)


def gen_conv(batch_input, out_channels, separable_conv):
    # Matches tf.layers.conv2d / tf.layers.separable_conv2d
    # Variable scope: conv2d/{kernel,bias}  or  separable_conv2d/{depthwise_kernel,pointwise_kernel,bias}
    initializer = tf.random_normal_initializer(0, 0.02)
    if separable_conv:
        with tf.variable_scope("separable_conv2d"):
            in_ch = batch_input.get_shape().as_list()[-1]
            dw = tf.get_variable("depthwise_kernel", [4, 4, in_ch, 1],
                                 initializer=initializer)
            pw = tf.get_variable("pointwise_kernel", [1, 1, in_ch, out_channels],
                                 initializer=initializer)
            bias = tf.get_variable("bias", [out_channels],
                                   initializer=tf.zeros_initializer())
            out = tf.nn.separable_conv2d(batch_input, dw, pw,
                                         strides=[1, 2, 2, 1], padding="SAME")
            return tf.nn.bias_add(out, bias)
    else:
        with tf.variable_scope("conv2d"):
            in_ch = batch_input.get_shape().as_list()[-1]
            kernel = tf.get_variable("kernel", [4, 4, in_ch, out_channels],
                                     initializer=initializer)
            bias = tf.get_variable("bias", [out_channels],
                                   initializer=tf.zeros_initializer())
            out = tf.nn.conv2d(batch_input, kernel,
                               strides=[1, 2, 2, 1], padding="SAME")
            return tf.nn.bias_add(out, bias)


def gen_deconv(batch_input, out_channels, separable_conv):
    # Matches tf.layers.conv2d_transpose / resize + tf.layers.separable_conv2d
    # Variable scope: conv2d_transpose/{kernel,bias}  or  separable_conv2d/{...}
    initializer = tf.random_normal_initializer(0, 0.02)
    if separable_conv:
        _b, h, w, _c = batch_input.shape
        resized = tf.image.resize_images(batch_input, [h * 2, w * 2],
                                         method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        with tf.variable_scope("separable_conv2d"):
            in_ch = resized.get_shape().as_list()[-1]
            dw = tf.get_variable("depthwise_kernel", [4, 4, in_ch, 1],
                                 initializer=initializer)
            pw = tf.get_variable("pointwise_kernel", [1, 1, in_ch, out_channels],
                                 initializer=initializer)
            bias = tf.get_variable("bias", [out_channels],
                                   initializer=tf.zeros_initializer())
            out = tf.nn.separable_conv2d(resized, dw, pw,
                                         strides=[1, 1, 1, 1], padding="SAME")
            return tf.nn.bias_add(out, bias)
    else:
        with tf.variable_scope("conv2d_transpose"):
            in_ch = batch_input.get_shape().as_list()[-1]
            kernel = tf.get_variable("kernel", [4, 4, out_channels, in_ch],
                                     initializer=initializer)
            bias = tf.get_variable("bias", [out_channels],
                                   initializer=tf.zeros_initializer())
            input_shape = tf.shape(batch_input)
            output_shape = tf.stack([
                input_shape[0],
                input_shape[1] * 2,
                input_shape[2] * 2,
                out_channels,
            ])
            out = tf.nn.conv2d_transpose(batch_input, kernel,
                                         output_shape=output_shape,
                                         strides=[1, 2, 2, 1], padding="SAME")
            return tf.nn.bias_add(out, bias)


def create_generator(generator_inputs, generator_outputs_channels, ngf, separable_conv):
    layers = []

    with tf.variable_scope("encoder_1"):
        output = gen_conv(generator_inputs, ngf, separable_conv)
        layers.append(output)

    encoder_specs = [
        ngf * 2,
        ngf * 4,
        ngf * 8,
        ngf * 8,
        ngf * 8,
        ngf * 8,
        ngf * 8,
    ]

    for out_channels in encoder_specs:
        with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
            rectified = lrelu(layers[-1], 0.2)
            convolved = gen_conv(rectified, out_channels, separable_conv)
            output = batchnorm(convolved)
            layers.append(output)

    decoder_specs = [
        (ngf * 8, 0.5),
        (ngf * 8, 0.5),
        (ngf * 8, 0.5),
        (ngf * 8, 0.0),
        (ngf * 4, 0.0),
        (ngf * 2, 0.0),
        (ngf, 0.0),
    ]

    num_encoder_layers = len(layers)
    for decoder_layer, (out_channels, dropout) in enumerate(decoder_specs):
        skip_layer = num_encoder_layers - decoder_layer - 1
        with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
            inp = layers[-1] if decoder_layer == 0 else tf.concat([layers[-1], layers[skip_layer]], axis=3)
            rectified = tf.nn.relu(inp)
            output = gen_deconv(rectified, out_channels, separable_conv)
            output = batchnorm(output)
            if dropout > 0.0:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)
            layers.append(output)

    with tf.variable_scope("decoder_1"):
        inp = tf.concat([layers[-1], layers[0]], axis=3)
        rectified = tf.nn.relu(inp)
        output = gen_deconv(rectified, generator_outputs_channels, separable_conv)
        output = tf.tanh(output)
        layers.append(output)

    return layers[-1]


def main():
    options_path = os.path.join(a.checkpoint_dir, "options.json")
    if not os.path.exists(options_path):
        raise Exception("options.json not found in %s" % a.checkpoint_dir)

    with open(options_path) as f:
        opts = json.load(f)

    ngf = opts.get("ngf", 64)
    separable_conv = opts.get("separable_conv", False)

    checkpoint = tf.train.latest_checkpoint(a.checkpoint_dir)
    if checkpoint is None:
        raise Exception("no checkpoint found in %s" % a.checkpoint_dir)

    print("checkpoint:     ", checkpoint)
    print("ngf=%d  separable_conv=%s" % (ngf, separable_conv))

    # Float32 I/O graph — required for ONNX compatibility.
    # Input:  [1, 256, 256, 3] float32 in [0, 1]
    # Output: [1, 256, 256, 3] float32 in [0, 1]
    input_image = tf.placeholder(tf.float32, shape=[1, CROP_SIZE, CROP_SIZE, 3],
                                 name="input_image")

    with tf.variable_scope("generator"):
        batch_output = deprocess(create_generator(preprocess(input_image), 3, ngf, separable_conv))

    output_image = tf.identity(batch_output, name="output_image")

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        saver.restore(sess, checkpoint)
        print("checkpoint loaded")

        frozen_graph_def = graph_util.convert_variables_to_constants(
            sess, sess.graph.as_graph_def(), ["output_image"]
        )

    output_dir = os.path.dirname(a.output_pb)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with tf.gfile.GFile(a.output_pb, "wb") as f:
        f.write(frozen_graph_def.SerializeToString())

    node_count = len(frozen_graph_def.node)
    print("frozen model saved to %s (%d nodes)" % (a.output_pb, node_count))
    print()
    print("tf2onnx command:")
    print("  python -m tf2onnx.convert --graphdef %s --inputs input_image:0 --outputs output_image:0 --output model.onnx" % a.output_pb)


main()
