import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.patches as patches
import os
import cv2
import torch
from PIL import Image

train_img_files = os.listdir('./falldetection/fall_dataset/images/train')
train_img_files.sort()
r1 = './falldetection/fall_dataset/images/train/'
train_label_files = os.listdir('./falldetection/fall_dataset/labels/train')
train_label_files.sort()
r2 = './falldetection/fall_dataset/labels/train/'

complete_images = []
complete_class = []

for i in range(len(train_img_files)):
    img = plt.imread(r1+train_img_files[i])
    with open(r2+train_label_files[i],'r') as file:
        r = file.readlines()
    bounding_boxes = []
    for j in r:
        j = j.split()
        bounding_boxes.append([int(j[0]),float(j[1]),float(j[2]),float(j[3]),float(j[4])])
    for box in bounding_boxes:
        image_height, image_width, _ = img.shape
        xmin, ymin, width, height = box[1:]
        xmin = int(xmin * image_width)
        ymin = int(ymin * image_height)
        width = int(width * image_width)
        height = int(height * image_height)
        complete_class.append(box[0])
        complete_images.append(img[ymin-height//2:ymin+height//2, xmin-width//2:xmin+width//2])
#resize
pref_size = (224,224)
for i in range(len(complete_images)):
    complete_images[i] = cv2.resize(complete_images[i],pref_size)
# preporcessing
df = pd.DataFrame()
df['Images'] = complete_images
df['Class'] = complete_class
df['Images']/=255
####
class MultiHeadSelfAttention(Layer):
    """
    Multi-Head Self Attention Layer.

    This layer implements the multi-head self-attention mechanism used in transformers.
    It projects the input into multiple heads, performs scaled dot-product attention
    on each head, and then concatenates and projects the results.

    Attributes:
        embed_dim: Dimensionality of the embedding.
        num_heads: Number of attention heads.
        dropout_rate: Dropout rate for regularization.
    """

    def __init__(self, embed_dim=256, num_heads=8, dropout_rate=0.1):
        """
        Initialize the layer.

        Args:
            embed_dim: Dimensionality of the embedding.
            num_heads: Number of attention heads.
            dropout_rate: Dropout rate for regularization.
        """
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.dropout_rate = dropout_rate

        if embed_dim % num_heads != 0:
            raise ValueError(f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}")

        self.projection_dim = embed_dim // num_heads

        # Define dense layers for query, key, and value projections
        self.query_dense = Dense(embed_dim)
        self.key_dense = Dense(embed_dim)
        self.value_dense = Dense(embed_dim)

        # Define dense layer to combine the heads
        self.combine_heads = Dense(embed_dim)

        # Define dropout and layer normalization layers
        self.dropout = Dropout(dropout_rate)
        self.layernorm = LayerNormalization(epsilon=1e-6)

    def attention(self, query, key, value):
        """
        Compute scaled dot-product attention.

        Args:
            query: Query tensor.
            key: Key tensor.
            value: Value tensor.

        Returns:
            attention: Result of the attention mechanism.
        """
        score = tf.matmul(query, key, transpose_b=True)  # Calculate dot product
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)  # Get dimension of key
        scaled_score = score / tf.math.sqrt(dim_key)  # Scale the scores
        weights = tf.nn.softmax(scaled_score, axis=-1)  # Apply softmax to get attention weights
        attention = tf.matmul(weights, value)  # Multiply weights with values
        return attention

    def separate_heads(self, x, batch_size):
        """
        Separate the heads for multi-head attention.

        Args:
            x: Input tensor.
            batch_size: Batch size of the input.

        Returns:
            x: Tensor with separated heads.
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        """
        Forward pass for the layer.

        Args:
            inputs: Input tensor.

        Returns:
            output: Output tensor after applying multi-head self-attention.
        """
        batch_size = tf.shape(inputs)[0]

        # Project inputs to query, key, and value tensors
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)

        # Separate the heads for multi-head attention
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        # Compute attention
        attention = self.attention(query, key, value)

        # Concatenate the heads and reshape the tensor
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))

        # Combine heads and apply dropout and layer normalization
        output = self.combine_heads(concat_attention)
        output = self.dropout(output)
        output = self.layernorm(inputs + output)

        # Reduce mean across the time dimension to get fixed-size output
        output = tf.reduce_mean(output, axis=1)
        return output

    def compute_output_shape(self, input_shape):
        """
        Compute the output shape of the layer.

        Args:
            input_shape: Shape of the input tensor.

        Returns:
            Output shape.
        """
        return input_shape[0], self.embed_dim


    def compute_output_shape(self, input_shape):
        return input_shape[0], self.embed_dim
######
#model
def Conv_2D_Block(x, model_width, kernel, strides=(1, 1), padding="same"):
    # 2D Convolutional Block with BatchNormalization
    x = tf.keras.layers.Conv2D(model_width, kernel, strides=strides, padding=padding, kernel_initializer="he_normal")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    return x
def classifier(inputs, class_number):
    # Construct the Classifier Group
    # inputs       : input vector
    # class_number : number of output classes
    out = tf.keras.layers.Dense(class_number, activation='softmax')(inputs)
    return out
def regressor(inputs, feature_number):
    # Construct the Regressor Group
    # inputs         : input vector
    # feature_number : number of output features
    out = tf.keras.layers.Dense(feature_number, activation='linear')(inputs)
    return out
def SE_Block(inputs, num_filters, ratio):
    squeeze = tf.keras.layers.GlobalAveragePooling2D()(inputs)

    excitation = tf.keras.layers.Dense(units=num_filters/ratio)(squeeze)
    excitation = tf.keras.layers.Activation('relu')(excitation)
    excitation = tf.keras.layers.Dense(units=num_filters)(excitation)
    excitation = tf.keras.layers.Activation('sigmoid')(excitation)
    excitation = tf.keras.layers.Reshape([1, 1, num_filters])(excitation)

    scale = inputs * excitation

    return scale
def Inception_ResNet_Module_A(inputs, filterB1_1, filterB2_1, filterB2_2, filterB3_1, filterB3_2, filterB3_3, filterB4_1, i):
    # Inception ResNet Module A - Block i
    branch1x1 = Conv_2D_Block(inputs, filterB1_1, (1, 1))

    branch3x3 = Conv_2D_Block(inputs, filterB2_1, (1, 1))
    branch3x3 = Conv_2D_Block(branch3x3, filterB2_2, (3, 3))

    branch3x3dbl = Conv_2D_Block(inputs, filterB3_1, (1, 1))
    branch3x3dbl = Conv_2D_Block(branch3x3dbl, filterB3_2, (3, 3))
    branch3x3dbl = Conv_2D_Block(branch3x3dbl, filterB3_3, (3, 3))

    branch_concat = tf.keras.layers.concatenate([branch1x1, branch3x3, branch3x3dbl], axis=-1)
    branch1x1_ln = tf.keras.layers.Conv2D(filterB4_1, (1, 1), activation='linear', strides=(1, 1), padding='same', kernel_initializer="he_normal")(branch_concat)

    x = tf.keras.layers.Add(name='Inception_ResNet_Block_A'+str(i))([inputs, branch1x1_ln])
    x = tf.keras.layers.BatchNormalization()(x)
    out = tf.keras.layers.Activation('relu')(x)

    return out


def Inception_ResNet_Module_B(inputs, filterB1_1, filterB2_1, filterB2_2, filterB2_3, filterB3_1, i):
    # Inception ResNet Module B - Block i
    branch1x1 = Conv_2D_Block(inputs, filterB1_1, (1, 1))

    branch7x7 = Conv_2D_Block(inputs, filterB2_1, (1, 1))
    branch7x7 = Conv_2D_Block(branch7x7, filterB2_2, (1, 7))
    branch7x7 = Conv_2D_Block(branch7x7, filterB2_3, (7, 1))

    branch_concat = tf.keras.layers.concatenate([branch1x1, branch7x7], axis=-1)
    branch1x1_ln = tf.keras.layers.Conv2D(filterB3_1, (1, 1), activation='linear', strides=(1, 1), padding='same', kernel_initializer="he_normal")(branch_concat)

    x = tf.keras.layers.Add(name='Inception_ResNet_Block_B'+str(i))([inputs, branch1x1_ln])
    x = tf.keras.layers.BatchNormalization()(x)
    out = tf.keras.layers.Activation('relu')(x)

    return out


def Inception_ResNet_Module_C(inputs, filterB1_1, filterB2_1, filterB2_2, filterB2_3, filterB3_1, i):
    # Inception ResNet Module C - Block i
    branch1x1 = Conv_2D_Block(inputs, filterB1_1, (1, 1))

    branch3x3 = Conv_2D_Block(inputs, filterB2_1, (1, 1))
    branch3x3 = Conv_2D_Block(branch3x3, filterB2_2, (1, 3))
    branch3x3 = Conv_2D_Block(branch3x3, filterB2_3, (3, 1))

    branch_concat = tf.keras.layers.concatenate([branch1x1, branch3x3], axis=-1)
    branch1x1_ln = tf.keras.layers.Conv2D(filterB3_1, (1, 1), activation='linear', strides=(1, 1), padding='same', kernel_initializer="he_normal")(branch_concat)

    x = tf.keras.layers.Add(name='Inception_ResNet_Block_C'+str(i))([inputs, branch1x1_ln])
    x = tf.keras.layers.BatchNormalization()(x)
    out = tf.keras.layers.Activation('relu')(x)

    return out


def Reduction_Block_A(inputs, filterB1_1, filterB2_1, filterB2_2, filterB2_3):
    # Reduction Block A
    branch_pool = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(inputs)

    branch3x3 = Conv_2D_Block(inputs, filterB1_1, 3, strides=(2, 2), padding='valid')

    branch3x3dbl = Conv_2D_Block(inputs, filterB2_1, (1, 1))
    branch3x3dbl = Conv_2D_Block(branch3x3dbl, filterB2_2, (3, 3))
    branch3x3dbl = Conv_2D_Block(branch3x3dbl, filterB2_3, (3, 3), strides=(2, 2), padding='valid')

    x = tf.keras.layers.concatenate([branch_pool, branch3x3, branch3x3dbl], axis=-1, name='Reduction_Block_A')
    x = tf.keras.layers.BatchNormalization()(x)
    out = tf.keras.layers.Activation('relu')(x)

    return out


def Reduction_Block_B(inputs, filterB1_1, filterB1_2, filterB2_1, filterB2_2, filterB3_1, filterB3_2, filterB3_3):
    # Reduction Block B
    branch_pool = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(inputs)

    branch3x3 = Conv_2D_Block(inputs, filterB1_1, (1, 1))
    branch3x3 = Conv_2D_Block(branch3x3, filterB1_2, (3, 3), strides=(2, 2), padding='valid')

    branch3x3_2 = Conv_2D_Block(inputs, filterB2_1, (1, 1))
    branch3x3_2 = Conv_2D_Block(branch3x3_2, filterB2_2, (3, 3), strides=(2, 2), padding='valid')

    branch3x3dbl = Conv_2D_Block(inputs, filterB3_1, (1, 1))
    branch3x3dbl = Conv_2D_Block(branch3x3dbl, filterB3_2, (3, 3))
    branch3x3dbl = Conv_2D_Block(branch3x3dbl, filterB3_3, (3, 3), strides=(2, 2), padding='valid')

    x = tf.keras.layers.concatenate([branch_pool, branch3x3, branch3x3_2, branch3x3dbl], axis=-1)
    x = tf.keras.layers.BatchNormalization()(x)
    out = tf.keras.layers.Activation('relu')(x)

    return out


class SEInception_ResNet:
    def __init__(self, length, width, num_channel, num_filters, ratio=4, problem_type='Classification'
                 output_nums=1, pooling='avg', dropout_rate=False, auxilliary_outputs=False):
        # length: Input Image Length (x-dim)
        # width: Input Image Width (y-dim) [Normally same as the x-dim i.e., Square shape]
        # model_depth: Depth of the Model
        # model_width: Width of the Model
        # kernel_size: Kernel or Filter Size of the Input Convolutional Layer
        # num_channel: Number of Channels of the Input Predictor Signals
        # problem_type: Regression or Classification
        # output_nums: Number of Output Classes in Classification mode and output features in Regression mode
        # pooling: Choose either 'max' for MaxPooling or 'avg' for Averagepooling
        # dropout_rate: If turned on, some layers will be dropped out randomly based on the selected proportion
        # auxilliary_outputs: Two extra Auxullary outputs for the Inception models, acting like Deep Supervision
        self.length = length
        self.width = width
        self.num_channel = num_channel
        self.num_filters = num_filters
        self.ratio = ratio
        self.problem_type = problem_type
        self.output_nums = output_nums
        self.pooling = pooling
        self.dropout_rate = dropout_rate
        self.auxilliary_outputs = auxilliary_outputs

    def MLP(self, x):
        if self.pooling == 'avg':
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
        elif self.pooling == 'max':
            x = tf.keras.layers.GlobalMaxPooling2D()(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = MultiHeadSelfAttention(embed_dim=512, num_heads=8)(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)

        if self.dropout_rate:
            x = tf.keras.layers.Dropout(self.dropout_rate)(x)
        outputs = tf.keras.layers.Dense(self.output_nums, activation='relu')(x)
        if self.problem_type == 'Classification':
            outputs = tf.keras.layers.Dense(self.output_nums, activation='softmax')(x)

        return outputs

    def SEInception_ResNet_v1(self):
        inputs = tf.keras.Input((self.length, self.width, self.num_channel))  # The input tensor
        # Stem
        x = Conv_2D_Block(inputs, 32, 3, strides=(2, 2), padding='valid')
        x = Conv_2D_Block(x, 32, 3, padding='valid')
        x = Conv_2D_Block(x, 64, 3)
        x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = Conv_2D_Block(x, 80, 1)
        x = Conv_2D_Block(x, 192, 3, padding='valid')
        x = Conv_2D_Block(x, 256, 3, strides=(2, 2), padding='valid')

        # 5x Inception ResNet A Blocks - 35 x 35 x 256
        for i in range(5):
            x = Inception_ResNet_Module_A(x, 32, 32, 32, 32, 32, 32, 256, i)
            x = SE_Block(x, int(np.shape(x)[-1]), self.ratio)

        aux_output_0 = []
        if self.auxilliary_outputs:
            # Auxilliary Output 0
            aux_pool = tf.keras.layers.AveragePooling2D(pool_size=(5, 5), strides=(3, 3))(x)
            aux_conv = Conv_2D_Block(aux_pool, 128, (1, 1))
            aux_conv = Conv_2D_Block(aux_conv, 768, (5, 5), padding='valid')
            aux_output_0 = self.MLP(aux_conv)

        x = Reduction_Block_A(x, 384, 192, 224, 256)  # Reduction Block A: 17 x 17 x 768

        # 10x Inception ResNet B Blocks - 17 x 17 x 768
        for i in range(10):
            x = Inception_ResNet_Module_B(x, 128, 128, 128, 128, 896, i)
            x = SE_Block(x, int(np.shape(x)[-1]), self.ratio)

        aux_output_1 = []
        if self.auxilliary_outputs:
            # Auxilliary Output 1
            aux_pool = tf.keras.layers.AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='valid')(x)
            aux_conv = Conv_2D_Block(aux_pool, 128, (1, 1))
            aux_conv = Conv_2D_Block(aux_conv, 768, (5, 5), padding='valid')
            aux_output_1 = self.MLP(aux_conv)

        x = Reduction_Block_B(x, 256, 384, 256, 256, 256, 256, 256)  # Reduction Block B: 8 x 8 x 1280

        # 5x Inception ResNet C Blocks - 8 x 8 x 1280
        for i in range(5):
            x = Inception_ResNet_Module_C(x, 128, 192, 192, 192, 1792, i)
            x = SE_Block(x, int(np.shape(x)[-1]), self.ratio)

        # Final Dense MLP Layer for the outputs
        final_output = self.MLP(x)
        # Create model.
        model = tf.keras.Model(inputs, final_output, name='Inception_v4')
        if self.auxilliary_outputs:
            model = tf.keras.Model(inputs, outputs=[final_output, aux_output_0, aux_output_1], name='Inception_ResNet_v1')

        return model

    def SEInception_ResNet_v2(self):
        inputs = tf.keras.Input((self.length, self.width, self.num_channel))  # The input tensor
        # Stem
        x = Conv_2D_Block(inputs, 32, 3, strides=(2, 2), padding='valid')
        x = Conv_2D_Block(x, 32, 3, padding='valid')
        x = Conv_2D_Block(x, 64, 3)
        #
        branch1 = Conv_2D_Block(x, 96, 3, strides=(2, 2), padding='valid')
        branch2 = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = tf.keras.layers.concatenate([branch1, branch2], axis=-1)
        #
        branch1 = Conv_2D_Block(x, 64, 1)
        branch1 = Conv_2D_Block(branch1, 96, 3, padding='valid')
        branch2 = Conv_2D_Block(x, 64, 1)
        branch2 = Conv_2D_Block(branch2, 64, 7)
        branch2 = Conv_2D_Block(branch2, 96, 3, padding='valid')
        x = tf.keras.layers.concatenate([branch1, branch2], axis=-1)
        #
        branch1 = Conv_2D_Block(x, 192, 3, padding='valid')
        branch2 = tf.keras.layers.MaxPooling2D((3, 3), strides=(1, 1))(x)
        x = tf.keras.layers.concatenate([branch1, branch2], axis=-1)

        # 5x Inception ResNet A Blocks - 35 x 35 x 256
        for i in range(10):
            x = Inception_ResNet_Module_A(x, 32, 32, 32, 32, 48, 64, 384, i)
            x = SE_Block(x, int(np.shape(x)[-1]), self.ratio)

        aux_output_0 = []
        if self.auxilliary_outputs:
            # Auxilliary Output 0
            aux_pool = tf.keras.layers.AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='valid')(x)
            aux_conv = Conv_2D_Block(aux_pool, 96, 1)
            aux_output_0 = self.MLP(aux_conv)

        x = Reduction_Block_A(x, 384, 192, 224, 256)  # Reduction Block A: 17 x 17 x 768

        # 10x Inception ResNet B Blocks - 17 x 17 x 768
        for i in range(20):
            x = Inception_ResNet_Module_B(x, 192, 128, 160, 192, 1024, i)
            x = SE_Block(x, int(np.shape(x)[-1]), self.ratio)

        aux_output_1 = []
        if self.auxilliary_outputs:
            # Auxilliary Output 1
            aux_pool = tf.keras.layers.AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='valid')(x)
            aux_conv = Conv_2D_Block(aux_pool, 128, 1)
            aux_conv = Conv_2D_Block(aux_conv, 768, 5)
            aux_output_1 = self.MLP(aux_conv)

        x = Reduction_Block_B(x, 256, 384, 256, 288, 256, 288, 320)  # Reduction Block B: 8 x 8 x 1280

        # 5x Inception ResNet C Blocks - 8 x 8 x 1280
        for i in range(10):
            x = Inception_ResNet_Module_C(x, 192, 192, 224, 256, 2016, i)
            x = SE_Block(x, int(np.shape(x)[-1]), self.ratio)

        # Final Dense MLP Layer for the outputs
        final_output = self.MLP(x)
        # Create model.
        model = tf.keras.Model(inputs, final_output)
        if self.auxilliary_outputs:
            model = tf.keras.Model(inputs, outputs=[final_output, aux_output_0, aux_output_1])

        return model


if __name__ == '__main__':
    # Configurations
    length = 224  # Length of each Image
    width = 224  # Width of each Image
    model_name = 'SEInceptionResNetV2' 
    model_width = 64 # Width of the Initial Layer, subsequent layers start from here
    num_channel = 3  # Number of Input Channels in the Model
    problem_type = 'Classification' # Classification or Regression
    output_nums = 3  # Number of Class for Classification Problems, always '1' for Regression Problems
    reduction_ratio = 4
    #
    Model = SEInception_ResNet(length, width, num_channel, model_width, ratio=reduction_ratio, problem_type=problem_type, output_nums=output_nums,
                      pooling='max', dropout_rate=False, auxilliary_outputs=False).SEInception_ResNet_v2()
    Model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003), loss=tf.keras.losses..SparseCategoricalCrossentropy(), metrics=['accuracy'])
# Summary of the model architecture(), 
    Model.summary()
