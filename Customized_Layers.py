import tensorflow as tf;

def Generator(image_Pattern, attribute_Pattern, initial_Filter_Count = 64, reuse = False):    
    with tf.variable_scope('generator', reuse=reuse):
        tiled_Attribute_Pattern = tf.tile(
            tf.expand_dims(tf.expand_dims(attribute_Pattern, axis=1), axis=1),
            multiples=(1, image_Pattern.get_shape()[1], image_Pattern.get_shape()[2], 1)
            )

        down_Sampling_Activation = tf.concat([image_Pattern, tiled_Attribute_Pattern], axis=3);
        for index, (filter_Count, kernel_Size, stride) in enumerate([(initial_Filter_Count, 7, 1), (initial_Filter_Count * 2, 4, 2), (initial_Filter_Count * 4, 4, 2)]):
            down_Sampling_Activation = tf.nn.relu(
                tf.contrib.layers.instance_norm(
                    inputs= tf.layers.conv2d(
                        inputs = down_Sampling_Activation, 
                        filters = filter_Count, 
                        kernel_size = kernel_Size,
                        strides = stride,
                        padding = "same",
                        use_bias = False,
                        kernel_initializer = tf.truncated_normal_initializer(stddev=0.02)
                        ),
                    epsilon = 1e-5
                    ),
                name="down_Sampling_Layer{}".format(index)
                )

        bottleneck_Activation = down_Sampling_Activation;
        for index in range(6):
            previous_Bottleneck_Activation = bottleneck_Activation;
            bottleneck_Activation = tf.nn.relu(
                tf.contrib.layers.instance_norm(
                    inputs= tf.layers.conv2d(
                        inputs = bottleneck_Activation, 
                        filters = initial_Filter_Count * 4, 
                        kernel_size = 3,
                        strides = 1,
                        padding = "same",
                        use_bias = False,
                        kernel_initializer = tf.truncated_normal_initializer(stddev=0.02)
                        ),
                    epsilon = 1e-5
                    ),
                name="bottleneck_Layer{}_0".format(index)
                )
            bottleneck_Activation = tf.contrib.layers.instance_norm(
                inputs= tf.layers.conv2d(
                    inputs = bottleneck_Activation, 
                    filters = initial_Filter_Count * 4, 
                    kernel_size = 3,
                    strides = 1,
                    padding = "same",
                    use_bias = False,
                    kernel_initializer = tf.truncated_normal_initializer(stddev=0.02)
                    ),
                epsilon = 1e-5,
                scope="bottleneck_Layer{}_1".format(index)
                ) + previous_Bottleneck_Activation

        up_Sampling_Activation = bottleneck_Activation;
        for index, (filter_Count, kernel_Size, stride) in enumerate([(initial_Filter_Count * 2, 4, 2), (initial_Filter_Count, 4, 2)]):
            up_Sampling_Activation = tf.nn.relu(
                tf.contrib.layers.instance_norm(
                    inputs= tf.layers.conv2d_transpose(
                        inputs = up_Sampling_Activation, 
                        filters = filter_Count, 
                        kernel_size = kernel_Size,
                        strides = stride,
                        padding = "same",
                        use_bias = False,
                        kernel_initializer = tf.truncated_normal_initializer(stddev=0.02)
                        ),
                    epsilon = 1e-5
                    ),
                name="up_Sampling_Layer{}".format(index)
                )
        
        fake_Image = tf.layers.conv2d(
            inputs = up_Sampling_Activation, 
            filters = 3,    #RGB
            kernel_size = 7,
            strides = 1,
            padding = "same",
            activation = tf.nn.tanh,
            use_bias = False,
            kernel_initializer = tf.truncated_normal_initializer(stddev=0.02),
            name="generated_Pattern"
            )

        return fake_Image;

def Discriminator(image_Pattern, initial_Filter_Count = 64, attribute_Count = 10, reuse = False):
    with tf.variable_scope('discriminator', reuse=reuse):
        hidden_Activation = image_Pattern;
        for index in range(6):
            hidden_Activation = tf.nn.leaky_relu(
                tf.layers.conv2d(
                    inputs = hidden_Activation, 
                    filters = initial_Filter_Count * (2 ** index), 
                    kernel_size = 4,
                    strides = 2,
                    padding = "same",
                    kernel_initializer = tf.truncated_normal_initializer(stddev=0.02)
                    ),
                alpha=0.01,
                name="hidden_Layer{}".format(index)
                )

        output_Activation = tf.layers.conv2d(
            inputs = hidden_Activation, 
            filters = 1 + attribute_Count,
            kernel_size = hidden_Activation.get_shape()[1:3],
            strides = 1,
            padding = "valid",
            name = "output_Layer",
            use_bias = False,
            kernel_initializer = tf.truncated_normal_initializer(stddev=0.02)
            )

        discrimination_Logit, attribute_Logit = tf.split(
            tf.squeeze(output_Activation, axis=[1,2]),
            num_or_size_splits = [1, attribute_Count],
            axis = 1
            )

        return discrimination_Logit, attribute_Logit;


def WGAN_Gradient_Penalty_Loss(
    original_Image,
    fake_Image,
    initial_Filter_Count = 64,
    attribute_Count = 10,
    discriminator_Reuse = True
    ):
    alpha =  tf.random_uniform(
        shape = [tf.shape(original_Image)[0], 1, 1, 1],
        minval= 0,
        maxval= 1
        )

    hat_Image = alpha * original_Image + (1 - alpha) * fake_Image
    hat_Logit, _ = Discriminator(
        hat_Image,
        initial_Filter_Count,
        attribute_Count,
        reuse=discriminator_Reuse
        )
    gradients = tf.gradients(hat_Logit, xs=[hat_Image])[0];
    slopes = tf.sqrt(tf.reduce_sum(gradients ** 2, axis=[1,2,3]));
    gradient_Penalty = tf.reduce_mean((slopes - 1) ** 2);
    
    return gradient_Penalty

if __name__ == "__main__":
    x = tf.placeholder(tf.float32, [None, 128, 128, 3])
    y = tf.placeholder(tf.float32, [None, 10])
    t = Generator(x, y);
    t2, t3 = Discriminator(t)
    print(t)
    print(t2)
    print(t3)