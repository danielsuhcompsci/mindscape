import tensorflow as tf

#input: (batch_size, channel, height, width); output: (batch_size, voxel_dim)
def train(input: tf.Tensor, shape):
    model = tf.keras.applications.resnet50.ResNet50(
        include_top=True,
        weights='imagenet',
        input_tensor=input,
        input_shape=shape,
        pooling=None,
        classes=1000
    )
    model.compile(optimizer="Adam", loss='mse', metrics=['accuracy'])
    return model
