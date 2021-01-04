import numpy as np
import tensorflow as tf
import util.util_funcs as uf
strategy = tf.distribute.MirroredStrategy()

BUFFER_SIZE = 10000

BATCH_SIZE_PER_REPLICA = 16
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
print(strategy.num_replicas_in_sync)


def input_fn(BUFFER_SIZE, BATCH_SIZE):
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    train_images = train_images[..., None]
    test_images = test_images[..., None]

    train_images = train_images / np.float32(255)
    test_images = test_images / np.float32(255)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(BUFFER_SIZE).batch(
        BATCH_SIZE)
    train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)

    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(BATCH_SIZE)
    test_dist_dataset = strategy.experimental_distribute_dataset(test_dataset)

    return train_dist_dataset, test_dist_dataset


def get_model(input_tensor):
    # input_tensor = tf.keras.layers.Input(shape=(height, width, channel), batch_size=batch, name="yolo_input")
    conv1 = tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1))(input_tensor)
    conv1 = tf.keras.layers.MaxPooling2D()(conv1)
    conv1 = tf.keras.layers.Flatten()(conv1)
    conv1 = tf.keras.layers.Dense(64, activation='relu')(conv1)
    conv1 = tf.keras.layers.Dense(10)(conv1)
    output = tf.nn.softmax(conv1)

    return tf.keras.Model(input_tensor, output, name="layer_test")


def get_loss(pred, grtr):
    grtr_onehot = tf.one_hot(grtr, depth=10, axis=-1)
    loss = tf.losses.categorical_crossentropy(grtr_onehot, pred)
    return tf.reduce_sum(loss)


def main():
    input_tensor = tf.keras.layers.Input(shape=(28, 28, 1), batch_size=BATCH_SIZE, name="multi_testing")
    total_loss = 0
    with strategy.scope():
        train_dataset, test_dataset = input_fn(BUFFER_SIZE, BATCH_SIZE)
        model = get_model(input_tensor)
        optimizer = tf.optimizers.Adam(lr=0.0001)
        for i, data in enumerate(train_dataset):
            single_loss = distributed_train_step(data, model, optimizer)
            total_loss += single_loss
            uf.print_progress_status(f"step : {i}" 
                                     f"total loss={total_loss:1.4f}")
        loss_result = total_loss / BATCH_SIZE


@tf.function
def distributed_train_step(dataset_inputs, model, optimizer):
    image, label = dataset_inputs

    def step_fn(image, label):
        with tf.GradientTape() as tape:
            pred = model(image)
            grtr = label
            loss = get_loss(pred, grtr) / BATCH_SIZE

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss[tf.newaxis]

    per_example_losses = strategy.run(step_fn, args=(image, label))
    sum_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_example_losses, axis=0)
    return sum_loss


if __name__ == "__main__":
    main()

