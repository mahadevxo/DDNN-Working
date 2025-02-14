from tensorflow.keras import applications as apps  # type: ignore
import tensorflow as tf  # type: ignore
tf.config.set_visible_devices([], 'GPU')

def save_model(model, filename):
    model.save(f"{filename}.keras")

def weight_pruning(weights, k: float):
    # Flatten the weights
    weights_flat = tf.reshape(weights, [-1])
    num_elements = tf.size(weights_flat)

    # Calculate number of weights to prune
    num_prune = tf.cast(tf.round(tf.cast(num_elements, tf.float32) * k), dtype=tf.int32)

    # Prune smallest absolute weights
    _, indices = tf.nn.top_k(-tf.abs(weights_flat), k=num_prune)

    # Create mask for remaining weights
    mask = tf.ones_like(weights_flat)
    mask = tf.tensor_scatter_nd_update(mask, tf.reshape(indices, [-1, 1]), tf.zeros([num_prune]))

    # Return pruned weights reshaped to original
    return (weights_flat * mask).numpy().reshape(weights.shape)

def apply_pruned_weights(layer, pruning_fn, k):
    if weights := layer.get_weights():
        # Apply pruning only to the kernel weights
        pruned_kernel = pruning_fn(weights[0], k)
        # Set pruned kernel with the original biases intact
        layer.set_weights([pruned_kernel] + weights[1:])

# Load VGG16 and freeze the entire model
model = apps.VGG16(weights='imagenet', include_top=True)
model.trainable = False

save_model(model, 'original_model')

# Apply pruning to convolutional layers
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.Conv2D):
        layer.trainable = True
        apply_pruned_weights(layer, weight_pruning, 0.5)
        layer.trainable = False

save_model(model, 'unit_pruned_model')