import tensorflow as tf
class Pruning:
    def __init__(self):
        print("Pruning class initialized")
    
    def weight_pruning(self, w: tf.Variable, k: float) -> tf.Variable:
        
        k = tf.cast(
            tf.round(
                tf.size(w, out_type=tf.float32) * tf.constant(k)
                ), 
            dtype=tf.int32
        )
        
        w_reshaped = tf.reshape(w, [-1])
        
        _, indices = tf.nn.top_k(
            tf.negative(
                tf.abs(w_reshaped)
            ),
            dtype=tf.int32,
        )
        
        mask = tf.scatter_nd_update(
            tf.Variable(
                tf.ones_like(w_reshaped, dtype=tf.float32), name='mask', trainable=False
            ),
            tf.reshape(indices, [-1, 1]),
            tf.zeros([k], dtype=tf.float32)
        )
        
        return w.assign(tf.reshape(w_reshaped * mask, tf.shape(w)))
    
    def unit_pruning(self, w: tf.Variable, k: float) -> tf.Variable:
        
        k = tf.cast(
            tf.round(
                tf.cast(
                    tf.shape(w)[1], tf.float32
                ) * tf.constant(k)
            ),
            dtype=tf.int32
        )
        
        norm = tf.norm(w, axis=0)
        
        row_indices = tf.tile(tf.range(tf.shape(w)[0]), [k])
        
        _, col_indices = tf.nn.top_k(
            tf.negative(norm),
            k,
            sorted=True,
            name=None
        )
        
        col_indices = tf.reshape(
            tf.tile(
                tf.reshape(col_indices, [-1, 1]), [1, tf.shape(w)[0]]
            ),
            [-1]
        )
        
        indices = tf.stack([row_indices, col_indices], axis=1)
        
        return w.assign(
            tf.scatter_nd_update
            (w, indices, tf.zeros(tf.shape(w)[0] * k, tf.float32))
        )
        
    def prune_selection(self, w: tf.Variable, k: float, selection: str) -> tf.Variable:
        if selection is None:
            return w
        elif selection == 'weight':
            return self.weight_pruning(w, k)
        elif selection == 'unit':
            return self.unit_pruning(w, k)
        else:
            raise ValueError(f'Invalid selection: {selection}')
