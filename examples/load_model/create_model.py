import tensorflow as tf

def example_1():

    # Two simple inputs
    a = tf.placeholder(tf.float32, shape=(1, 100), name="input_a")
    b = tf.placeholder(tf.float32, shape=(1, 100), name="input_b")

    # Output
    c = tf.add(a, b, name='result')

    # To add an init operation to the model
    i = tf.initializers.global_variables()

    # Write the model definition
    with open('load_model.pb', 'wb') as f:
        f.write(tf.get_default_graph().as_graph_def().SerializeToString())


if __name__ == "__main__":
    example_1()
