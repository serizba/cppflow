import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def main():

    inputs = tf.placeholder(tf.float64, shape=(None, 784), name='input')
    labels = tf.placeholder(tf.float64, shape=(None, 10), name='label')

    x = tf.keras.layers.Dense(128, activation='relu')(inputs)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    preds = tf.keras.layers.Dense(10, activation='softmax')(x)
    preds = tf.identity(preds, name="prediction")
    print(preds)


    loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(labels, preds))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    # Initialize all variables
    init_op = tf.global_variables_initializer()

    # Saver
    saver = tf.train.Saver()

    # Save graph definition
    # Write the model definition
    with open('model.pb', 'wb') as f:
        f.write(tf.get_default_graph().as_graph_def().SerializeToString())

    # MNIST data
    mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True)

    with tf.Session() as sess:
        sess.run(init_op)

        for i in range(10000):
            if i%100 == 0:
                print(i)
            batch = mnist_data.train.next_batch(50)
            l = sess.run([loss, train_step], feed_dict={inputs: batch[0], labels: batch[1]})[0]

        saver.save(sess, "checkpoint/train.ckpt")
        print("Finished with loss=", l)


if __name__=="__main__":
    main()