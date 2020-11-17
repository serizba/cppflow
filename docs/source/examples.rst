Examples
========

This guide show how to create and run some basic examples that show the functionality of cppflow. The code of these examples is available on the `GitHub repo <https://github.com/serizba/cppflow/tree/cppflow2/examples>`_. To run these examples, just use the provided CMake on each of them (after having installed the TF C API as in :ref:`Installation`):

.. code:: bash

    git clone git@github.com:serizba/cppflow.git
    cd cppflow/examples/load_model
    mkdir build
    cd build
    cmake ..
    make
    ./example


Create and load model
---------------------

To create a model that you can open with cppflow you just need to create a ``tf.Module`` or a ``tf.keras.Model`` and save it. Using the functional API of keras this is as easy as:

.. code:: python

    import tensorflow as tf


    input = tf.keras.Input(shape=(5,))

    output = tf.keras.layers.Dense(5, activation=tf.nn.relu)(input)
    output = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(output)
    model = tf.keras.Model(inputs=input, outputs=output)

    model.compile()

    # Export the model to a SavedModel
    model.save('model', save_format='tf')

Now a new directory named ``model`` is created, and it contains the saved model. You can open it from cppflow using the ``model`` class, to then feed it with a tensor to obtain the output.

.. code:: c++

    #include <iostream>
    #include "cppflow/cppflow.h"


    int main() {

        auto input = cppflow::fill({10, 5}, 1.0f);
        cppflow::model model("../model");
        auto output = model(input);
        
        std::cout << output << std::endl;

        return 0;
    }

.. _EfficientNet:

Inference on EfficientNet
-------------------------

For this example we use a pretrained EfficientNet network that is available in `Keras applications <https://www.tensorflow.org/api_docs/python/tf/keras/applications>`_. Running the following code will create a ``model`` directory with the definition of the EfficientNet and its weights.

.. code:: python

    import tensorflow as tf

    model = tf.keras.applications.EfficientNetB0()

    # Export the model to a SavedModel
    model.save('model', save_format='tf')

Now we can open the model from cppflow and perform inference with a real image. 

.. image:: my_cat.jpg
   :width: 300
   :alt: Inference on EfficientNet from c++ with a picture of a cat

We can load the image using ``cppflow::read_file`` and ``cppflow::decode_jpeg``. Then we have to convert it to float and feed it to the network. 

.. code:: c++

    #include <iostream>
    #include "cppflow/cppflow.h"


    int main() {

        auto input = cppflow::decode_jpeg(cppflow::read_file(std::string("../my_cat.jpg")));
        input = cppflow::cast(input, TF_UINT8, TF_FLOAT);
        input = cppflow::expand_dims(input, 0);
        cppflow::model model("../model");
        auto output = model(input);

        std::cout << "It's a tiger cat: " << cppflow::arg_max(output, 1) << std::endl;
        
        return 0;
    }

To see the prediction of the network we apply ``cppflow::arg_max`` to the ouput and it will show the number of the predicted class, which corresponds with a tiger cat.

.. _MultiInputOutput:

Multi input/output model
------------------------

For this example we will create a Keras model that takes two inputs and produce two outputs:

.. code:: python

    import tensorflow as tf

    input_1 = tf.keras.Input(shape=(5,), name='my_input_1')
    input_2 = tf.keras.Input(shape=(5,), name='my_input_2')

    x1 = tf.keras.layers.Dense(5, activation=tf.nn.relu)(input_1)
    x2 = tf.keras.layers.Dense(5, activation=tf.nn.relu)(input_2)

    output_1 = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid, name='my_outputs_1')(x1)
    output_2 = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid, name='my_outputs_2')(x2)

    model = tf.keras.Model(inputs=[input_1, input_2], outputs=[output_1, output_2])

    model.compile()

    # Export the model to a SavedModel
    model.save('model', save_format='tf')

Now, we will inspect the model with the `saved_model_cli <https://www.tensorflow.org/guide/saved_model#show_command>`_ to retrieve the name of the operations, and to know how to call the model.

.. code-block:: shell-session            
                
    $ saved_model_cli show --dir model
    'serve'
    $ saved_model_cli show --dir model --tag_set serve
    SignatureDef key: "__saved_model_init_op"
    SignatureDef key: "serving_default"
    $ saved_model_cli show --dir model --tag_set serve --signature_def serving_default
    The given SavedModel SignatureDef contains the following input(s):
      inputs['my_input_1'] tensor_info:
          dtype: DT_FLOAT
          shape: (-1, 5)
          name: serving_default_my_input_1:0
      inputs['my_input_2'] tensor_info:
          dtype: DT_FLOAT
          shape: (-1, 5)
          name: serving_default_my_input_2:0
    The given SavedModel SignatureDef contains the following output(s):
      outputs['my_outputs_1'] tensor_info:
          dtype: DT_FLOAT
          shape: (-1, 1)
          name: StatefulPartitionedCall:0
      outputs['my_outputs_2'] tensor_info:
          dtype: DT_FLOAT
          shape: (-1, 1)
          name: StatefulPartitionedCall:1
    Method name is: tensorflow/serving/predict

From this output we can see that there are two inputs (``serving_default_my_input_1:0`` and ``serving_default_my_input_2:0``) and two outputs (``StatefulPartitionedCall:0`` and ``StatefulPartitionedCall:1``). You can run the model specifying multiple inputs as a vector of tuples ``<name of the input, input tensor`` and multiple outputs as a vector with the name of the outputs:

.. code:: c++


    #include <iostream>
    #include "cppflow/cppflow.h"

    int main() {

        auto input_1 = cppflow::fill({10, 5}, 1.0f);
        auto input_2 = cppflow::fill({10, 5}, -1.0f);
        cppflow::model model("../model");

        auto output = model({{"serving_default_my_input_1:0", input_1}, {"serving_default_my_input_2:0", input_2}}, {"StatefulPartitionedCall:0", "StatefulPartitionedCall:1"});

        std::cout << "output_1: " << output[0] << std::endl;
        std::cout << "output_2: " << output[1] << std::endl;
        return 0;
    }
