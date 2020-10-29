
Quickstart
==========

.. _First example:

First example
-------------

Once you have downloaded the TF C API and cppflow you can start playing with tensors from C++:

.. code:: c++

    #include <iostream>
    #include "cppflow/cppflow.h"

    int main() {
        
        // Create a tensor from a list, a = [1.0, 2.0, 3.0]
        auto a = cppflow::tensor({1.0, 2.0, 3.0});
        // Create a tensor of shape 3 filled with 1.0, b = [1.0, 2.0, 3.0]
        auto b = cppflow::fill({3}, 1.0);

        std::cout << a + b << std::endl;

        return 0;
    }

Easy right?, now you can compile it with the terminal (if you have configured the TF C API as stated in :ref:`Installation`) using the following command:

.. code:: bash

    g++ -std=c++17 -o main.out -I /path/to/libtensorflow2/include/ -I/path/to/cppflow/include/ main.cpp -ltensorflow
    ./main.out

You should see the result of ``a + b``:

.. code::

   (tensor: shape=[3], data=
    [2 3 4])

Using CMake
-----------

Probably a more convenient way of compiling your code is using CMake. 

.. code:: cmake

    cmake_minimum_required(VERSION 3.10)
    project(example)

    find_library(TENSORFLOW_LIB tensorflow HINT $ENV{HOME}/libtensorflow2/lib)

    set(CMAKE_CXX_STANDARD 17)

    add_executable(example main.cpp)
    target_include_directories(example PRIVATE /path/to/cppflow/include/ $ENV{HOME}/libtensorflow2/include)
    target_link_libraries (example "${TENSORFLOW_LIB}")



Load a model
------------

You can easily run TensorFlow models with cppflow by loading a `saved model <https://www.tensorflow.org/guide/saved_model>`_. Imagine you have a model saved on the folder ``coolpredictor`` that takes an image as an input and produces a vector of probabilities of the class belonging to each possible class. You can load the model, read your image, preprocess it, run the model and get the output just using cppflow:

.. code:: c++

    #include <iostream>
    #include "cppflow/cppflow.h"


    int main() {

        // Load the model
        cppflow::model model("coolpredictor");
        
        // Load an image
        auto input = cppflow::decode_jpeg(cppflow::read_file(std::string("image.jpg")));

        // Cast it to float, normalize to range [0, 1], and add batch_dimension
        input = cppflow::cast(input, TF_UINT8, TF_FLOAT);
        input = input / 255.f;
        input = cppflow::expand_dims(input, 0);

        // Run
        auto output = model(input);

        // Show the predicted class
        std::cout << cppflow::arg_max(output, 1) << std::endl;
    }

For a complete (runnable) example you can check :ref:`this example<EfficientNet>`, which performs inference on an EfficientNet trained on ImageNet.

Complex model call (multi input/output models)
----------------------------------------------

By default, calling ``model(input)`` will use the input operation named ``serving_default_input_1`` and output operation ``StatefulPartitionedCall``. If you need to use other operations or you need to use multiple inputs or outputs you can directly specify your desired inputs and outputs operations:

.. code:: c++

    
    // Calling model in default mode
    auto output = model(input);

    // Calling model as in default mode (but specifying signature)
    auto output = model({{"serving_default_input_1:0", input}},{"StatefulPartitionedCall:0"});

    // Calling model with two inputs, named "serving_default_input_1" and "serving_default_input_2"
    auto output = model({{"serving_default_input_1:0", input1}, {"serving_default_input_2:0", input2}},{"StatefulPartitionedCall:0"});

    // Calling model with two outputs, named "StatefulPartitionedCall:0" and "StatefulPartitionedCall:1"
    auto output = model({{"serving_default_input_1:0", input}},{"StatefulPartitionedCall:0", "StatefulPartitionedCall:1"});

    // Calling model with two inputs and two outputs
    auto output = model({{"serving_default_my_input_1:0", input_1}, {"serving_default_my_input_2:0", input_2}}, {"StatefulPartitionedCall:0", "StatefulPartitionedCall:1"});

.. note::
    If you don't know the name of the operations of your model you can use the `saved_model_cli <https://www.tensorflow.org/guide/saved_model#show_command>`_ to print all the information:

    ``saved_model_cli show --dir /path/to/model --all``

    Or you can use ``model::get_operations()`` to retrieve the name of the available operations.


For a complete (runnable) example you can check :ref:`this example<MultiInputOutput>`., which uses a toy model with two inputs and two outputs.
