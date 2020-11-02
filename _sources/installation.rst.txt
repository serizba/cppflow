.. _Installation:

Installation
============

One of the advantages of cppflow is that you don't need to compile or install TensorFlow, you just need to download the `TF C API <https://www.tensorflow.org/install/lang_c>`_. As cppflow is a header-only library, once you have the C API, the only thing you need to do is include the cppflow files from your project.

To install the C API in your system you have two options:

Install the TF C API globally
-----------------------------
You can install the C API in a system directory and do not worry about it again. For this, you just have to `download it <https://www.tensorflow.org/install/lang_c>`_ it and then:

.. code:: bash

    sudo tar -C /usr/local -xzf (downloaded file)
    sudo ldconfig

Install the TF C API in custom directory
----------------------------------------
.. note::
    The easiest way is to extract the library into your HOME directory, in a folder called ``libtensorflow2``. In this case, you can directly use the CMake files from the examples, which will search for the library in your HOME directory.

    ``mkdir -p ~/libtensorflow2/ && tar -C ~/libtensorflow2/ -xzf (downloaded file)``

You can also install the library in a custom directory. In this case, after `downloading it <https://www.tensorflow.org/install/lang_c>`_ and unpacking it you will need to update your PATH or add a hint to your ``CMakeLists.txt`` to find the library.

.. code:: bash

    mkdir -p /path/to/mydir/
    tar -C /path/to/mydir -xzf (downloaded file)

Now, update your path:

.. code:: bash

    export LIBRARY_PATH=$LIBRARY_PATH:/path/to/mydir/lib
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/mydir/lib

Or add a hint in ``CMakeLists.txt`` as in the examples:

.. code:: cmake

    find_library(TENSORFLOW_LIB tensorflow HINT /path/to/mydir/lib)


You are done, now you can proceed to build your :ref:`first example<First example>`.
