.. _Installation:

Installation
============

One of the advantages of cppflow is that you don't need to compile or install TensorFlow, you just need to download the `TF C API <https://www.tensorflow.org/install/lang_c>`_. Cppflow is a header-only library, and thus you can just include the cppflow files in your project.

To install the C API in your system you have two options:

Install the TF C API globally
-----------------------------
You can install the C API in a system directory and do not worry about it again. For this, you just have to `download it <https://www.tensorflow.org/install/lang_c>`_ it and then:

.. code:: bash

    sudo tar -C /usr/local -xzf (downloaded file)
    sudo ldconfig

Install the TF C API in custom directory
----------------------------------------

You can also install the library in a custom directory. In this case, after `downloading it <https://www.tensorflow.org/install/lang_c>`_ and unpacking it you will need to update your PATH or tell CMake where you placed the library with ``-DCMAKE_PREFIX_PATH=...``.

.. code:: bash

    mkdir -p /path/to/mydir/
    tar -C /path/to/mydir -xzf (downloaded file)

Now, update your path:

.. code:: bash

    export LIBRARY_PATH=$LIBRARY_PATH:/path/to/mydir/lib
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/mydir/lib


Install cppflow
-----------------------------

Cppflow is just a header-only library, and thus it does not require to build it. To facilitate the installation, we provide a CMake file that will install the library in your system. To install it, you just have to:

.. code:: bash

    mkdir build
    cd build
    cmake ..
    make -j
    make install

.. note::
    If you installed the TF C API in a custom directory, you will need to tell CMake where you placed the library with ``-DCMAKE_PREFIX_PATH=/path/to/mydir/``.

This will also compile the examples, if you don't want to compile them, you can use ``-DBUILD_EXAMPLES=OFF``.


You are done, now you can proceed to build your :ref:`first example<First example>`.
