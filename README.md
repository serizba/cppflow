# CppFlow 2

This version is under development.

Run TensorFlow models in c++ without Bazel, without TensorFlow installation and without compiling Tensorflow. Perform tensor manipulation, use eager execution and run saved models directly from C++.

```c++
    // Read the graph
    cppflow::model model("saved_model_folder");
    
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
```

You can take a look to the [examples](https://github.com/serizba/cppflow/tree/cppflow2/examples) to see a full example on how to load a deep network and feed it with a sample image.

## Documentation

Check the docs at [https://serizba.github.io/cppflow/](https://serizba.github.io/cppflow/).

There you can find quickstart guides and more information about how to install the library and run the examples.

## Development

CppFlow is basically a wrapper over TensorFlow C API. The basic class, [tensor](https://github.com/serizba/cppflow/blob/cppflow2/include/cppflow/tensor.h) is a wrapper of a TF eager tensor, and it just constains a pointer to its TF representation. 

The TF C API provides the tools to call all the TF [raw ops](https://www.tensorflow.org/api_docs/python/tf/raw_ops), but using them is confusing. CppFlow includes a facade over these functions, so they can be called easily as normal C++ functions. To achieve this, the file [ops](https://github.com/serizba/cppflow/blob/cppflow2/include/cppflow/raw_ops.h) contains (mostly) all the TF raw ops functions, but with a simple C++ interface. This file has been generated automatically using a [small script](https://github.com/serizba/cppflow/blob/cppflow2/include/cppflow/ops_generator/generator.py).

CppFlow also includes a wrapper on TF saved models, the [model](https://github.com/serizba/cppflow/blob/cppflow2/include/cppflow/model.h) class, so they can be easily opened and executed.

As this is still a work under development, there are still many things to do... some of them may be:


  - Model complex invoking: Enable calling model with more than one input and to produce more than one output
  - Model eager API: Calling model with the eager API instead of the TF_SessionRun API. I have tried using TF_GraphToFunction but I could not achieve it.
  - Cover more raw_ops: Currently, the generator that creates the raw_ops facade converts many of the raw_ops but not all of them. Improve the generator to cover these cases (which are marked in the generator code).
  - Include testing

## How To Run It

Since it uses TensorFlow 2 C API you just have to [download it](https://www.tensorflow.org/install/lang_c).  

You can either install the library system wide by following the tutorial on the Tensorflow page or you can place the contents of the archive in a folder called `libtensorflow2` in the home directory.

Afterwards, you can run the examples:

```sh
git clone git@github.com:serizba/cppflow.git
cd cppflow/examples/load_model
mkdir build
cd build
cmake ..
make
./example
```
