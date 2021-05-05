# ![cppflow](docs/source/cppflow.svg)

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

You can take a look to the [examples](https://github.com/serizba/cppflow/tree/master/examples/) to see a full example on how to load a deep network and feed it with a sample image.

CppFlow uses [Tensorflow C API](https://www.tensorflow.org/install/lang_c) to run the models, meaning you can use it without installing Tensorflow and without compiling the whole Tensorflow repository with bazel, you just need to download the C API. With this project you can manage and run your models in C++ without worrying about void, malloc or free. With CppFlow you easily can:

* Open saved models created with Python
* Execute Tensorflow neural networks in C++
* Perform tensor manipulation directly from C++

## How To Run It

Since it uses TensorFlow 2 C API you just have to [download it](https://www.tensorflow.org/install/lang_c), check the [docs](https://serizba.github.io/cppflow/installation.html) to see a guide on how to do it.  

You can either install the library system wide or you can just place the contents of the archive in a folder called `libtensorflow2` in your HOME directory.

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


## Documentation

Check the docs at [https://serizba.github.io/cppflow/](https://serizba.github.io/cppflow/).

There you can find quickstart guides and more information about how to install the library and run the examples.

## Development

CppFlow is basically a wrapper over Tensorflow C API. The basic class, [tensor](https://github.com/serizba/cppflow/blob/master/include/cppflow/tensor.h) is a wrapper of a TF eager tensor, and it just constains a pointer to its TF representation. 

The TF C API provides the tools to call all the TF [raw ops](https://www.tensorflow.org/api_docs/python/tf/raw_ops), but using them is confusing. CppFlow includes a facade over these functions, so they can be called easily as normal C++ functions. To achieve this, the file [ops](https://github.com/serizba/cppflow/blob/master/include/cppflow/raw_ops.h) contains (mostly) all the TF raw ops functions, but with a simple C++ interface. This file has been generated automatically using a [small script](https://github.com/serizba/cppflow/blob/master/include/cppflow/ops_generator/generator.py).

CppFlow also includes a wrapper on TF saved models, the [model](https://github.com/serizba/cppflow/blob/master/include/cppflow/model.h) class, so they can be easily opened and executed.

There are still many things to implement... some of them may be:


* Model complex invoking
* Model eager API: Calling model with the eager API instead of the TF_SessionRun API. I have tried using TF_GraphToFunction but I could not achieve it.
* Cover more raw_ops: Currently, the generator that creates the raw_ops facade converts many of the raw_ops but not all of them. Improve the generator to cover these cases (which are marked in the generator code).
* Include testing

## Cppflow 1

You can also use the [older version](https://github.com/serizba/cppflow/tree/243ff2fc4e33632b91676cad7d6cfc3c92308601) of this work.

## Remark

CppFlow is not related with TensorFlow. The CppFlow icon is a modified version of the TensorFlow logo. TensorFlow, the TensorFlow logo and any related marks are trademarks of Google Inc.