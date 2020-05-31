import tensorflow as tf

def create_serialized_options(fraction, growth):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=fraction, allow_growth=growth)
    config = tf.ConfigProto(gpu_options=gpu_options)
    serialized = config.SerializeToString()
    return '{' + ','.join(list(map(hex, serialized))) + '}'

if __name__ == "__main__":
    print("Create serialized options which allow TF to use a certain percentage of GPU memory and allow TF to expand this memory if required.")
    for i in range(1, 10):
        memory_fraction_to_use = 0.1
        enable_memory_growth = True
        print("GPU memory to be used: ", i * 10.0, "%")
        print(create_serialized_options(memory_fraction_to_use, enable_memory_growth))
