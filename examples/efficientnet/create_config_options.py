'''Sample output with TF 2.5
GPU memory to be used:  10.0 %
{0x32,0xb,0x9,0x9a,0x99,0x99,0x99,0x99,0x99,0xb9,0x3f,0x20,0x1}

GPU memory to be used:  20.0 %
{0x32,0xb,0x9,0x9a,0x99,0x99,0x99,0x99,0x99,0xc9,0x3f,0x20,0x1}

GPU memory to be used:  30.000000000000004 %
{0x32,0xb,0x9,0x34,0x33,0x33,0x33,0x33,0x33,0xd3,0x3f,0x20,0x1}

GPU memory to be used:  40.0 %
{0x32,0xb,0x9,0x9a,0x99,0x99,0x99,0x99,0x99,0xd9,0x3f,0x20,0x1}

GPU memory to be used:  50.0 %
{0x32,0xb,0x9,0x0,0x0,0x0,0x0,0x0,0x0,0xe0,0x3f,0x20,0x1}

GPU memory to be used:  60.00000000000001 %
{0x32,0xb,0x9,0x34,0x33,0x33,0x33,0x33,0x33,0xe3,0x3f,0x20,0x1}

GPU memory to be used:  70.0 %
{0x32,0xb,0x9,0x67,0x66,0x66,0x66,0x66,0x66,0xe6,0x3f,0x20,0x1}

GPU memory to be used:  80.0 %
{0x32,0xb,0x9,0x9a,0x99,0x99,0x99,0x99,0x99,0xe9,0x3f,0x20,0x1}

GPU memory to be used:  90.0 %
{0x32,0xb,0x9,0xcd,0xcc,0xcc,0xcc,0xcc,0xcc,0xec,0x3f,0x20,0x1}
'''

import tensorflow.compat.v1 as tf

def create_serialized_options(fraction, growth):
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = fraction
    config.gpu_options.allow_growth = growth
    serialized = config.SerializeToString()
    return '{' + ','.join(list(map(hex, serialized))) + '}'

if __name__ == "__main__":
    print("Create serialized options which allow TF to use a certain percentage of GPU memory and allow TF to expand this memory if required.")
    memory_fraction_interval = 0.1
    for i in range(1, int(1/memory_fraction_interval)):
        memory_fraction_to_use = memory_fraction_interval * i
        enable_memory_growth = True
        print("GPU memory to be used: ", memory_fraction_to_use*100, "%")
        print(create_serialized_options(memory_fraction_to_use, enable_memory_growth))
        print()
