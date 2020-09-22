import tensorflow as tf

if int(tf.__version__.split('.')[0]) < 2:
    raise RuntimeError("Need tensorflow 2.0")

class Test(tf.Module):
    def __init__(self):
        super(Test, self).__init__()
        self.var1 = tf.Variable(1.0, name="var1")
    
    @tf.function(input_signature=[tf.TensorSpec((), tf.float32), tf.TensorSpec((), tf.float32)])
    def __call__(self, val1, val2):
        result1 = self.var1 + val1 + 2*val2
        result2 = self.var1 + val1 + 4*val2
        return {'result1':result1, 'result2':result2}


if __name__ == "__main__":
    model_dir = 'model'
    model = Test()
    print(model(3.0, 10.0))
    tf.saved_model.save(model, export_dir=model_dir)
