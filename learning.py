import tensorflow as tf
import torch

# c = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
# print(tf.shape(c)[0])
# print(c.get_shape().as_list())

a = torch.Tensor([[1,2],[3,4]])
if 2:
    print(list(a.shape))