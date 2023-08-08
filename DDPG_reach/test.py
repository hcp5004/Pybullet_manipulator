import tensorflow as tf

import tensorflow as tf

 

# Define a 2x2 array of 1's

x = tf.ones((2,2))
with tf.GradientTape() as t:
    # Record the actions performed on tensor x with `watch
    t.watch(x) 
    # Define y as the sum of the elements in x
    y =  tf.reduce_sum(x)
    # Let z be the square of y
    z = tf.square(y) 
# Get the derivative of z wrt the original input tensor x
dz_dx = t.gradient(z, x)
# Print our result
print(dz_dx)