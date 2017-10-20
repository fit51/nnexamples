#%%
import numpy as np
import tensorflow as tf

#%%
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0) # float32 implicitly
print(node1, node2)
#%%
sess = tf.Session()
print(sess.run([node1, node2]))
#%%
node3 = tf.add(node1, node2)
sess.run(node3)

#%%
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b #shortcut for tf.add(a, b)
sess.run(adder_node, {a: 20, b: 3})
sess.run(adder_node, {a: [1, 2], b: [5, 6]})

#%%
W = tf.Variable([100], dtype=tf.float32)
b = tf.Variable([200], dtype=tf.float32)
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
linear = W*x + b
squared_deltas = tf.square(linear - y)
#loss
loss = tf.reduce_sum(squared_deltas)
#optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
sess.run(init)
#training set
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

sess.run(loss, {x: x_train, y: y_train})

for i in range(1000):
    sess.run(train, {x: x_train, y: y_train})
    if i%100 == 0:
        print(sess.run([W, b]))

#Evaluate accuracy
curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))


#soft max
#%%
y = np.array([[1, 2, 3, 4],
[2, 3, 4, 5],
[3, 4, 5, 6]])
#%%
x = np.random.randn(3, 4)
soft = tf.nn.softmax(x)
sess.run(soft)

#%%
sess.close()
"./notMNIST_large/".split('/')[1]