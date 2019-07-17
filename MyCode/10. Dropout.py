# 과적합 문제를 해결하기 위해 효과가 좋은 방법 중 하나인 드롭아웃을 적용한다.
# 드롭아웃의 원리는 매우 간단한데, 학습 시 전체 신경망 중 일부만을 사용하도록 하는 것이다.
# 즉, 학습 단계마다 일부 뉴런을 제거(사용하지 않도록) 함으로써 일부 특징이 특정 뉴런들에 고정되는 것을 막아,
# 가중치의 균형을 잡도록 하여 과적합을 방지한다.
# 다만, 학습 시 일부 뉴런을 학습시키지 않기 때문에 신경망이 충분히 학습되기까지의 시간은 조금 더 오래걸리는 편이다.


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

X = tf.placeholder(tf.float32, [None, 28 * 28])
Y = tf.placeholder(tf.float32, [None, 10])

keep_prob = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_normal([784, 256], stddev=0.01))
L1 = tf.nn.relu(tf.matmul(X, W1))
#L1 = tf.nn.dropout(L1, 0.8)
L1 = tf.nn.dropout(L1, keep_prob)
# 09. MNIST.py 코드에 tf.nn.dropout 함수를 적용시키면 된다.
# dropout에 0.8은 80%만큼 뉴런을 활성화 시키겠다는 의미이다.
# 주의할점은 학습이 끝난 뒤 예측할 땐 신경망 전체를 사용하도록 설정해줘야 한다.
# 따라서 keep_prob라는 플레이스 홀더를 만들어, 학습 시에는 0.8을 넣어 드롭아웃을 적용하고,
# 예측 시에는 1을 넣어 신경망 전체를 사용하도록 만들어야 한다.


W2 = tf.Variable(tf.random_normal([256, 256], stddev=0.01))
L2 = tf.nn.relu(tf.matmul(L1, W2))
#L2 = tf.nn.dropout(L2, 0.8)
L2 = tf.nn.dropout(L2, keep_prob)

W3 = tf.Variable(tf.random_normal([256, 10], stddev=0.01))
model = tf.matmul(L2, W3)


cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=model))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size)

for epoch in range(30):         # dropout을 적용했으므로 학습 세대를 30회로 늘려준다.
    total_cost = 0
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 0.8})
        # keep_prob를 0.8로 두어 80%의 뉴런으로 학습시킨다.
        total_cost += cost_val
    print("Epoch : ", "%04d" % (epoch + 1),
          "Avg. Cost = ", "{:.3f}".format(total_cost / total_batch))
print("최적화 완료!")

is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print("정확도 : ", sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1.0}))


