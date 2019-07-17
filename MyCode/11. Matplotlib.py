# 10. Dropout.py 에 Matplotlib을 적용한다.

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
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

labels = sess.run(model, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1.0})
# 테스트 데이터를 이용해 예측 모델을 실행하고 결과를 lables에 저장한다.
fig = plt.figure()
# 손글씨를 출력할 그래프를 준비

for i in range(10):
    #2행 5열의 그래프를 만들고, i + 1번째에 숫자 이미지를 출력
    subplot = fig.add_subplot(2, 5, i + 1)
    #x, y axis 출력 안함
    subplot.set_xticks([])
    subplot.set_yticks([])
    subplot.set_title('%d' % np.argmax(labels[i]))
    subplot.imshow(mnist.test.images[i].reshape((28, 28)))

fig.savefig('demo.png', bbox_inches='tight')
