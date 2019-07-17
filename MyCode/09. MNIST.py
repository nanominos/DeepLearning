# MNIST 데이터셋을 신경망으로 학습시키는 방법에 대해 연습
# MNIST는 손으로 쓴 숫자들의 이미지를 모아놓은 데이터 셋으로 0~9까지 28x28 픽셀 크기의 이미지로 구성해놓은 것.
# http://yann.lecun.com/exdb/mnist  에서 데이터 셋 다운로드

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

# tensorflow에서 기본으로 제공하는 mnist 데이터 셋을 불러온다.

X = tf.placeholder(tf.float32, [None, 28 * 28])
Y = tf.placeholder(tf.float32, [None, 10])

# MNIST는 28*28인 784 픽셀로 이루어져 있는데 이는 784가지의 특징이 있다고 볼 수 있다.
# 또한 0~9까지 10개의 숫자가 있으니 10개 분류로 나눌 수 있다.
# 이미지 여러개를 한번에 학습시키는쪽이 효과가 좋지만 그만큼 메모리와 높은 컴퓨팅 성능이 뒷받침 되어야하기 때문에 나눠서 학습시킨다.
# 이를 미니배치(Minibatch)라고 한다.
# placeholder에 None으로 되어있는 부분은 한번에 학습시킬 MNIST의 이미지 개수를 지정하는 값이 들어가야 한다.
# 한번에 학습할 개수를 계속 바꿔가면서 실험해보려는 경우 None을 넣어주면 Tensorflow가 알아서 계산한다.

# 2개의 은닉층이 다음처럼 구성된 신경망을 만든다.
'''
784(입력, 특징 개수)  -->  256(첫번째 은닉층 뉴런 개수)  --> 256(두번째 은닉층 뉴런 개수)  --> 10 (결과값 0~9 분류 개수)
'''

W1 = tf.Variable(tf.random_normal([784, 256], stddev=0.01))
L1 = tf.nn.relu(tf.matmul(X, W1))

W2 = tf.Variable(tf.random_normal([256, 256], stddev=0.01))
L2 = tf.nn.relu(tf.matmul(L1, W2))

W3 = tf.Variable(tf.random_normal([256, 10], stddev=0.01))
model = tf.matmul(L2, W3)

# 간략히 작성하기 위해 bias 제외했다. 원한다면 따로 추가해보자.
# 표준편차가 0.01인 정규분포를 가지는 임의의 값으로 뉴런을 초기화시킨다.
# 출력층에는 보통 활성화 함수를 사용하지 않는다.

cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=model))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


#
#  테스트용 데이터와 학습용 데이터를 따로 구분하는 이유?
#                                               ㄴ---> Overfitting!
'''
    머신러닝을 위한 학습 데이터는 항상 학습용과 테스트용으로 분리해서 사용해야 한다.
    학습 데이터는 학습을 시킬 때 사용하고,
    테스트 데이터는 학습이 잘 되었는지를 확인하는 데 사용한다.
    별도의 테스트 데이터를 사용하는 이유는 학습 데이터로 예측을 하면 예측 정확도가 매우 높게 나오지만,
    학습 데이터에 포함되지 않은 새로운 데이터를 예측할 때는 정확도가 매우 떨어지는 경우가 많기 때문이다.
    이처럼 학습 데이터는 예측을 매우 잘 하지만, 실제 데이터는 그렇지 못한 상태를 과적합(Overfitting)이라고 한다.
    
    이러한 현상을 확인하고 방지하기 위해 학습 데이터와 테스트 데이터를 분리하고,
    학습이 끝나면 항상 테스트 데이터를 사용하여 학습 결과를 검증해야 한다.
    MNIST의 경우 학습 데이터 6만개 테스트 데이터 1만개로 구성되어있다.
    Tensorflow를 사용한다면, mnist.train , mnist.test 로 사용할 수 있다.
'''
print("W3", W3)
print("MODEL", model)
print("MODEL RUN", sess.run(model, feed_dict={X: mnist.test.images, Y: mnist.test.labels})[0])

batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size)
# MNIST는 데이터 수가 수만개로 매우 크므로 학습에 미니배치를 사용할 예정이다.
# 미니배치의 크기를 100개로 설정하고, 학습 데이터의 총 개수를 배치 크기로 나눠 미니배치가 총 몇개인지를 저장해둔다.

for epoch in range(15):
    total_cost = 0
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys})
        total_cost += cost_val
    print("Epoch : ", "%04d" % (epoch + 1),
          "Avg. Cost = ", "{:.3f}".format(total_cost / total_batch))
print("최적화 완료!")

# MNIST 데이터 전체를 15번 학습한다. ( 학습 데이터 전체를 1바퀴 도는것을 에포크(epoch)라고 한다.
# 그 안에서 미니배치의 총 개수만큼 반복하여 학습한다.
# mnist.train.next_batch 함수를 이용해 학습할 데이터를 배치 크기만큼 가져온 뒤,
# 입력값인 이미지 데이터는 batch_xs에, 출력값인 레이블 데이터는 batch_ys에 저장한다.

is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
# 학습 결과 비교
# 예측한 결과값은 원-핫 인코딩 형식으로 각 argmax를 통해 가장 큰 값의 인덱스를 뱉는다.
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print("ISCORRECT", is_correct)
print("ISCORRECT RUN", sess.run(is_correct, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))
print("정확도 : ", sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))



