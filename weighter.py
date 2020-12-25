import tensorflow as tf
import numpy as np
import csv
import time

total_time = time.time()

RATE = 44100

chords = ['C','C#','D','Eb','E', \
    'F','F#','G','Ab','A','Bb','B', \
        'Cm','C#m','Dm','Ebm','Em', \
    'Fm','F#m','Gm','Abm','Am','Bbm','Bm']

# x, y 데이터(x는 feature, y는 label)

x_data = np.array([])
y_data = np.array([], dtype=np.int16)

print('data loading')
for i in range(24):
    load_time = time.time()
    print(chords[i]+' loaded...', end=' ')
    with open('data/'+chords[i]+'.csv', 'r') as f:
        rdr = csv.reader(f)
        for fft in rdr:
            x_data = np.append(x_data, list(map(float, fft)))
            y_data = np.append(y_data, i)
    print(np.round(time.time() - load_time, 2), 'sec')

x_data = x_data.reshape((-1, RATE//2-20))
y_data = np.eye(24)[y_data]

s = np.arange(x_data.shape[0])
np.random.shuffle(s)

x_data = x_data[s]
y_data = y_data[s]

print(x_data.shape[0], 'data confirmed')
# x와 y 데이터가 들어갈 형상 선언
layer0 = tf.compat.v1.placeholder(tf.float32, [None, RATE//2-20], name='in')

Y = tf.compat.v1.placeholder(tf.float32, [None, 24])

# 학습될 가중치와(W) 편향(b)
w1 = tf.Variable(tf.random.uniform([RATE//2-20, 24]))
b1 = tf.Variable(tf.zeros([24]))
layer1 = tf.sigmoid(tf.add(tf.matmul(layer0, w1), b1))  # 순전파 오퍼레이션

w2 = tf.Variable(tf.random.uniform([24, 24]))
b2 = tf.Variable(tf.zeros([24]))
layer2 = tf.nn.softmax(tf.sigmoid(tf.add(tf.matmul(layer1, w2), b2)))  # 순전파 오퍼레이션

layer2 = tf.identity(layer2, "out")

cost = tf.reduce_mean(-tf.reduce_sum(tf.math.log(layer2) * Y, axis = 1)) # 손실함수
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01) # 옵티마이저
train_op = optimizer.minimize(cost) # 학습 오퍼레이션

# 데이터를 가지고 학습 - 텐서플로우 세션(텐서플로우의 모든 연산은 세션 안에서 실행된다)
print('learning')
sess = tf.compat.v1.Session()
saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())

ckpt = tf.train.get_checkpoint_state('model/model.ckpt')
if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    saver.restore(sess, ckpt.model_checkpoint_path)
else:
    sess.run(tf.compat.v1.global_variables_initializer())

for step in range(100):
    load_time = time.time()
    print('step', step+1, '...', end='')
    sess.run(train_op, feed_dict={layer0: x_data, Y: y_data})
    print(np.round(time.time() - load_time, 2), 'sec')

# 검증
prediction = tf.argmax(layer2, axis=1) # 학습된 모델에 대한 예측값
real = tf.argmax(Y, axis=1) # 실제 값
is_correct = tf.equal(prediction, real)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('accuracy: %2f' % sess.run(accuracy*100, feed_dict={layer0: x_data, Y: y_data}))
print('total', np.round(time.time() - total_time, 2), 'sec')

saver.save(sess, 'model/model.ckpt')
