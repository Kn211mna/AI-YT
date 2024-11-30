import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()  # Використовуємо TensorFlow 1.x у середовищі 2.x

# Генеруємо дані для навчання
x_data = np.random.rand(1000).astype(np.float32)  # 1000 випадкових точок
y_data = 2.0 * x_data + 1.0 + np.random.normal(0, 2, x_data.shape)  # y = 2x + 1 + шум

# Оголошення placeholder для X та y
X = tf.placeholder(tf.float32, shape=[None, 1], name="X")  # Матриця розмірності (міні-батч × 1)
y = tf.placeholder(tf.float32, shape=[None], name="y")  # Вектор довжини розмір міні-батча

# Ініціалізація параметрів моделі
k = tf.Variable(tf.random.normal([1]), name="k")  # k ініціалізується нормальним розподілом
b = tf.Variable(tf.zeros([1]), name="b")  # b початково нульове

# Модель лінійної регресії
y_pred = tf.squeeze(tf.matmul(X, tf.reshape(k, [-1, 1])) + b)  # Передбачення моделі

# Функція втрат
loss = tf.reduce_mean(tf.square(y_pred - y))

# Оптимізатор
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train_op = optimizer.minimize(loss)

# Ініціалізація змінних
init = tf.global_variables_initializer()

# Розмір міні-батчу
batch_size = 100


# Функція для вибору міні-батчу
def get_batch(x, y, batch_size):
    indices = np.random.choice(len(x), batch_size)
    return x[indices].reshape(-1, 1), y[indices]


# Тренування моделі
with tf.Session() as sess:
    sess.run(init)

    for epoch in range(2001):
        x_batch, y_batch = get_batch(x_data, y_data, batch_size)
        _, loss_value, k_value, b_value = sess.run(
            [train_op, loss, k, b],
            feed_dict={X: x_batch, y: y_batch}
        )

        if epoch % 100 == 0:
            print(f"Епоха {epoch}: втрата={loss_value:.4f}, k={k_value[0]:.4f}, b={b_value[0]:.4f}")
