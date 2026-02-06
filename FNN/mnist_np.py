import numpy as np
import gzip

def load_mnist(images_path, labels_path):
    """读取MNIST图像和标签"""
    # 读取图像
    with gzip.open(images_path, 'rb') as f:
        magic, num_images, rows, cols = np.frombuffer(f.read(16), dtype='>i4')
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num_images, rows, cols)
    
    # 读取标签
    with gzip.open(labels_path, 'rb') as f:
        magic, num_labels = np.frombuffer(f.read(8), dtype='>i4')
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    
    return images, labels

class FNN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        # 初始化参数，使用较小的随机值
        self.weight_input_hidden = np.random.randn(input_size, hidden_size) * 0.01
        self.weight_hidden_output = np.random.randn(hidden_size, output_size) * 0.01
        self.bias_hidden = np.zeros((1, hidden_size))
        self.bias_output = np.zeros((1, output_size))
        self.learning_rate = learning_rate
        
    def sigmoid(self, x):
        x = np.clip(x, -50, 50)
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative_from_sigmoid(self, s):
        return s * (1 - s)
    
    def softmax(self, x):
        # 减去最大值提高数值稳定性
        x_max = np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def forward(self, input):
        self.input = input
        # hidden层计算
        self.hidden_layer_input = np.dot(self.input, self.weight_input_hidden) + self.bias_hidden
        self.hidden_layer_output = self.sigmoid(self.hidden_layer_input)
        # output层计算
        self.output_layer_input = np.dot(self.hidden_layer_output, self.weight_hidden_output) + self.bias_output
        self.prediction = self.softmax(self.output_layer_input)
        return self.prediction
    
    def backward(self, prediction, label):
        # 计算交叉熵损失
        epsilon = 1e-8  # 防止log(0)
        loss = -np.sum(label * np.log(prediction + epsilon))
        
        # 输出层梯度
        gradient_output_layer = prediction - label
        
        # 计算hidden层到output层的梯度
        d_weight_hidden_output = np.dot(self.hidden_layer_output.T, gradient_output_layer)
        d_bias_output = np.sum(gradient_output_layer, axis=0, keepdims=True)
        
        # hidden层梯度（关键修正：需要传递输出层的梯度）
        gradient_hidden_layer_input = np.dot(gradient_output_layer, self.weight_hidden_output.T)
        gradient_hidden_layer = gradient_hidden_layer_input * self.sigmoid_derivative_from_sigmoid(self.hidden_layer_output)
        
        # 计算input层到hidden层的梯度
        d_weight_input_hidden = np.dot(self.input.T, gradient_hidden_layer)
        d_bias_hidden = np.sum(gradient_hidden_layer, axis=0, keepdims=True)
        
        # 更新参数（使用学习率）
        self.weight_input_hidden -= self.learning_rate * d_weight_input_hidden
        self.bias_hidden -= self.learning_rate * d_bias_hidden
        self.weight_hidden_output -= self.learning_rate * d_weight_hidden_output
        self.bias_output -= self.learning_rate * d_bias_output
        
        return loss

# 主程序
model = FNN(784, 100, 10, learning_rate=0.1)

# 加载数据
train_images, train_labels = load_mnist('../datasets/mnist/train-images-idx3-ubyte.gz', '../datasets/mnist/train-labels-idx1-ubyte.gz')

# 数据预处理
size = train_images.shape[0]
train_images = train_images.reshape(size, -1).astype(np.float32) / 255.0  # 归一化到[0,1]
train_labels = train_labels.reshape(size, -1)

# 训练
for epoch in range(5):  # 增加epoch循环
    total_loss = 0
    correct = 0
    
    for i in range(size):  # 先测试前1000个样本
        # 前向传播
        prediction = model.forward(train_images[i].reshape(1, 784))
        
        # 准备one-hot标签
        label = np.zeros((1, 10))
        label[0, train_labels[i][0]] = 1
        
        # 反向传播
        loss = model.backward(prediction, label)
        total_loss += loss
        
        # 计算准确率
        if np.argmax(prediction) == train_labels[i][0]:
            correct += 1
        
        if i % 1000 == 0:
            accuracy = correct / (i + 1) * 100
            print(f"Epoch {epoch}, Sample {i}: Loss={loss:.4f}, Accuracy={accuracy:.2f}%")
    
    avg_loss = total_loss / min(1000, size)
    accuracy = correct / min(1000, size) * 100
    print(f"\nEpoch {epoch} completed: Avg Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}%\n")