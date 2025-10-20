import scipy.io as sio
import matplotlib.pyplot as plt

# 修改为你的实际路径
data = sio.loadmat(r"/workspace/trained_outputs/train/train_loss.mat")
train_loss = data['train_loss'].squeeze()

plt.figure()
plt.plot(train_loss, label='Train Loss')
plt.title('Training Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True)
plt.savefig("train_loss.png")
print("✅ 图像已保存为 train_loss.png")
