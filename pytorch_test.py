import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import torch.utils.data as data

# df = pd.read_csv(r'classData.csv', usecols=[
#     'G', 'C', 'B', 'A', 'Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc'])
# my_tensor = torch.tensor(df.values, dtype=torch.float64)
# Output (S)	Ia	Ib	Ic	Va	Vb	Vc

class Tansig(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 2 / (1 + torch.exp(-2 * x)) - 1

class Logsig(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 1 / (1 + torch.exp(-x))

df = pd.read_excel(r'detect_dataset.xlsx', usecols=[
    'Output (S)', 'Ia', 'Ib', 'Ic','Va', 'Vb', 'Vc'])
my_tensor = torch.tensor(df.values, dtype=torch.float64)

features = my_tensor[:, 1:].float()  # 其他列是输出
labels = my_tensor[:, 0].float()  # 第一列是结果


dataset = data.TensorDataset(features,labels)
# 创建数据加载器
dataloader = data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
# for i, data in enumerate(dataloader):
#     # 获取输入数据和标签
#     inputs, labels = data

#     # 在此处执行模型训练或推理操作
#     # ...

#     print(f'Batch {i + 1}: inputs = {inputs}, labels = {labels}')
#     break

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.my_func = nn.Sequential(
            nn.Linear(6, 10),
            Tansig(),
            nn.Linear(10, 5),
            Tansig(),
            nn.Linear(5, 3),
            Logsig(),
            nn.Linear(3, 1)
        )

    def forward(self, x):
        ret = self.my_func(x)
        return ret

# 创建模型对象
model = MyNet()

# 定义损失函数
criterion = nn.MSELoss()

# 定义优化器，使用随机梯度下降法（SGD）
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 记录每个 epoch 的损失值
train_losses = []

# 循环迭代训练集
for epoch in range(100):
    running_loss = 0.0

    # 循环训练数据加载器，获取批次的输入和标签
    for i, data in enumerate(dataloader):
        inputs, labels = data

        # 将梯度清零
        optimizer.zero_grad()

        # 计算模型输出结果
        outputs = model(inputs)

        # 计算损失值
        loss = criterion(outputs, labels.unsqueeze(1))

        # 反向传播，计算梯度
        loss.backward()

        # 更新权重参数
        optimizer.step()

        # 累加损失值
        running_loss += loss.item()

    # 记录平均损失值
    train_loss = running_loss / len(dataloader)
    train_losses.append(train_loss)

    # 打印当前 epoch 的损失值
    print(f'Epoch {epoch + 1}, Loss: {train_loss:.4f}')

# 在验证集上测试模型性能
with torch.no_grad():
    # 创建验证数据加载器
    val_dataloader = data.DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=0)

    # 获取全部验证数据
    inputs, labels = next(iter(val_dataloader))

    # 计算模型在验证集上的输出结果
    outputs = model(inputs)

    # 计算损失值
    val_loss = criterion(outputs, labels.unsqueeze(1))

    # 打印验证集上的损失值和模型输出结果
    print(f'Validation Loss: {val_loss:.4f}')
    print(f'Model Outputs: {outputs.squeeze()}')
