import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# 设置 torch 训练的 随机数种子
torch.manual_seed(42)

def form_tensors():
    df_rtni = pd.read_csv('./cache/ForTrain/df_RtnI.csv', low_memory=False)
    df_rtnj = pd.read_csv('./cache/ForTrain/df_RtnJ.csv', low_memory=False)

    # convert to tensors
    rtni = torch.tensor(df_rtni.drop(['yyyymm'], axis=1).values, dtype=torch.float32)

    rtnj = []
    for yyyymm in df_rtnj['yyyymm'].unique():
        rtnj.append(df_rtnj.loc[df_rtnj['yyyymm'] == yyyymm].sort_values('permno')['RET'].tolist())
    rtnj = torch.tensor(rtnj, dtype=torch.float32)
    print(rtnj.shape)

    # 制造截面数据
    X = []
    for yyyymm in df_rtnj['yyyymm'].unique():
        cross_section_table = df_rtnj[df_rtnj['yyyymm'] == yyyymm].sort_values('permno').drop(['yyyymm','permno','RET'], axis=1).values
        X.append(cross_section_table)
    
    X = torch.tensor(np.array(X), dtype=torch.float32)

    # 保存 tensor 到本地
    torch.save(rtni, './cache/ForTrain/rtni.pt')
    torch.save(rtnj, './cache/ForTrain/rtnj.pt')
    torch.save(X, './cache/ForTrain/X.pt')

class PricingKernel(nn.Module):
    def __init__(self):
        super().__init__()
        # 卷积层1：输入通道30，输出通道64，卷积核在高度方向为1，宽度方向设为3（可调整）
        self.conv1 = nn.Conv2d(in_channels=30, out_channels=64, kernel_size=(1,3), padding=(0,1))
        self.relu = nn.ReLU()
        # 全局平均池化：将每个通道的整个空间（1x923）池化为一个值
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # 输出每个通道为1x1
        # 注意：经过全局池化后，形状变为 [batch, 64, 1, 1]，然后我们需要将其展平，再通过一个全连接层得到30维输出
        self.fc = nn.Linear(64, 30)  # 将64个通道的特征映射到30个因子

    def forward(self, x):
        # x 形状: [batch, 30, 1, 923]
        x = self.conv1(x)  # 输出: [batch, 64, 1, 923] (因为padding=(0,1)保持宽度)
        x = self.relu(x)
        x = self.global_pool(x)  # 输出: [batch, 64, 1, 1]
        x = x.view(x.size(0), -1)  # 展平: [batch, 64]
        x = self.fc(x)  # 输出: [batch, 30]
        return x

class TestingPortfolioGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        # 特征提取卷积层
        self.conv1 = nn.Conv2d(in_channels=30, out_channels=64, kernel_size=(1,3), padding=(0,1))
        self.relu1 = nn.ReLU()
        
        # 进一步特征提取
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1,3), padding=(0,1))
        self.relu2 = nn.ReLU()
        
        # 输出层：输出每只股票的权重
        self.conv_out = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=(1,1))
        
    def forward(self, x):
        # x 形状: [batch, 30, 1, 923]
        x = self.conv1(x)     # 输出: [batch, 64, 1, 923]
        x = self.relu1(x)
        
        x = self.conv2(x)     # 输出: [batch, 128, 1, 923]
        x = self.relu2(x)
        
        x = self.conv_out(x)  # 输出: [batch, 1, 1, 923]
        x = x.squeeze(1).squeeze(1)  # 压缩维度: [batch, 923]
        
        # 使用 softmax 确保每行加和为1
        x = F.softmax(x, dim=1)  # 在股票维度上应用softmax
        
        return x

def adversarial_training_step(inputs, rtni_batch, rtnj_batch, pricing_model, pricing_optimizer):
    """
    对抗训练步骤：训练定价核，最小化定价误差的平方和的均值
    """
    pricing_optimizer.zero_grad()
    
    # 1. 通过定价核网络得到权重向量 omega
    omega = pricing_model(inputs)  # [batch_size, 30]
    
    # 2. 计算随机贴现因子 (SDF): M_{t+1} = 1 - omega^T * R_{t+1}^e
    sdf = 1 - (omega * rtni_batch).sum(dim=1)  # [batch_size]
    
    # 3. 计算定价误差: pricing_error = sdf * rtnj
    sdf_expanded = sdf.unsqueeze(1).expand(-1, 923)  # [batch_size, 923]
    pricing_error = sdf_expanded * rtnj_batch  # [batch_size, 923]
    
    # 4. 损失函数：最小化定价误差的平方和的均值
    pricing_loss = torch.mean(pricing_error ** 2)
    
    # 5. 反向传播
    pricing_loss.backward()
    pricing_optimizer.step()
    
    return pricing_loss, pricing_error

def generator_training_step(inputs, rtni_batch, rtnj_batch, pricing_model, portfolio_model, portfolio_optimizer):
    """
    生成训练步骤：训练测试组合生成器，最大化加权定价误差的平方和的均值
    """
    portfolio_optimizer.zero_grad()
    
    # 1. 使用已训练的定价核（不更新梯度）
    with torch.no_grad():
        omega = pricing_model(inputs)
        sdf = 1 - (omega * rtni_batch).sum(dim=1)
        sdf_expanded = sdf.unsqueeze(1).expand(-1, 923)
        pricing_error = sdf_expanded * rtnj_batch
    
    # 2. 训练测试组合生成器
    test_weights = portfolio_model(inputs)  # [batch_size, 923]
    
    # 3. 计算加权定价误差
    weighted_pricing_error = pricing_error * test_weights  # [batch_size, 923]
    
    # 4. 损失函数：最大化加权定价误差的平方和的均值（使用负号）
    portfolio_loss = -torch.mean(weighted_pricing_error ** 2)
    
    # 5. 反向传播
    portfolio_loss.backward()
    portfolio_optimizer.step()
    
    return portfolio_loss, weighted_pricing_error

if __name__ == "__main__":
    # form_tensors()

    # 读取 tensor 并全部转化为 torch.float32
    rtni = torch.load('./cache/ForTrain/rtni.pt'); rtni = rtni.type(torch.float32)
    rtnj = torch.load('./cache/ForTrain/rtnj.pt'); rtnj = rtnj.type(torch.float32)
    X = torch.load('./cache/ForTrain/X.pt'); X = X.type(torch.float32)
    
    # 转化 X 为可以训练的格式 
    # 调整维度：将因子维度移到第1维（通道维），股票维度移到第2维，并增加一个高度维度（值为1）
    # 步骤: (T, 923, 30) -> (T, 30, 923) -> (T, 30, 1, 923)
    X = X.permute(0, 2, 1)  # 维度重排: (T, 30, 923)
    X = X.unsqueeze(2)      # 在位置2插入新维度: (T, 30, 1, 923)

    # 创建数据集（无监督情况，只有数据）
    dataset = TensorDataset(X, rtni, rtnj)  # 如果只有数据，没有标签，这样创建
    # 或者如果有标签，假设labels_tensor形状为(T, ...)，则 dataset = TensorDataset(data_tensor, labels_tensor)

    # 创建数据加载器
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 初始化网络
    pricing_model = PricingKernel()
    portfolio_model = TestingPortfolioGenerator()

    # 创建优化器
    pricing_optimizer = optim.Adam(pricing_model.parameters(), lr=0.001)
    portfolio_optimizer = optim.Adam(portfolio_model.parameters(), lr=0.001)

    avg_pricing_losses = []; avg_portfolio_losses = []

    num_epoches = 1000
    for epoch in range(num_epoches):
        total_pricing_loss = 0
        total_portfolio_loss = 0

        for batch_idx, batch in enumerate(dataloader):
            inputs = batch[0]  # 形状: [batch_size, 30, 1, 923]
            rtni_batch = batch[1]  # 形状: [batch_size, 30]
            rtnj_batch = batch[2]  # 形状: [batch_size, 923]

            # 第一步 对抗训练 - 最小化定价误差的平方和的均值
            pricing_loss, pricing_error = adversarial_training_step(
                inputs, rtni_batch, rtnj_batch, pricing_model, pricing_optimizer
            )
            
            # 第二步 生成训练 - 最大化加权定价误差的平方和的均值
            portfolio_loss, weighted_pricing_error = generator_training_step(
                inputs, rtni_batch, rtnj_batch, pricing_model, portfolio_model, portfolio_optimizer
            )
            
            total_pricing_loss += pricing_loss.item()
            total_portfolio_loss += portfolio_loss.item()

        # 每个 epoch 结束后打印平均损失
        avg_pricing_loss = total_pricing_loss / len(dataloader)
        avg_portfolio_loss = total_portfolio_loss / len(dataloader)
        print(f'Epoch {epoch}/1000:')
        print(f'Average Pricing Loss: {avg_pricing_loss:.10f}', end='\t')
        print(f'Average Portfolio Loss: {avg_portfolio_loss:.10f}')
        # 记录平均损失
        avg_pricing_losses.append(avg_pricing_loss); avg_portfolio_losses.append(avg_portfolio_loss)

        """
        if epoch % 10 == 0:
            torch.save(pricing_model.state_dict(), f'./cache/models/pricing_model_epoch_{epoch}.pt')
            torch.save(portfolio_model.state_dict(), f'./cache/models/portfolio_model_epoch_{epoch}.pt')
        """
        # 每10个 epoch 保存一次模型

        # 加入早停机制 如果 pricing_loss 在连续 5 个 epoch 中没有明显下降，则停止训练
        if (epoch > 5) and (abs(avg_pricing_loss - np.mean(avg_pricing_losses[-5:])) <= 1e-5):
            print(f'Early stopping at epoch {epoch} due to no improvement in pricing loss.')
            print(f'Final Average Pricing Loss Differences: {avg_pricing_loss - np.mean(avg_pricing_losses[-5:]):.10f}')

            """
            # 保存最终模型
            torch.save(pricing_model.state_dict(), './cache/models/final_pricing_model.pt')
            torch.save(portfolio_model.state_dict(), './cache/models/final_portfolio_model.pt')
            """
            break
    
    """
    # 保存平均损失
    np.save('./cache/avg_losses_series.npy', np.array([avg_pricing_losses, avg_portfolio_losses]))
    """
