import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    """
    生成器网络，用于生成测试资产的权重 g(I_t, I_{t,j})
    输入: 信息集 I_t 和资产特定信息 I_{t,j}
    输出: 测试资产权重
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # RNN层用于捕获时间序列相关性
        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()  # 确保输出在[0,1]区间
        )

    def forward(self, x):
        """
        前向传播
        Args:
            x: 形状为 (batch_size, seq_len, input_dim) 的输入
        Returns:
            权重 g(I_t, I_{t,j})
        """
        # RNN处理时间序列
        rnn_out, (hidden, cell) = self.rnn(x)
        
        # 使用最后一个时间步的输出
        last_output = rnn_out[:, -1, :]
        
        # 通过全连接层生成权重
        weights = self.fc_layers(last_output)
        
        return weights


class Discriminator(nn.Module):
    """
    判别器网络 用于学习SDF权重 omega(I_t, I_{t,i})
    输入: 信息集 I_t 和资产特定信息 I_{t,i}
    输出: SDF权重
    """
    def __init__(self, input_dim, num_assets, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.num_assets = num_assets
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # RNN层用于捕获时间序列相关性
        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 全连接层用于生成每个资产的权重
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_assets),
            nn.Softmax(dim=-1)  # 权重和为1
        )

    def forward(self, x):
        """
        前向传播
        Args:
            x: 形状为 (batch_size, seq_len, input_dim) 的输入
        Returns:
            SDF权重 ω(I_t, I_{t,i})
        """
        # RNN处理时间序列
        rnn_out, (hidden, cell) = self.rnn(x)
        
        # 使用最后一个时间步的输出
        last_output = rnn_out[:, -1, :]
        
        # 通过全连接层生成权重
        omega = self.fc_layers(last_output)
        
        return omega


class AssetPricingGAN(nn.Module):
    """
    完整的资产定价GAN模型
    """
    def __init__(self, factor_dim, num_assets, gen_hidden_dim=64, disc_hidden_dim=64, 
                 num_layers=2, dropout=0.2):
        super().__init__()
        self.num_assets = num_assets
        
        # 生成器和判别器
        self.generator = Generator(
            input_dim=factor_dim,
            hidden_dim=gen_hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
        self.discriminator = Discriminator(
            input_dim=factor_dim,
            num_assets=num_assets,
            hidden_dim=disc_hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )

    def compute_sdf(self, omega, excess_returns):
        """
        计算随机贴现因子 M_{t+1} = 1 - ω^T R^e_{t+1}
        Args:
            omega: SDF权重 (batch_size, num_assets)
            excess_returns: 超额收益 (batch_size, num_assets)
        Returns:
            SDF值
        """
        # 计算加权超额收益
        weighted_returns = torch.sum(omega * excess_returns, dim=-1, keepdim=True)
        # SDF = 1 - 加权超额收益
        sdf = 1.0 - weighted_returns
        return sdf

    def compute_loss(self, factors, excess_returns_pricing, excess_returns_testing, 
                    fix_generator=False, fix_discriminator=False):
        """
        计算GAN损失函数
        Args:
            factors: 因子数据 (batch_size, seq_len, factor_dim)
            excess_returns_pricing: 定价资产超额收益 (batch_size, num_assets)
            excess_returns_testing: 测试资产超额收益 (batch_size, num_assets)
            fix_generator: 是否固定生成器
            fix_discriminator: 是否固定判别器
        Returns:
            损失值
        """
        # 生成器产生测试权重
        if fix_generator:
            with torch.no_grad():
                g_weights = self.generator(factors)
        else:
            g_weights = self.generator(factors)
        
        # 判别器产生SDF权重
        if fix_discriminator:
            with torch.no_grad():
                omega = self.discriminator(factors)
        else:
            omega = self.discriminator(factors)
        
        # 计算SDF
        sdf = self.compute_sdf(omega, excess_returns_pricing)
        
        # 计算定价误差: M_{t+1} * R^e_{t+1,j} * g(I_t, I_{t,j})
        pricing_errors = sdf * excess_returns_testing * g_weights
        
        # 计算损失: 平均平方定价误差
        loss = torch.mean(pricing_errors ** 2)
        
        return loss

    def forward(self, factors):
        """
        前向传播
        Args:
            factors: 因子数据 (batch_size, seq_len, factor_dim)
        Returns:
            生成器权重和判别器权重
        """
        g_weights = self.generator(factors)
        omega = self.discriminator(factors)
        return g_weights, omega


# 训练函数
def train_gan(model, dataloader, num_epochs=100, lr_g=0.001, lr_d=0.001):
    """
    训练GAN模型
    """
    # 优化器
    optimizer_g = torch.optim.Adam(model.generator.parameters(), lr=lr_g)
    optimizer_d = torch.optim.Adam(model.discriminator.parameters(), lr=lr_d)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss_d = 0
        total_loss_g = 0
        
        for batch_idx, (factors, excess_returns_pricing, excess_returns_testing) in enumerate(dataloader):
            factors = factors.to(device)
            excess_returns_pricing = excess_returns_pricing.to(device)
            excess_returns_testing = excess_returns_testing.to(device)
            
            # 训练判别器 (最小化损失)
            optimizer_d.zero_grad()
            loss_d = model.compute_loss(
                factors, excess_returns_pricing, excess_returns_testing, 
                fix_generator=True
            )
            loss_d.backward()
            optimizer_d.step()
            total_loss_d += loss_d.item()
            
            # 训练生成器 (最大化损失)
            optimizer_g.zero_grad()
            loss_g = -model.compute_loss(  # 注意负号，生成器要最大化损失
                factors, excess_returns_pricing, excess_returns_testing,
                fix_discriminator=True
            )
            loss_g.backward()
            optimizer_g.step()
            total_loss_g += loss_g.item()
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}: D_loss={total_loss_d/len(dataloader):.6f}, '
                  f'G_loss={total_loss_g/len(dataloader):.6f}')
    
    return model
