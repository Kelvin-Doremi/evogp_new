import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from evogp.algorithm import DefaultCrossover, DefaultMutation, TournamentSelection
from evogp.pipeline import Regressor
from evogp.tree import Forest, GenerateDescriptor
import pickle
from torch import optim
import os

# 设置设备（GPU如果可用，否则CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
if torch.cuda.is_available():
    print(f"GPU设备: {torch.cuda.get_device_name(0)}")

# ========== 开始蒸馏流程 ==========
print("========== 开始神经网络到GP模型的蒸馏流程 ==========")

# 加载保存的模型和数据信息
print("\n正在加载训练好的模型和数据...")
with open('models/data_info.pkl', 'rb') as f:
    data_info = pickle.load(f)

# 加载数据并转换为torch张量，确保内存连续
X_train = torch.FloatTensor(data_info['X_train']).to(device).contiguous()
X_val = torch.FloatTensor(data_info['X_val']).to(device).contiguous()
y_train = torch.FloatTensor(data_info['y_train']).to(device).contiguous()
y_val = torch.FloatTensor(data_info['y_val']).to(device).contiguous()
input_dim = data_info['input_dim']
final_val_loss = data_info['final_val_loss']

print(f"数据加载完成！训练集形状: {X_train.shape}, 验证集形状: {X_val.shape}")
print(f"原始神经网络验证MSE: {final_val_loss:.6f}")

# 加载模型
class CaliforniaModel(nn.Module):
    def __init__(self, input_dim):
        super(CaliforniaModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.ReLU(),

            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # 线性输出，用于回归
        )
        
    def forward(self, x):
        return self.network(x)

model = CaliforniaModel(input_dim=input_dim)
model.load_state_dict(torch.load('models/neural_network.pth', map_location=device))
model = model.to(device)
model.eval()
print("模型加载完成！")

# 创建数据加载器（数据已经在GPU上）
train_dataset = TensorDataset(X_train, y_train.view(-1, 1))
val_dataset = TensorDataset(X_val, y_val.view(-1, 1))
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)

# 提取前两层的结构
class FrontLayers(nn.Module):
    """前两层：Linear(input_dim, 128) + ReLU + Linear(128, 2) + ReLU"""
    def __init__(self, input_dim):
        super(FrontLayers, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.layers(x)

class BackLayers(nn.Module):
    """后两层：Linear(2, 128) + ReLU + Linear(128, 1)"""
    def __init__(self):
        super(BackLayers, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        return self.layers(x)

# 提取前两层和后两层
front_layers = FrontLayers(input_dim=input_dim).to(device)
back_layers = BackLayers().to(device)

# 加载训练好的权重
front_layers.layers[0].weight.data = model.network[0].weight.data.clone()
front_layers.layers[0].bias.data = model.network[0].bias.data.clone()
front_layers.layers[2].weight.data = model.network[2].weight.data.clone()
front_layers.layers[2].bias.data = model.network[2].bias.data.clone()

back_layers.layers[0].weight.data = model.network[4].weight.data.clone()
back_layers.layers[0].bias.data = model.network[4].bias.data.clone()
back_layers.layers[2].weight.data = model.network[6].weight.data.clone()
back_layers.layers[2].bias.data = model.network[6].bias.data.clone()

# 1. 提取前两层的输出作为蒸馏数据
print("\n步骤1: 提取前两层的输出数据...")
front_layers.eval()
front_outputs_train = []
front_outputs_val = []

with torch.no_grad():
    for batch_X, _ in train_loader:
        front_out = front_layers(batch_X)
        front_outputs_train.append(front_out)
    
    for batch_X, _ in val_loader:
        front_out = front_layers(batch_X)
        front_outputs_val.append(front_out)

front_outputs_train = torch.cat(front_outputs_train, dim=0).contiguous()
front_outputs_val = torch.cat(front_outputs_val, dim=0).contiguous()
print(f"前两层输出形状 - 训练集: {front_outputs_train.shape}, 验证集: {front_outputs_val.shape}")

# 2. 使用evogp蒸馏前两层为GP模型
print("\n步骤2: 使用evogp蒸馏前两层为GP模型...")
gp_front_models = []

for output_dim in range(front_outputs_train.shape[1]):
    print(f"  正在训练GP模型用于输出维度 {output_dim+1}/{front_outputs_train.shape[1]}...")
    
    # 准备SR任务的数据：输入是原始输入，输出是神经网络前两层的输出
    # 数据已经在GPU上，确保内存连续
    # y需要是二维的 (n_samples, 1)
    gp_train_X = X_train.contiguous()
    gp_train_y = front_outputs_train[:, output_dim].contiguous().view(-1, 1)
    gp_val_X = X_val.contiguous()
    gp_val_y = front_outputs_val[:, output_dim].contiguous()  # torch张量，用于MSE计算
    
    # 使用evogp进行符号回归

    descriptor = GenerateDescriptor(
        max_tree_len=128,
        input_len=input_dim,
        output_len=1,
        using_funcs=["+", "-", "*", "/", "sin", "cos", "tan"],
        max_layer_cnt=7,
        # const_range=[-5, 5],
        # sample_cnt=10000,
        const_samples=[-2, -1, 0, 1, 2],
        layer_leaf_prob=0.3,
    )

    gp_model = Regressor(
        initial_forest=Forest.random_generate(pop_size=1000, descriptor=descriptor),
        crossover=DefaultCrossover(),
        mutation=DefaultMutation(
            mutation_rate=0.1, descriptor=descriptor.update(max_layer_cnt=4)
        ),
        selection=TournamentSelection(
            tournament_size=20, survivor_rate=0.5, elite_rate=0.1
        ),
        generation_limit=100,
        optimize_constants=True,
        bfgs_top_k=20,
        bfgs_max_iter=100,
        bfgs_async=True,
        bfgs_start_gen=50,  # 前 50 代不做常数优化
    )
    gp_model.fit(gp_train_X, gp_train_y)
    
    # 评估GP模型
    gp_pred_train = gp_model.predict(gp_train_X)
    gp_pred_val = gp_model.predict(gp_val_X)
    # 使用torch计算MSE
    train_mse = torch.mean((gp_pred_train - gp_train_y) ** 2).item()
    val_mse = torch.mean((gp_pred_val - gp_val_y.view(-1, 1)) ** 2).item()
    print(f"    训练MSE: {train_mse:.6f}, 验证MSE: {val_mse:.6f}")
    
    gp_front_models.append(gp_model)

print("前两层GP模型蒸馏完成！")
    
import copy

# 3. 用GP模型替换前两层，用GP输出作为输入重训练后两层
print("\n步骤3: 使用GP模型输出微调后两层...")

# 使用GP模型生成新的输入数据
def gp_front_predict(X_data):
    """使用GP模型预测前两层的输出"""
    predictions = []
    for gp_model in gp_front_models:
        pred = gp_model.predict(X_data)
        # 确保pred是二维的 (n_samples, 1)
        if pred.dim() == 1:
            pred = pred.unsqueeze(1)
        predictions.append(pred)
    # 使用torch.cat沿着dim=1连接，得到 (n_samples, num_models)，确保内存连续
    output = torch.cat(predictions, dim=1).contiguous()
    # 原始网络前两层输出经过ReLU，因此必须非负
    return torch.relu(output)

# 生成新的训练和验证数据
gp_train_X_new = gp_front_predict(gp_train_X)
gp_val_X_new = gp_front_predict(gp_val_X)

print(f"GP模型输出的新输入数据形状 - 训练集: {gp_train_X_new.shape}, 验证集: {gp_val_X_new.shape}")

# gp_train_X_new 和 gp_val_X_new 已经是torch张量（在GPU上），直接使用
gp_train_dataset = TensorDataset(gp_train_X_new, y_train.view(-1, 1))
gp_val_dataset = TensorDataset(gp_val_X_new, y_val.view(-1, 1))
gp_train_loader = DataLoader(gp_train_dataset, batch_size=1024, shuffle=True)
gp_val_loader = DataLoader(gp_val_dataset, batch_size=1024, shuffle=False)

# 重训练后两层
print("开始微调后两层...")
# 使用深拷贝，基于原始权重进行微调，而不是随机初始化
back_layers_train = copy.deepcopy(back_layers)
criterion = nn.MSELoss()
# 使用较小的学习率进行微调
optimizer_back = optim.Adam(back_layers_train.parameters(), lr=1e-4)

retrain_epochs = 100
for epoch in range(retrain_epochs):
    back_layers_train.train()
    train_loss = 0.0
    for batch_X, batch_y in gp_train_loader:
        optimizer_back.zero_grad()
        outputs = back_layers_train(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer_back.step()
        train_loss += loss.item()
    
    back_layers_train.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in gp_val_loader:
            outputs = back_layers_train(batch_X)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()
    
    if (epoch + 1) % 20 == 0:
        print(f"  微调 Epoch [{epoch+1}/{retrain_epochs}], Train Loss: {train_loss/len(gp_train_loader):.4f}, Val Loss: {val_loss/len(gp_val_loader):.4f}")

print("后两层微调完成！")

# 4. 提取后两层的输出作为蒸馏数据
print("\n步骤4: 提取后两层的输出数据...")
back_layers_train.eval()
back_outputs_train = []
back_outputs_val = []

with torch.no_grad():
    for batch_X, _ in gp_train_loader:
        back_out = back_layers_train(batch_X)
        back_outputs_train.append(back_out)
    
    for batch_X, _ in gp_val_loader:
        back_out = back_layers_train(batch_X)
        back_outputs_val.append(back_out)

back_outputs_train = torch.cat(back_outputs_train, dim=0).contiguous()
back_outputs_val = torch.cat(back_outputs_val, dim=0).contiguous()
print(f"后两层输出形状 - 训练集: {back_outputs_train.shape}, 验证集: {back_outputs_val.shape}")

# 5. 使用evogp蒸馏后两层为GP模型
print("\n步骤5: 使用evogp蒸馏后两层为GP模型...")
# 后两层输出1维
gp_back_models = []

for output_dim in range(back_outputs_train.shape[1]):
    print(f"  正在训练GP模型用于输出维度 {output_dim+1}/{back_outputs_train.shape[1]}...")
    
    # 准备SR任务的数据：输入是GP前两层的输出（2维），输出是神经网络后两层的输出
    # 确保内存连续
    gp_train_X_back = gp_train_X_new.contiguous()
    gp_train_y_back = back_outputs_train[:, output_dim].contiguous().view(-1, 1)
    gp_val_X_back = gp_val_X_new.contiguous()
    gp_val_y_back = back_outputs_val[:, output_dim].contiguous()  # torch张量，用于MSE计算
    
    # 使用evogp进行符号回归
    descriptor_back = GenerateDescriptor(
        max_tree_len=128,
        input_len=gp_train_X_back.shape[1],  # 输入是GP前两层的输出（2维）
        output_len=1,
        using_funcs=["+", "-", "*", "/", "sin", "cos", "tan"],
        max_layer_cnt=7,
        # const_range=[-5, 5],
        # sample_cnt=10000,
        const_samples=[-2, -1, 0, 1, 2],
        layer_leaf_prob=0.3,
    )

    gp_model = Regressor(
        initial_forest=Forest.random_generate(pop_size=1000, descriptor=descriptor_back),
        crossover=DefaultCrossover(),
        mutation=DefaultMutation(
            mutation_rate=0.1, descriptor=descriptor_back.update(max_layer_cnt=4)
        ),
        selection=TournamentSelection(
            tournament_size=20, survivor_rate=0.5, elite_rate=0.1
        ),
        generation_limit=200,
        optimize_constants=True,
        bfgs_top_k=20,
        bfgs_max_iter=100,
        bfgs_async=True,
        bfgs_start_gen=100,  # 前 100 代不做常数优化
    )
    gp_model.fit(gp_train_X_back, gp_train_y_back)
    
    # 评估GP模型
    gp_pred_train = gp_model.predict(gp_train_X_back)
    gp_pred_val = gp_model.predict(gp_val_X_back)
    # 使用torch计算MSE
    train_mse = torch.mean((gp_pred_train - gp_train_y_back) ** 2).item()
    val_mse = torch.mean((gp_pred_val - gp_val_y_back.view(-1, 1)) ** 2).item()
    print(f"    训练MSE: {train_mse:.6f}, 验证MSE: {val_mse:.6f}")
    
    gp_back_models.append(gp_model)

print("后两层GP模型蒸馏完成！")

# 6. 整合为完整的GP模型
print("\n步骤6: 整合前后GP模型为完整的GP模型...")

class CompleteGPModel:
    """完整的GP模型，整合前后两部分的GP模型"""
    def __init__(self, front_gp_models, back_gp_models):
        self.front_gp_models = front_gp_models  # 前两层的GP模型列表
        self.back_gp_models = back_gp_models    # 后两层的GP模型列表
    
    def predict(self, X):
        """预测函数：X -> GP前两层 -> GP后两层 -> 最终输出"""
        # 确保输入是torch张量且连续
        if not isinstance(X, torch.Tensor):
            X = torch.FloatTensor(X).to(device)
        X = X.contiguous()
        
        # 第一步：通过GP前两层模型
        front_outputs = []
        for gp_model in self.front_gp_models:
            pred = gp_model.predict(X)
            # 确保pred是二维的 (n_samples, 1)
            if pred.dim() == 1:
                pred = pred.unsqueeze(1)
            front_outputs.append(pred)
        # 使用torch.cat沿着dim=1连接，得到 (n_samples, num_models)
        front_output = torch.cat(front_outputs, dim=1).contiguous()
        # 原始网络前两层输出经过ReLU，因此必须非负
        front_output = torch.relu(front_output)
        
        # 第二步：通过GP后两层模型
        back_outputs = []
        for gp_model in self.back_gp_models:
            pred = gp_model.predict(front_output)
            # 确保pred是二维的 (n_samples, 1)
            if pred.dim() == 1:
                pred = pred.unsqueeze(1)
            back_outputs.append(pred)
        # 使用torch.cat沿着dim=1连接
        back_output = torch.cat(back_outputs, dim=1).contiguous()
        
        return back_output

# 创建完整的GP模型
complete_gp_model = CompleteGPModel(gp_front_models, gp_back_models)

# 评估完整GP模型
print("评估完整GP模型性能...")
gp_final_pred_train = complete_gp_model.predict(X_train)
gp_final_pred_val = complete_gp_model.predict(X_val)

# 使用torch计算MSE
train_mse_final = torch.mean((gp_final_pred_train.flatten() - y_train) ** 2).item()
val_mse_final = torch.mean((gp_final_pred_val.flatten() - y_val) ** 2).item()

print(f"完整GP模型 - 训练MSE: {train_mse_final:.6f}, 验证MSE: {val_mse_final:.6f}")

# 保存GP模型
print("\n正在保存GP模型...")
os.makedirs('models/gp_models', exist_ok=True)
with open('models/gp_models/front_gp_models.pkl', 'wb') as f:
    pickle.dump(gp_front_models, f)
with open('models/gp_models/back_gp_models.pkl', 'wb') as f:
    pickle.dump(gp_back_models, f)
with open('models/gp_models/complete_gp_model.pkl', 'wb') as f:
    pickle.dump(complete_gp_model, f)
print("GP模型已保存到 models/gp_models/ 目录")

print("\n========== 蒸馏流程完成！==========")
print(f"原始神经网络验证MSE: {final_val_loss:.6f}")
print(f"完整GP模型验证MSE: {val_mse_final:.6f}")
print(f"MSE差异: {abs(val_mse_final - final_val_loss):.6f}")
