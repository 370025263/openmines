import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from openmines.test.dqn import DQN, DuelingDQN, MODEL_NAME, LEARNING_RATE, TIME_ATTENTION


class DispatchDataset(Dataset):
    def __init__(self, dataset_path):
        """加载调度数据集"""
        self.data = []

        print("Loading dataset...")
        with open(dataset_path, 'r') as f:
            for line in f:
                sample = json.loads(line)
                # 确保state是正确的长度
                state = sample['state']
                # 删除最后一个元素，因为它会在模型中与time_delta拼接
                # state = state[:-1]
                state = torch.tensor(state, dtype=torch.float32)
                time_delta = torch.tensor([sample['delta_time']], dtype=torch.float32)

                # 获取事件类型
                event_name = sample['event_type']
                if event_name == "init":
                    event_type = 0
                elif event_name == "haul":
                    event_type = 1
                else:  # unhaul
                    event_type = 2
                event_type = torch.tensor(event_type, dtype=torch.long)

                # 专家动作
                action = torch.tensor(sample['action'], dtype=torch.long)

                self.data.append((state, event_type, time_delta, action))
        print(f"Loaded {len(self.data)} samples")
        print(f"State dimension: {self.data[0][0].shape[0]}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def pretrain_dqn(args):
    """使用收集的专家数据预训练DQN"""
    # 设置设备
    if torch.cuda.is_available() and not args.cpu:
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # 设置运行ID和tensorboard
    run_id = f"pretrain_{MODEL_NAME}_{args.run_id}"
    log_dir = os.path.join("runs", run_id)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    try:
        # 加载数据集
        dataset = DispatchDataset(args.dataset_path)

        # 使用自定义的collate_fn来处理数据移动到设备
        def collate_fn(batch):
            states, event_types, time_deltas, actions = zip(*batch)
            return (
                torch.stack(states).to(device),
                torch.stack(event_types).to(device),
                torch.stack(time_deltas).to(device),
                torch.stack(actions).to(device)
            )

        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0 if device.type == 'cuda' else 4,
            collate_fn=collate_fn
        )

        # 从数据集中获取一个样本来初始化网络
        sample_state, _, _, _ = dataset[0]
        state_dim = sample_state.shape[0]  # 这里已经是正确的维度了

        # 从环境配置获取动作空间维度
        with open(args.env_config, 'r') as f:
            env_config = json.load(f)
        action_dims = [
            len(env_config['load_sites']),  # load_num for init
            len(env_config['dump_sites']),  # unload_num for haul
            len(env_config['load_sites'])  # load_num for unhaul
        ]

        print(f"State dimension: {state_dim}")
        print(f"Action dimensions: {action_dims}")

        # 初始化网络
        if MODEL_NAME == "DQN":
            model = DQN(state_dim, action_dims).to(device)
        else:
            model = DuelingDQN(state_dim, action_dims).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        best_accuracy = 0
        total_steps = 0

        for epoch in range(args.epochs):
            model.train()
            epoch_loss = 0
            epoch_correct = 0
            epoch_total = 0

            progress_bar = tqdm(dataloader, desc=f'Epoch {epoch + 1}/{args.epochs}')

            for batch_idx, (states, event_types, time_deltas, actions) in enumerate(progress_bar):
                try:
                    optimizer.zero_grad()

                    # 前向传播
                    q_values = model(states, event_types, time_deltas)

                    # 计算损失
                    loss = criterion(q_values, actions)

                    # 反向传播
                    loss.backward()
                    optimizer.step()

                    # 计算准确率
                    predictions = q_values.argmax(dim=1)
                    correct = (predictions == actions).sum().item()

                    # 更新统计
                    epoch_loss += loss.item()
                    epoch_correct += correct
                    epoch_total += actions.size(0)

                    # 计算当前批次的准确率
                    current_accuracy = epoch_correct / epoch_total
                    current_loss = epoch_loss / (batch_idx + 1)

                    # 更新进度条
                    progress_bar.set_postfix({
                        'Loss': f'{current_loss:.4f}',
                        'Accuracy': f'{current_accuracy:.4f}'
                    })

                    # 记录到tensorboard
                    writer.add_scalar('Pretrain/Loss', loss.item(), total_steps)
                    writer.add_scalar('Pretrain/Accuracy', current_accuracy, total_steps)

                    total_steps += 1

                except Exception as e:
                    print(f"Error in batch {batch_idx}: {str(e)}")
                    continue

            # 计算epoch的平均值
            epoch_accuracy = epoch_correct / epoch_total
            epoch_avg_loss = epoch_loss / len(dataloader)

            # 如果是最好的模型就保存
            if epoch_accuracy > best_accuracy:
                best_accuracy = epoch_accuracy
                model_save_path = os.path.join(log_dir, f"pretrained_{MODEL_NAME}_best.pth")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'accuracy': epoch_accuracy,
                    'state_dim': state_dim,
                    'action_dims': action_dims,
                    'config': {
                        'state_dim': state_dim,
                        'action_dims': action_dims,
                        'run_id': run_id,
                    }
                }, model_save_path)

            print(f"\nEpoch {epoch + 1}")
            print(f"Average Loss: {epoch_avg_loss:.4f}")
            print(f"Accuracy: {epoch_accuracy:.4f}")
            print(f"Best Accuracy: {best_accuracy:.4f}")

        writer.close()
        print(f"Pretraining completed. Best accuracy: {best_accuracy:.4f}")
        return model_save_path

    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        writer.close()
        return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pretrain DQN with expert data")

    parser.add_argument("--dataset_path", type=str, required=True,
                        help="数据集文件路径(.jsonl)")
    parser.add_argument("--env_config", type=str,
                        default="../../conf/north_pit_mine.json",
                        help="环境配置文件路径")
    parser.add_argument("--epochs", type=int, default=50,
                        help="预训练轮数")
    parser.add_argument("--batch_size", type=int, default=100,
                        help="批次大小")
    parser.add_argument("--run_id", type=str, default=None,
                        help="运行ID,用于tensorboard日志")
    parser.add_argument("--cpu", action="store_true",
                        help="强制使用CPU训练")

    args = parser.parse_args()
    if args.run_id is None:
        args.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    pretrain_dqn(args)