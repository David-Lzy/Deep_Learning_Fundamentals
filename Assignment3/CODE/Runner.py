import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import os
import sys
import getpass
from binance.spot import Spot

loc_list = os.path.abspath(__file__).split(os.sep)
HOME_LOC = os.path.join(os.sep, *loc_list[:-2])
sys.path.append(HOME_LOC)
os.chdir(HOME_LOC)
print(HOME_LOC)


from CODE.Utils.evaluate import *


class Trainer:
    def __init__(
        self,
        model,
        data,
        batch_size=None,
        class_num=None,
        learning_rate=0.01,
        weight_decay=0.0,
        model_p=dict(),
        device=None,
        scheduler=None,
    ):
        self.training_loss = []
        self.training_accuracy = []
        self.testing_accuracy = []

        batch_size = model_p.get("batch_size", batch_size or 32)
        self.batch_size = batch_size
        self.handle_data(data, batch_size)
        self.model = model(**model_p)
        self.device = (
            device
            if not device is None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)

        class_num = model_p.get("num_classes", class_num or 2)
        if class_num == 2:
            self.criterion = nn.BCELoss()
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.scheduler = (
            torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.5)
            if scheduler is None
            else None
            if scheduler is False
            else torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=scheduler[0], gamma=scheduler[1]
            )
        )

    def handle_data(self, data, batch_size):
        if isinstance(data, tuple) and len(data) == 4:
            X_train, X_test, y_train, y_test = data

            # Check if the data is in numpy format
            if isinstance(X_train, np.ndarray):
                X_train_tensor = Variable(torch.Tensor(X_train))
                y_train_tensor = Variable(torch.Tensor(y_train))
                X_test_tensor = Variable(torch.Tensor(X_test).long())
                y_test_tensor = Variable(torch.Tensor(y_test).long())
            # Check if the data is in pandas DataFrame format
            elif hasattr(X_train, "values"):
                X_train_tensor = Variable(torch.Tensor(X_train.values))
                y_train_tensor = Variable(torch.Tensor(y_train.values))
                X_test_tensor = Variable(torch.Tensor(X_test.values).long())
                y_test_tensor = Variable(torch.Tensor(y_test.values).long())

            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            self.train_loader = DataLoader(
                dataset=train_dataset, batch_size=batch_size, shuffle=False
            )
            test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
            self.test_loader = DataLoader(
                dataset=test_dataset, batch_size=batch_size, shuffle=False
            )
            self.test_data = X_test_tensor, y_test_tensor

        elif isinstance(data, tuple) and len(data) == 2:
            self.train_loader, self.test_data = data
        else:
            raise ValueError("Invalid data format provided.")

    def train(self):
        total_loss = 0
        correct_train_preds = 0
        total_train_samples = 0

        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            # Forward pass
            outputs = self.model(inputs)
            # Compute the loss
            loss = self.criterion(outputs, targets)
            total_loss += loss.item() * len(targets)
            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()
            # Compute the number of correct predictions for training accuracy
            _, predicted = torch.max(outputs, 1)
            correct_train_preds += (predicted == targets).sum().item()
            total_train_samples += len(targets)

        return total_loss, total_train_samples, correct_train_preds

    def evaluate(
        self,
        total_loss,
        total_train_samples,
        correct_train_preds,
        epoch,
        num_epochs,
        silent=False,
    ):
        # Compute training loss and accuracy
        avg_train_loss = total_loss / total_train_samples
        train_acc = correct_train_preds / total_train_samples
        self.training_loss.append(avg_train_loss)
        self.training_accuracy.append(train_acc)

        # Compute testing accuracy
        with torch.no_grad():
            correct_test_preds = 0
            total_test_samples = 0
            for inputs, targets in self.test_data:  # 这里修改为迭代DataLoader
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                test_outputs = self.model(inputs)
                _, predicted = torch.max(test_outputs, 1)
                correct_test_preds += (predicted == targets).sum().item()
                total_test_samples += len(targets)
            test_acc = correct_test_preds / total_test_samples
            self.testing_accuracy.append(test_acc)

        eval_duration = time.time() - self.__start_time__  # <-- 计算评估所花费的时间

        if not silent:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_train_loss:.4f}, Training Accuracy: {train_acc * 100:.2f}%, Testing Accuracy: {test_acc * 100:.2f}%, Evaluation Time: {eval_duration/60:.2f} minutes."
            )

    def run(self, num_epochs, evaluation_interval=10, silent=False):
        self.__start_time__ = time.time()

        for epoch in range(num_epochs):
            total_loss, total_train_samples, correct_train_preds = self.train()
            # Evaluate the model every evaluation_interval epochs
            if (epoch + 1) % evaluation_interval == 0:
                self.evaluate(
                    total_loss,
                    total_train_samples,
                    correct_train_preds,
                    epoch,
                    num_epochs,
                    silent=silent,
                )
            self.model.train()  # Switch back to training mode
            if not self.scheduler is None:
                self.scheduler.step()
        return self.get_result()

    def get_result(self):
        self.result = (
            self.training_loss[-1],
            self.training_accuracy[-1],
            self.testing_accuracy[-1],
        )
        return self.result


# Usage:
# trainer = Trainer(model, train_loader, (X_test_tensor, y_test_tensor))
# trainer.run(num_epochs=1000, evaluation_interval=100)


class GroupedTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 计算 group_size 为数据集样本数除以 batch_size
        self.group_size = len(self.train_loader.dataset) // self.batch_size
        self.criterion = nn.MSELoss()
        self.losses = []
        self.train_sharpe_ratios = []
        self.train_annualized_returns = []
        self.test_sharpe_ratios = []
        self.test_annualized_returns = []
        self.train_accuracies = []
        self.test_accuracies = []
        self.losses = []

    def handle_data(self, data, batch_size):
        if isinstance(data, tuple) and len(data) == 4:
            X_train, X_test, y_train, y_test = data

            # Check if the data is in numpy format
            if isinstance(X_train, np.ndarray):
                X_train_tensor = Variable(torch.Tensor(X_train))
                y_train_tensor = Variable(torch.Tensor(y_train))
                X_test_tensor = Variable(torch.Tensor(X_test))
                y_test_tensor = Variable(torch.Tensor(y_test))
            # Check if the data is in pandas DataFrame format
            elif hasattr(X_train, "values"):
                X_train_tensor = Variable(torch.Tensor(X_train.values))
                y_train_tensor = Variable(torch.Tensor(y_train.values))
                X_test_tensor = Variable(torch.Tensor(X_test.values))
                y_test_tensor = Variable(torch.Tensor(y_test.values))

            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            self.train_loader = DataLoader(
                dataset=train_dataset, batch_size=batch_size, shuffle=False
            )
            test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
            self.test_loader = DataLoader(
                dataset=test_dataset, batch_size=batch_size, shuffle=False
            )
            self.test_data = X_test_tensor, y_test_tensor

        elif isinstance(data, tuple) and len(data) == 2:
            self.train_loader, self.test_data = data
        else:
            raise ValueError("Invalid data format provided.")

    def train_1(self, epoch, iter_n=10, silent=False):
        i_epoch = 0
        sub_loop_n = max(epoch // (self.group_size * self.batch_size), 1)
        total_loop_n = sub_loop_n * self.group_size * self.batch_size
        print(
            f"total_loop_n: {total_loop_n}, len(self.group_size): {(self.group_size)}"
        )
        for group_index in range(self.group_size):
            print(f"group_index: {group_index}")
            for i in range(sub_loop_n):
                # 从每个组中抽取一个batch进行训练
                j = 0
                for inputs, targets in self.train_loader:
                    j += 1
                    if j % self.group_size == group_index:
                        inputs, targets = inputs.to(self.device), targets.to(
                            self.device
                        )
                        self.optimizer.zero_grad()
                        outputs = self.model(inputs)

                        loss = self.criterion(outputs, targets)
                        total_loss = loss.item() * len(targets)
                        if i_epoch % iter_n == 0:
                            self.test(i_epoch, total_loss, silent=silent)
                        loss.backward()
                        self.optimizer.step()
                        i_epoch += 1
                        # if i_epoch >= epoch:
                        #     return total_loss

        return total_loss

    def train(self, epoch, iter_n=10, silent=False):
        i_epoch = 0
        for i in tqdm(range(epoch)):
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()

                # 获取模型的预测结果
                outputs = self.model(inputs)

                # 将预测结果和目标值的符号转换为二分类标签
                # 符号相同为1，符号不同为0
                predicted_labels = torch.sign(outputs) == torch.sign(targets)
                predicted_labels = predicted_labels.float()  # 转换为浮点数

                # 使用二元交叉熵损失
                criterion = nn.BCEWithLogitsLoss()
                loss = criterion(outputs, predicted_labels)

                # 反向传播和优化
                loss.backward()
                self.optimizer.step()

                if i_epoch % iter_n == 0:
                    if not silent:
                        print(f"Epoch [{i_epoch+1}/{epoch}], Loss: {loss.item()}")
                    self.test(
                        i_epoch, loss.mean().cpu().detach().numpy(), silent=silent
                    )

                i_epoch += 1

        return loss.item()

    # def train(self, epoch, iter_n=10, silent=False):
    #     i_epoch = 0
    #     sub_loop_n = max(epoch // (self.group_size * len(self.train_loader)), 1)
    #     total_loop_n = sub_loop_n * self.group_size * len(self.train_loader)
    #     for group_index in range(self.group_size):

    def test(self, i_epoch, total_loss, silent=True):
        train_correct = 0
        train_total = 0
        test_correct = 0
        test_total = 0

        # 对训练集进行评估
        with torch.no_grad():
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                predicted = torch.sign(outputs) == torch.sign(targets)
                train_correct += predicted.sum().item()
                train_total += predicted.numel()

        # 计算训练集准确率
        train_accuracy = train_correct / train_total

        # 对测试集进行评估
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                predicted = torch.sign(outputs) == torch.sign(targets)
                test_correct += predicted.sum().item()
                test_total += predicted.numel()

        # 计算测试集准确率
        test_accuracy = test_correct / test_total

        # 打印结果
        if not silent:
            print(
                f"Epoch: {i_epoch} Loss: {total_loss:.2f}, "
                f"Train Accuracy: {train_accuracy:.2f}, "
                f"Test Accuracy: {test_accuracy:.2f}"
            )

        # 保存这些指标以供进一步分析
        self.train_accuracies.append(train_accuracy)
        self.test_accuracies.append(test_accuracy)
        self.losses.append(total_loss)

        # 返回性能指标
        return {
            "Train Accuracy": train_accuracy,
            "Test Accuracy": test_accuracy,
        }

    def test1(self, i_epoch, total_loss, silent=True):
        all_train_daily_returns = []
        all_test_daily_returns = []

        # 对训练集进行评估
        with torch.no_grad():
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs).cpu().numpy()
                targets = targets.cpu().numpy()
                # 处理每个预测（长度为2的数组）
                for output, target in zip(outputs, targets):
                    # 逐元素比较符号并计算实际收益率
                    daily_returns = [
                        abs(o) if np.sign(o) == np.sign(t) else -abs(o)
                        for o, t in zip(output, target)
                    ]
                    all_train_daily_returns.extend(daily_returns)

        # 计算整体训练集的平均日收益率
        train_daily_returns = np.mean(all_train_daily_returns)
        train_sharpe_ratio = calculate_sharpe_ratio(all_train_daily_returns)
        train_cumulative_returns = np.cumsum(all_train_daily_returns)
        train_annualized_return = calculate_annualized_return(
            train_cumulative_returns, len(all_train_daily_returns)
        )

        # 对测试集进行评估
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs).cpu().numpy()
                targets = targets.cpu().numpy()
                # 处理每个预测（长度为2的数组）
                for output, target in zip(outputs, targets):
                    # 逐元素比较符号并计算实际收益率
                    daily_returns = [
                        o if np.sign(o) == np.sign(t) else -o if abs(o) > 0.02 else 0
                        for o, t in zip(output, target)
                    ]
                    all_test_daily_returns.extend(daily_returns)

        # 计算整体测试集的平均日收益率
        test_daily_returns = np.mean(all_test_daily_returns)
        test_sharpe_ratio = calculate_sharpe_ratio(all_test_daily_returns)
        test_cumulative_returns = np.cumsum(all_test_daily_returns)
        test_annualized_return = calculate_annualized_return(
            test_cumulative_returns, len(all_test_daily_returns)
        )

        # 打印结果
        if not silent:
            print(
                f"Epoch: {i_epoch} Loss: {total_loss:.2f}, "
                f"Train Sharpe Ratio: {train_sharpe_ratio:.2f}, Train Annualized Return: {train_annualized_return:.2f}, "
                f"Test Sharpe Ratio: {test_sharpe_ratio:.2f}, Test Annualized Return: {test_annualized_return:.2f}"
            )

        # 保存这些指标以供进一步分析
        self.train_sharpe_ratios.append(train_sharpe_ratio)
        self.train_annualized_returns.append(train_annualized_return)
        self.test_sharpe_ratios.append(test_sharpe_ratio)
        self.test_annualized_returns.append(test_annualized_return)

        self.losses.append(total_loss)

        # 返回性能指标
        return {
            "Train Sharpe Ratio": train_sharpe_ratio,
            "Train Annualized Return": train_annualized_return,
            "Test Sharpe Ratio": test_sharpe_ratio,
            "Test Annualized Return": test_annualized_return,
        }

    def save_metrics_to_csv(self, file_path):
        """
        Save training metrics to a CSV file.
        :param file_path: Path to the CSV file.
        """
        # 创建一个字典，其中包含所有的评估指标
        metrics = {
            "Losses": self.losses,
            "Train Accuracy": self.train_accuracies,
            "Test Accuracy": self.test_accuracies,
        }

        # 转换为DataFrame
        df = pd.DataFrame(metrics)

        # 保存为CSV文件
        df.to_csv(file_path, index=False)
