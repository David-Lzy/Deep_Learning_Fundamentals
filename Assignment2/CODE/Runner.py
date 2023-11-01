import time
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset


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
        self.handle_data(data, batch_size)
        self.model = model(**model_p)
        self.device = device if not device is None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        class_num = model_p.get("num_classes", class_num or 2)
        if class_num == 2:
            self.criterion = nn.BCELoss()
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=3, gamma=0.5
            ) if scheduler is None else None if scheduler is False else torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=scheduler[0], gamma=scheduler[1])
        

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
                dataset=train_dataset, batch_size=batch_size, shuffle=True
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
