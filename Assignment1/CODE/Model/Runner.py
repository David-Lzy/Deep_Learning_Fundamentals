import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset


class Trainer:
    def __init__(
        self,
        model,
        data,
        learning_rate=0.001,
        batch_size=32,
        weight_decay=0.0,
        model_p=dict(),
    ):
        self.training_loss = []
        self.training_accuracy = []
        self.testing_accuracy = []

        X_train, X_test, y_train, y_test = data
        # Prepare the data for PyTorch
        X_train_tensor = Variable(torch.Tensor(X_train.values))
        y_train_tensor = Variable(torch.Tensor(y_train.values))
        X_test_tensor = Variable(torch.Tensor(X_test.values))
        y_test_tensor = Variable(torch.Tensor(y_test.values))
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

        self.train_loader = DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True
        )
        self.test_data = X_test_tensor, y_test_tensor

        self.model = model(X_train.shape[1], **model_p)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

    def train(self):
        total_loss = 0
        correct_train_preds = 0
        total_train_samples = 0

        for inputs, targets in self.train_loader:
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            # Forward pass
            outputs = self.model(inputs)
            # Compute the loss
            loss = self.criterion(outputs, targets.view(-1, 1))
            total_loss += loss.item() * len(targets)
            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()
            # Compute the number of correct predictions for training accuracy
            correct_train_preds += (
                ((outputs > 0.5).type(torch.FloatTensor).view(-1) == targets)
                .sum()
                .item()
            )
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
            test_outputs = self.model(self.test_data[0])
            test_preds = (test_outputs > 0.5).type(torch.FloatTensor)
            test_acc = accuracy_score(self.test_data[1], test_preds)
            self.testing_accuracy.append(test_acc)

        if not silent:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_train_loss:.4f}, Training Accuracy: {train_acc * 100:.2f}%, Testing Accuracy: {test_acc * 100:.2f}%"
            )

    def run(self, num_epochs, evaluation_interval=10, silent=False):
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
