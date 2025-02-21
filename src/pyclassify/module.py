import torch
import torch.nn as nn
import lightning.pytorch as pl
import torchmetrics

class Classifier(pl.LightningModule):
    def __init__(self, model: nn.Module):
        super().__init__()

        self.model = model
        self.train_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=self.model.num_classes)
        self.val_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=self.model.num_classes)
        self.test_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=self.model.num_classes)

    def forward(self, x):
        return self.model(x)

    def _classifier_step(self, batch):
        features, true_labels = batch
        logits = self(features)
        CrossEntropyLoss = nn.functional.cross_entropy(logits, true_labels)
        predicted_labels = torch.argmax(logits, dim=1)
        return predicted_labels, true_labels, CrossEntropyLoss
    
    def training_step(self, batch):
        predicted_labels, true_labels, CrossEntropyLoss = self._classifier_step(batch)
        self.train_accuracy(predicted_labels, true_labels)
        self.log('train_accuracy', self.train_accuracy, on_step=True, on_epoch=False)
        self.log('train_loss', CrossEntropyLoss)
        return CrossEntropyLoss
    
    def validation_step(self, batch):
        predicted_labels, true_labels, CrossEntropyLoss = self._classifier_step(batch)
        self.val_accuracy(predicted_labels, true_labels)
        self.log('val_accuracy', self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_loss', CrossEntropyLoss)

    def test_step(self, batch):
        predicted_labels, true_labels, _ = self._classifier_step(batch)
        self.test_accuracy(predicted_labels, true_labels)
        self.log('test_accuracy', self.test_accuracy, on_step=False, on_epoch=True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        return optimizer