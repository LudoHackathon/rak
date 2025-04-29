import torch.nn as nn
from torch.optim import AdamW
from lightning import LightningModule


class LitLM(LightningModule):
    def __init__(self, lm, vocab_size, lr=1e-3):
        super().__init__()
        self.model = lm
        self.vocab_size = vocab_size
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch):
        input_ids, target_ids = batch
        output = self.model(input_ids)
        loss = self.criterion(output.view(-1, self.vocab_size), target_ids.view(-1))
        return loss

    def training_step(self, batch):
        return self._shared_step(batch)
    
    def validation_step(self, batch):
        return self._shared_step(batch)
    
    def test_step(self, batch):
        return self._shared_step(batch)

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr)
