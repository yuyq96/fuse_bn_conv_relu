import copy

import torch
from torch import nn
from torch import optim
from torch.ao.quantization.backend_config import BackendConfig

from torch.ao.quantization.backend_config import BackendPatternConfig
from torch.quantization import quantize_fx

from ao.quantization.fuser_method_mappings import CUSTOM_PATTERN_TO_FUSER_METHOD


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(8)
        self.fc = nn.Linear(8, 16)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc(self.bn(x)))


model = Model()
print(model)

criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters())

for _ in range(10):
    input = torch.randn(4, 8)
    target = torch.randn(4, 16)
    pred = model(input)
    loss = criterion(pred, target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

dummpy_inputs = torch.randn(1, 8)


backend_pattern_configs = []
for pattern, fuser_method in CUSTOM_PATTERN_TO_FUSER_METHOD.items():
    backend_pattern_configs.append(
        BackendPatternConfig(pattern).set_fuser_method(fuser_method)
    )
backend_config = BackendConfig('').set_backend_pattern_configs(backend_pattern_configs)

model_to_fuse = copy.copy(model)
model_to_fuse.eval()
model_fused = quantize_fx.fuse_fx(model_to_fuse, backend_config=backend_config)

print(model_fused.__repr__())
model.eval()
print((model_fused(dummpy_inputs) - model(dummpy_inputs))[0])
