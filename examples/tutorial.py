import sys
import torch
import torch.nn as nn
import torch.optim as optim

import pyrannc


class Net(nn.Module):
    def __init__(self, hidden, layers):
        super(Net, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(hidden, hidden) for i in range(layers)])

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x


batch_size = int(sys.argv[1])
hidden = int(sys.argv[2])
layers = int(sys.argv[3])

model = Net(hidden, layers)
if pyrannc.get_rank() == 0:
    print("#Parameters={}".format(sum(p.numel() for p in model.parameters())))

opt = optim.SGD(model.parameters(), lr=0.01)
model = pyrannc.RaNNCModule(model, opt)

x = torch.randn(batch_size, hidden, requires_grad=True).to(torch.device("cuda"))
out = model(x)

target = torch.randn_like(out)
out.backward(target)

opt.step()
print("Finished on rank{}".format(pyrannc.get_rank()))
