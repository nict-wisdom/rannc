import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicModel(nn.Module):

    INPUT_DIM = (1, 32, 32)
    OUTPUT_DIM = (10,)

    def __init__(self):
        super(BasicModel, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class SmallParamModel(nn.Module):

    INPUT_DIM = (3,)
    OUTPUT_DIM = (3,)

    def __init__(self):
        super(SmallParamModel, self).__init__()
        self.fc1 = nn.Linear(3, 4, bias=False)
        self.fc2 = nn.Linear(4, 3, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = x * 2
        x = x * 3
        return x


class NestModel(nn.Module):
    INPUT_DIM = (3,)
    OUTPUT_DIM = (3,)

    def __init__(self):
        super(NestModel, self).__init__()
        self.submodel = SmallParamModel()
        self.fc = nn.Linear(3, 3, bias=False)

    def forward(self, x):
        x = self.submodel(x)
        x = self.fc(x)
        x = x * 2
        x = x * 3
        return x


class LossOutModel(nn.Module):
    INPUT_DIM = (3,)
    OUTPUT_DIM = (3,)

    def __init__(self):
        super(LossOutModel, self).__init__()
        self.fc1 = nn.Linear(3, 3, bias=False)
        self.fc2 = nn.Linear(3, 3, bias=False)

    def forward(self, x, y):
        x = self.fc1(x)
        x = self.fc2(x)
        criterion = nn.MSELoss()
        loss = criterion(x, y)
        return loss


class SharedParamModel(nn.Module):

    INPUT_DIM = (3,)
    OUTPUT_DIM = (3,)

    def __init__(self):
        super(SharedParamModel, self).__init__()
        self.fc1 = nn.Linear(3, 3, bias=False)
        self.fc2 = nn.Linear(3, 3, bias=False)
        self.fc2.weight = self.fc1.weight

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = x*2
        x = x*3
        return x


class ForkJoinModel(nn.Module):

    INPUT_DIM = (3,)
    OUTPUT_DIM = (3,)

    def __init__(self):
        super(ForkJoinModel, self).__init__()
        self.fc1 = nn.Linear(3, 3, bias=False)
        self.fc2 = nn.Linear(3, 3, bias=False)
        self.fc3 = nn.Linear(3, 3, bias=False)
        self.fc4 = nn.Linear(3, 3, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        y1 = self.fc2(x)
        y2 = self.fc3(x)*2
        y = self.fc4(y1+y2)
        return y


class SharedInputModel(nn.Module):

    INPUT_DIM = (3,)
    OUTPUT_DIM = (3,)

    def __init__(self):
        super(SharedInputModel, self).__init__()
        self.fc1 = nn.Linear(3, 3, bias=False)
        self.fc2 = nn.Linear(3, 3, bias=False)
        self.fc3 = nn.Linear(3, 3, bias=False)
        self.fc4 = nn.Linear(3, 3, bias=False)

    def forward(self, x):
        y1 = self.fc1(x)
        y2 = self.fc2(y1)
        y3 = self.fc3(y1)
        y4 = self.fc4(y1)
        return y1 + y2 + y3 + y4


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        do_convert = self.weight.dtype != x.dtype
        input_dtype = x.dtype
        if do_convert:
            x = x.to(self.weight.dtype)
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        x = self.weight * x + self.bias
        if do_convert:
            x = x.to(input_dtype)
        return x

class LayerNormModel(nn.Module):

    INPUT_DIM = (3,)
    OUTPUT_DIM = (3,)

    def __init__(self):
        super(LayerNormModel, self).__init__()
        self.fc1 = nn.Linear(3, 2, bias=False)
        self.norm = BertLayerNorm(2)
        self.fc2 = nn.Linear(2, 3, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.fc2(x)
        return x


def norm_to_float(module):
    for name, _module in module.named_modules():
        if name.endswith('norm'.lower()):
            print("changing to fp32={}".format(name))
            _module.float()


class BufferModel1(nn.Module):

    INPUT_DIM = (10,)
    OUTPUT_DIM = (10,)

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)
        self.bn = nn.BatchNorm1d(10, affine=True, track_running_stats=False)

    def forward(self, x):
        x = self.fc(x)
        return self.bn(x)


class BufferModel2(nn.Module):

    INPUT_DIM = (10,)
    OUTPUT_DIM = (10,)

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)
        self.bn = nn.BatchNorm1d(10, affine=False, track_running_stats=True)

    def forward(self, x):
        x = self.fc(x)
        return self.bn(x)


class BufferModel3(nn.Module):

    INPUT_DIM = (10,)
    OUTPUT_DIM = (10,)

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)
        self.bn = nn.BatchNorm1d(10, affine=False, track_running_stats=False)

    def forward(self, x):
        x = self.fc(x)
        return self.bn(x)


class BufferModel3(nn.Module):

    INPUT_DIM = (10,)
    OUTPUT_DIM = (10,)

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)
        self.bn = nn.BatchNorm1d(10, affine=False, track_running_stats=False)

    def forward(self, x):
        x = self.fc(x)
        return self.bn(x)


class BufferModel3(nn.Module):

    INPUT_DIM = (10,)
    OUTPUT_DIM = (10,)

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)
        self.bn = nn.BatchNorm1d(10, affine=False, track_running_stats=False)

    def forward(self, x):
        x = self.fc(x)
        return self.bn(x)


class IdentityModel(nn.Module):

    INPUT_DIM = (10,)
    OUTPUT_DIM = (10,)

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class OneOpModel(nn.Module):

    INPUT_DIM = (10,)
    OUTPUT_DIM = (10,)

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * 2


class TensorMulModel(torch.nn.Module):

    INPUT_DIM = (10,)
    OUTPUT_DIM = (10,)

    def  __init__(self):
        super().__init__()
        self._two = torch.tensor([2]).cuda()

    def forward(self, x):
        x = torch.mul(x, self._two)
        return  x


class EmbeddingModel(nn.Module):

    SEQ_LEN = 16
    VOCAB_SIZE = 10
    EMB_SIZE = 8
    INPUT_DIM = ()
    OUTPUT_DIM = ()

    @staticmethod
    def get_dataset(dataset_size, input_dim, output_dim):
        ds_x = torch.randint(EmbeddingModel.VOCAB_SIZE, (dataset_size, EmbeddingModel.SEQ_LEN))
        ds_tgt = torch.randn(dataset_size, EmbeddingModel.SEQ_LEN, EmbeddingModel.EMB_SIZE)
        return ds_x, ds_tgt

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(EmbeddingModel.VOCAB_SIZE, EmbeddingModel.EMB_SIZE)

    def forward(self, x):
        return self.emb(x)


@torch.jit.script
def test_loop_function(x):
    for b in x:
        for i in range(0, b.numel()):
            if b[i] % 2 == 0:
                b[i] = 0
    return x


class FunctionModel(nn.Module):

    SEQ_LEN = 16
    VOCAB_SIZE = 10
    EMB_SIZE = 8
    INPUT_DIM = ()
    OUTPUT_DIM = ()

    @staticmethod
    def get_dataset(dataset_size, input_dim, output_dim):
        ds_x = torch.randint(FunctionModel.VOCAB_SIZE, (dataset_size, FunctionModel.SEQ_LEN))
        ds_tgt = torch.randn(dataset_size, FunctionModel.SEQ_LEN, FunctionModel.EMB_SIZE)
        return ds_x, ds_tgt

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(FunctionModel.VOCAB_SIZE, FunctionModel.EMB_SIZE)

    def forward(self, x):
        x = test_loop_function(x)
        return self.emb(x)
