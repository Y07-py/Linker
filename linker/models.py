import linker.functions as F
import linker.layers as L
from linker.utils import get_dot_graph
from linker.layers import Layer

class Model(Layer):
    def plot(self, *inputs, to_file='model.png'):
        y = self.forward(*inputs)
        return plot_dot_graph(y, verbose=True, to_file=to_file)

class MLP(Model): # 多層パーセプトロン
    def __init__(self, fc_output_sizes, activation=F.sigmoid):
        super().__init__()
        self.activation = activation
        self.layers = []

        for i, out_size in enumerate(fc_output_sizes):
            layer = L.Linear(out_size)
            setattr(self, 'layer' + str(i), layer)
            self.layers.append(layer)
    
    def forward(self, x):
        for l in self.layers[:-1]:
            x = self.activation(l(x))
        return self.layers[-1](x)