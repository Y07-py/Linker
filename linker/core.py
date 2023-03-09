import numpy as np
import heapq
from dataclasses import dataclass, field
from typing import Any
import contextlib
import weakref
from linker import utils

# heapqモジュールを用いる際、関数の型を無視して優先度を比較するためのラッパークラス

@dataclass(order=True)
class PrioritizedFunction:
    priority: int
    function: Any = field(compare=False)

# 入力変数がndarrayインスタンスであるかを確認するメソッド

def as_array(x): 
    if np.isscalar(x):
        return np.array(x)
    return x

#  Var型以外の数値データとの演算も可能にするためのメソッド

def as_variable(obj):
    if isinstance(obj, Var):
        return obj
    return Var(obj)

# 逆伝播を有効にするか無効にするかを決定するクラス及びメソッド

class Config:
    enable_backprop = True 

@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)

def no_grad(): # 勾配が必要のないときにwith構文を用いて呼ぶため
    return using_config('enable_backprop', False)

def setup_variable(): 
    Var.__mul__ = mul
    Var.__rmul__ = mul
    Var.__add__ = add
    Var.__radd__ = add
    Var.__neg__ = neg
    Var.__sub__ = sub
    Var.__rsub__ = rsub
    Var.__truediv__ = div
    Var.__rtruediv__ = rdiv
    Var.__pow__ = pow

class Var:
    __array_priority__ = 200 # Varインスタンスの演算の際の優先度を上げるため
    def __init__(self, data, name=None):  #ndarrayインスタンスのみを扱いたい
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))
        
        self.data = data
        self.grad = None # 微分した値を保持
        self.func = None
        self.priority = 0  # 関数の優先度を決定する（微分の結果を次の変数に伝えるときに必要）
        self.name = name
    
    def __len__(self):
        return len(self.data)
    
    def __repr__(self):
        if self.data is None:
            return 'Variable(None)'
        p = str(self.data).replace('\n', '\n' + '  ' * 8)
        return 'Variable(' + p + ')'

    #自動微分のためのつながりを作る
    
    def set_function(self, func):
        self.func = func 
        self.priority = func.priority - 1
    
    #逆伝播の自動化
    
    def backward(self, retain_grad=False, create_graph=False):
        if self.grad is None:
            self.grad = Var(np.ones_like(self.data)) #入力変数と同じ形状かつデータ型のVarインスタンスを生成
            
        funcs = []
        scenario = set() # backwardメソッドで同様の関数が複数回呼ばれることを防ぐために使う
        
        def add_func(f): # 優先度付きキューを用いる
            if f not in scenario:
                heapq.heappush(funcs, PrioritizedFunction(f.priority, f)) 
                scenario.add(f)
        
        add_func(self.func)
        
        while funcs:
            f = heapq.heappop(funcs).function
            gys = [output().grad for output in f.outputs]
            
            with using_config('enable_backprop', create_graph): #逆伝播の有効と無効の切り替え
                gxs = f.backward(*gys) #　複数の微分された値に対応
                if not isinstance(gxs, tuple):
                    gxs = (gxs,)
                
                for x, gx in zip(f.inputs, gxs):
                    if x.grad is None: #　同じ変数を複数用いた計算の場合の微分の結果が置き換わることを防ぐ
                        x.grad = gx
                    else:
                        x.grad = x.grad + gx
                    
                    if x.func is not None:
                        add_func(x.func)
                
                if not retain_grad:
                    for output in f.outputs:
                        output().grad = None
        
    def cleargrad(self): # 微分のリセットを行う
        self.grad = None
    
    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return reshape(self, shape)
    
    def transpose(self, *axes):
        if len(axes) == 0:
            axes = None
        elif len(axes) == 1:
            if isinstance(axes[0], (tuple, list)) or axes[0] is None:
                axes = axes[0]
        return transpose(self, axes)
    
    @property
    def shape(self):
        return self.data.shape
    
    @property
    def ndim(self):
        return self.data.ndim
    
    @property
    def size(self):
        return self.data.size
    
    @property
    def dtype(self):
        return self.data.dtype
    
    @property
    def T(self):
        return transpose(self)

class Function:
    
    # Varインスタンスが入力されることが前提
    #　複数の入出力に対応できるように
    
    def __call__(self, *inputs): #　任意の数の引数に対応
        inputs = [as_variable(x) for x in inputs]
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Var(as_array(output)) for output in ys] #出力変数がndarrayインスタンスであることを保証
        
        if Config.enable_backprop: # 逆伝播を行うかの決定
            self.priority = min([x.priority for x in inputs])
            for output in outputs:
                output.set_function(self)#出力変数に対してこのclassそのものを記憶させる
                self.inputs = inputs #入力された変数の保持
                self.outputs = [weakref.ref(output) for output in outputs] # 循環参照を避けるため出力変数を弱参照に
                    
        return outputs if len(outputs) > 1 else outputs[0]
    
    #  このクラスは継承されることを前提とする
    
    def forward(self, x):
        raise NotImplementedError()
    
    def backward(self, gy):
        raise NotImplementedError()

class Add(Function):
    def forward(self, x0, x1):
        self.x0_shape = x0.shape
        self.x1_shape = x1.shape
        y = x0 + x1
        return (y,)
    
    def backward(self, gy):
        if self.x0_shape != self.x1_shape:
            gx0 = sum_to(gy, self.x0_shape)
            gx1 = sum_to(gy, self.x1_shape)
        return gx0, gx1

def add(x0, x1):
    x1 = as_array(x1)
    return Add()(x0, x1)

class Mul(Function):
    def forward(self, x0, x1):
        self.x0_shape = x0.shape
        self.x1_shape = x1.shape
        y = x0 * x1
        return y
    
    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy * x1
        gx1 = gy * x0
        if self.x0_shape != self.x1_shape:
            gx0 = sum_to(gx0, self.x0_shape)
            gx1 = sum_to(gx1, self.x1_shape)
        return gx0, gx1

def mul(x0, x1):
    x1 = as_array(x1)
    return Mul()(x0, x1)

class Neg(Function):
    def forward(self, x):
        return -x
    
    def backward(self, gy):
        return -gy

def neg(x):
    return Neg()(x)

class Sub(Function):
    def forward(self, x0, x1):
        self.x0_shape = x0.shape
        self.x1_shape = x1.shape
        y = x0 - x1
        return y
    
    def backward(self, gy):
        if self.x0_shape != self.x1_shape:
            gx0 = sum_to(gy, self.x0_shape)
            gx1 = sum_to(gy, self.x1_shape)
        return gx0, -gx1

def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0, x1)

def rsub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x1, x0) #x1, x0を入れ替える

class Div(Function):
    def forward(self, x0, x1):
        self.x0_shape = x0.shape
        self.x1_shape = x1.shape
        y = x0 / x1
        return y
    
    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)

        if self.x0_shape != self.x1_shape:
            gx0 = sum_to(gx0, self.x0_shape)
            gx1 = sum_to(gx1, self.x1_shape)

        return gx0, gx1

def div(x0, x1):
    x1 = as_array(x1)
    return Div()(x0, x1)

def rdiv(x0, x1):
    x1 = as_array(x1)
    return Div()(x1, x0) # x1, x0を入れ替える

class Pow(Function):
    def __init__(self, c):
        self.c = c
    
    def forward(self, x):
        y = x ** self.c
        return y
    
    def backward(self, gy):
        x = self.inputs[0].data
        c = self.c
        gx = c * x ** (c - 1) * gy
        return gx

def pow(x, c):
    return Pow(c)(x)

# テンソルの形状変化のためのクラス

class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape
    
    def forward(self, x):
        self.x_shape = x.shape # ndarrayインスタンスであることが前提
        y = x.reshape(self.shape)
        return y
    
    def backward(self, gy):
        return reshape(gy, self.x_shape)

def reshape(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)

class Transpose(Function):
    def __init__(self, axes=None): # 軸の入れ替えを可能に
        self.axes = axes

    def forward(self, x):
        y = x.transpose(self.axes)
        return y
    
    def backward(self, gy):
        if self.axes is None:
            return transpose(gy)
        axes_len = len(self.axes)
        inv_axes = tuple(np.argsort([ax % axes_len for ax in self.axes]))
        return transpose(inv_axes)

def transpose(x, axes=None):
    return Transpose(axes)(x)

class BroadcastTo(Function):
    def __init__(self, shape):
        self.shape = shape
    
    def forward(self, x):
        self.x_shape = x.shape
        y = np.broadcast_to(x, self.shape)
        return y
    
    def backward(self, gy):
        gx = sum_to(gy, self.x_shape)
        return gx

def broadcast_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)

class SumTo(Function):
    def __init__(self, shape):
        self.shape = shape
    
    def forward(self, x):
        self.x_shape = x.shape
        y = utils.sum_to(x, self.shape)
        return y
    
    def backward(self, gy):
        gx = broadcast_to(gy, self.x_shape)
        return gx

def sum_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)