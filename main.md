# メモ概要
chainerとpytorchの実装で気をつけないとなぁと思うとこをまとめる.  
基本的にchainerはV5.0.0b3のドキュメントを見ながら書いている．  
pytorchはV0.4.0ので書いている．

# Convolution系の話
## Convolution2dの呼び出し方法
- chainer
```python
 chainer.links.Convolution2D(
  in_channels,  # 入力のチャンネル数
  out_channels,   # 出力のチャンネル数
  ksize=None,   # カーネルのサイズ intもしくはTuple(h, w)
  stride=1,   # カーネルの移動幅 intもしくはTuple(h, w)
  pad=0,   # 枠のパディング幅 intもしくはTuple(h, w)
  nobias=False,   # Wx+bの定数バイアスbを使用しない(Noneが代入される)
  initialW=None,  # Initializer or ndarray(shape -> (OutChannels, InChannels/groups, kernel_h, kernel_w)) or None(LeCunNormal: scaled Gaussian distribution)
  initial_bias=None, # None(LeCunNormal) or Initializer or ndarray(shape -> (OutChannels, ))
  dilate=1, # Diation factor intもしくはTuple
  groups=1 # Grouped Convolutionを行える．2グループに分けるなら2．depth-wise convolutionをしたいならout_channelsと同じ値
)
```
- pytorch
```python
torch.nn.Conv2d(
  in_channels,  # 入力のチャンネル数
  out_channels,  # 出力のチャンネル数
  kernel_size,   # カーネルのサイズ intもしくはTuple(h, w)
  stride=1,   # カーネルの移動幅 intもしくはTuple(h, w)
  padding=0,   # 枠のパディング幅 intもしくはTuple(h, w)
  dilation=1, # Diation factor intもしくはTuple
  groups=1, # Grouped Convolutionを行える．2グループに分けるなら2．depth-wise convolutionをしたいならout_channelsと同じ値
  bias=True  # Wx+bの定数バイアスbを使用する
)
```

# 勾配計算系の話
## chainer
### chainer.grad
ある出力に対して，ある入力の勾配を求める．  
$f(x) = x^2 + 3x + 1$ のとき chainer.grad([f(10)], [10])のように使用する．($f'(10)$を求める)  

```python
# outputs -> inputsの最小パスを辿って勾配計算を行う
# inputsと同じサイズのgradのリストを返すが，grad_varに勾配値をセットしない
# 返り値はgradをVariableに格納したもののList
chainer.grad(
  outputs,  # VariableのTuple or List
  inputs,  # VariableのTuple or List
  grad_outputs=None,  # outputsの勾配値(もしあれば)
  grad_inputs=None,  # inputsの勾配値(もしあれば)
  set_grad=False,  # inputsのVariable.grad_varにgradの値を代入する
  retain_grad=False,  # outputs→inputsまでの中間Variableにもgradの値を代入する
  enable_double_backprop=False,  # 2回微分をできるようにする
  loss_scale=None
)
```
ネットワークモデルに対して使った場合，Convolution2D.W.gradなんかはNoneのまま(たぶん最小パスしか計算しないから)．
モデルの重みについても出したかったら
```python
❌ chainer.grad([loss], [input])
⭕️ chainer.grad([loss], [input]+[p for p in net.params()])
```
果たしてこれをやりたいかは謎

例:
```python
>>> x = np.array([1,2,3], dtype=np.float32)
>>> x = Variable(x)
>>> y = x ** 2 + 2*x + 1
>>> z = y ** 3 - 4*y**2 + 2
>>> chainer.grad([z], [x])
[variable([  64., 1026., 5120.])]
>>> chainer.grad([z], [y])
[variable([ 16., 171., 640.])]
>>> x.grad, y.grad, z.grad
(None, None, None)
>>> chainer.grad([z], [x], set_grad=True)
[variable([  64., 1026., 5120.])]
>>> x.grad, y.grad, z.grad
(array([  64., 1026., 5120.], dtype=float32), None, None)
>>> chainer.grad([z], [x], retain_grad=True)
[variable([  64., 1026., 5120.])]
>>> x.grad, y.grad, z.grad
(array([  64., 1026., 5120.], dtype=float32), array([ 16., 171., 640.], dtype=float32), None)
# retain_gradは(outputs, inputs]のVariableの勾配値をいれる(outputsのgradがすでに設定されていればそれを破壊しない)
```

初期勾配を入れていた時の計算結果例:
- grad_inputsを指定
```python
>>> x = np.array([1,2,3], dtype=np.float32)
>>> x = Variable(x)
>>> y = x ** 2 + 2*x + 1
>>> z = y ** 3 - 4*y**2 + 2
>>> x.grad = np.array([2,3,4], dtype=np.float32)
>>> chainer.grad([z], [x])
[variable([  64., 1026., 5120.])]
>>> x.grad
array([2., 3., 4.], dtype=float32)
>>> x.grad = None
>>> chainer.grad([z], [x], grad_inputs=[np.array([2,3,4], dtype=np.float32)])
[variable([  66., 1029., 5124.])]
>>> x.grad
None
>>> chainer.grad([z], [x], grad_inputs=[np.array([2,3,4], dtype=np.float32)], set_grad=True)
[variable([  66., 1029., 5124.])]
>>> x.grad
array([  66., 1029., 5124.], dtype=float32)
```
inputsの勾配が既に入っていても特に気にしない. grad_inputsを指定すると勾配計算後に加算される．

### chainer.Variable.backward()
```python
Variable.backward(
  retain_grad=False, # 途中の入力Variableの勾配を持つかどうか
  enable_double_backprop=False,
  loss_scale=None
)
```
このVarialeがスカラー(Variable.data.shape == ())かつgradがNoneのとき，  
自動でgradに1が代入されてからbackwardが始まる．  
MSEとかSCEとかを気軽に使えるのはこの機能のおかげ．  
**Parameterのgradは必ず値がセットされる** Variableのgradが欲しいとき以外はretain_gradはFalse  
ネットワークモデルに対してなんども呼び出した場合，Convolution2D.W.gradなんかはどんどん加算されていく．   
