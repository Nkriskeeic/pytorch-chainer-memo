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
