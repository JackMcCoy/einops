<a href='http://arogozhnikov.github.io/images/einops/einops_video.mp4' >
<div align="center">
  <img src="http://arogozhnikov.github.io/images/einops/einops_video.gif" alt="einops package examples" />
  <br>
  <small><a href='http://arogozhnikov.github.io/images/einops/einops_video.mp4'>This video in better quality.</a></small>
  <br><br>
</div>
</a>

# einops 
[![Build Status](https://travis-ci.org/arogozhnikov/einops.svg?branch=master)](https://travis-ci.org/arogozhnikov/einops)  [![PyPI version](https://badge.fury.io/py/einops.svg)](https://badge.fury.io/py/einops)

Flexible and powerful tensor operations for readable and reliable code. 
Supports numpy, pytorch, tensorflow, and [others](#supported-frameworks).


<!--<div align="center">
  <img src="http://arogozhnikov.github.io/images/einops/einops_logo_350x350.png" 
  alt="einops package logo" width="250" height="250" />
  <br><br>
</div> -->

## Tweets 

> In case you need convincing arguments for setting aside time to learn about einsum and einops...
[Tim Rocktäschel, FAIR](https://twitter.com/_rockt/status/1230818967205425152)

> Writing better code with PyTorch and einops 👌
[Andrej Karpathy, AI at Tesla](https://twitter.com/karpathy/status/1290826075916779520)

> Slowly but surely, einops is seeping in to every nook and cranny of my code. If you find yourself shuffling around bazillion dimensional tensors, this might change your life
[Nasim Rahaman, MILA (Montreal)](https://twitter.com/nasim_rahaman/status/1216022614755463169)



## Contents

- [Tutorial](#Tutorial--Documentation) 
- [API micro-reference](#API)
- [Installation](#Installation)
- [Naming](#Naming)
- [Why using einops](#Why-using-einops-notation)
- [Supported frameworks](#Supported-frameworks)
- [Contributing](#Contributing)
- [Github repository (for issues/questions)](https://github.com/arogozhnikov/einops)


## Tutorial / Documentation 

Tutorial is the most convenient way to see `einops` in action (and right now works as a documentation)

- part 1: [einops fundamentals](https://github.com/arogozhnikov/einops/blob/master/docs/1-einops-basics.ipynb) 
- part 2: [einops for deep learning](https://github.com/arogozhnikov/einops/blob/master/docs/2-einops-for-deep-learning.ipynb)
- part 3: [real code fragments improved with einops](https://arogozhnikov.github.io/einops/pytorch-examples.html) (so far only for pytorch)   

## Installation

Plain and simple:
```bash
pip install einops
```

<!--
`einops` has no mandatory dependencies (code examples also require jupyter, pillow + backends). 
To obtain the latest github version 

```bash
pip install https://github.com/arogozhnikov/einops/archive/master.zip
```
-->

## API 

`einops` has minimalistic and powerful API.

Three operations provided ([einops tutorial](https://github.com/arogozhnikov/einops/blob/master/docs/) 
shows those cover stacking, reshape, transposition, squeeze/unsqueeze, repeat, tile, concatenate, view and numerous reductions)

```python
from einops import rearrange, reduce, repeat
# rearrange elements according to the pattern
output_tensor = rearrange(input_tensor, 't b c -> b c t')
# combine rearrangement and reduction
output_tensor = reduce(input_tensor, 'b c (h h2) (w w2) -> b h w c', 'mean', h2=2, w2=2)
# copy along a new axis 
output_tensor = repeat(input_tensor, 'h w -> h w c', c=3)
```
And two corresponding layers (`einops` keeps separate version for each framework) with the same API.

```python
from einops.layers.chainer import Rearrange, Reduce
from einops.layers.gluon import Rearrange, Reduce
from einops.layers.keras import Rearrange, Reduce
from einops.layers.torch import Rearrange, Reduce
from einops.layers.tensorflow import Rearrange, Reduce
```

Layers behave similarly to operations and have same parameters 
(for the exception of first argument, which is passed during call)

```python
layer = Rearrange(pattern, **axes_lengths)
layer = Reduce(pattern, reduction, **axes_lengths)

# apply created layer to a tensor / variable
x = layer(x)
```

Example of using layers within a model:
```python
# example given for pytorch, but code in other frameworks is almost identical  
from torch.nn import Sequential, Conv2d, MaxPool2d, Linear, ReLU
from einops.layers.torch import Rearrange

model = Sequential(
    Conv2d(3, 6, kernel_size=5),
    MaxPool2d(kernel_size=2),
    Conv2d(6, 16, kernel_size=5),
    MaxPool2d(kernel_size=2),
    # flattening
    Rearrange('b c h w -> b (c h w)'),  
    Linear(16*5*5, 120), 
    ReLU(),
    Linear(120, 10), 
)
```

<!--
Additionally two auxiliary functions provided
```python
from einops import asnumpy, parse_shape
# einops.asnumpy converts tensors of imperative frameworks to numpy
numpy_tensor = asnumpy(input_tensor)
# einops.parse_shape gives a shape of axes of interest 
parse_shape(input_tensor, 'batch _ h w') # e.g {'batch': 64, 'h': 128, 'w': 160}
```
-->

## Naming

`einops` stays for Einstein-Inspired Notation for operations 
(though "Einstein operations" is more attractive and easier to remember).

Notation was loosely inspired by Einstein summation (in particular by `numpy.einsum` operation).

## Why using `einops` notation


### Semantic information (being verbose in expectations)

```python
y = x.view(x.shape[0], -1)
y = rearrange(x, 'b c h w -> b (c h w)')
```
while these two lines are doing the same job in *some* context,
second one provides information about input and output.
In other words, `einops` focuses on interface: *what is input and output*, not *how* output is computed.

The next operation looks similar:
```python
y = rearrange(x, 'time c h w -> time (c h w)')
```
But it gives reader a hint: 
this is not an independent batch of images we are processing, 
but rather a sequence (video). 

Semantic information makes code easier to read and maintain. 

### More checks

Reconsider the same example:
```python
y = x.view(x.shape[0], -1) # x: (batch, 256, 19, 19)
y = rearrange(x, 'b c h w -> b (c h w)')
```
second line checks that input has four dimensions, 
but you can also specify particular dimensions. 
That's opposed to just writing comments about shapes since 
[comments don't work and don't prevent mistakes](https://medium.freecodecamp.org/code-comments-the-good-the-bad-and-the-ugly-be9cc65fbf83)
as we know   
```python
y = x.view(x.shape[0], -1) # x: (batch, 256, 19, 19)
y = rearrange(x, 'b c h w -> b (c h w)', c=256, h=19, w=19)
```

### Result is strictly determined

Below we have at least two ways to define depth-to-space operation
```python
# depth-to-space
rearrange(x, 'b c (h h2) (w w2) -> b (c h2 w2) h w', h2=2, w2=2)
rearrange(x, 'b c (h h2) (w w2) -> b (h2 w2 c) h w', h2=2, w2=2)
```
there are at least four more ways to do it. Which one is used by the framework?

These details are ignored, since *usually* it makes no difference, 
but it can make a big difference (e.g. if you use grouped convolutions on the next stage), 
and you'd like to specify this in your code.


### Uniformity

```python
reduce(x, 'b c (x dx) -> b c x', 'max', dx=2)
reduce(x, 'b c (x dx) (y dx) -> b c x y', 'max', dx=2, dy=3)
reduce(x, 'b c (x dx) (y dx) (z dz)-> b c x y z', 'max', dx=2, dy=3, dz=4)
```
These examples demonstrated that we don't use separate operations for 1d/2d/3d pooling, 
those all are defined in a uniform way. 

Space-to-depth and depth-to space are defined in many frameworks. But how about width-to-height?
```python
rearrange(x, 'b c h (w w2) -> b c (h w2) w', w2=2)
```

### Framework independent behavior

Even simple functions are defined differently by different frameworks

```python
y = x.flatten() # or flatten(x)
```

Suppose `x` shape was `(3, 4, 5)`, then `y` has shape ...
- numpy, cupy, chainer: `(60,)`
- keras, tensorflow.layers, mxnet and gluon: `(3, 20)`
- pytorch: no such function

### Independence of framework terminology

Example: tile vs repeat causes lots of confusion. To copy image along width:
```python
np.tile(image, (1, 2))    # in numpy
image.repeat(1, 2)        # pytorch's repeat ~ numpy's tile
```

With einops you don't need to decipher which axis was repeated:
```python
repeat(image, 'h w -> h (tile w)', tile=2)  # in numpy
repeat(image, 'h w -> h (tile w)', tile=2)  # in pytorch
repeat(image, 'h w -> h (tile w)', tile=2)  # in tf
repeat(image, 'h w -> h (tile w)', tile=2)  # in jax
repeat(image, 'h w -> h (tile w)', tile=2)  # in mxnet
... (etc.)
```

<!-- TODO examples for depth-to-space and pixel shuffle? transpose vs permute? -->

## Supported frameworks

Einops works with ...

- [numpy](http://www.numpy.org/)
- [pytorch](https://pytorch.org/)
- [tensorflow](https://www.tensorflow.org/), including [eager mode](https://www.tensorflow.org/guide/eager)
- [cupy](https://cupy.chainer.org/)
- [chainer](https://chainer.org/)
- [gluon](https://mxnet.apache.org/)
- [keras](https://keras.io/) and [tf.keras](https://www.tensorflow.org/guide/keras)
- [mxnet](https://gluon.mxnet.io/) (experimental)
- [jax](https://github.com/google/jax) (experimental)

## Contributing 

Best ways to contribute are

- spread the word about `einops`
- if you like explaining things, alternative tutorials would be very helpful
  - some people grasp `einops` ideas immediately, while many others need help-by-example
- translating examples in languages other than English is also a good idea
- use `einops` notation in your papers to strictly define used operations!

## Supported python versions

`einops` works with python 3.5 or later. 

<!--
There is nothing specific to python 3 in the code, 
we simply [need to move further](http://github.com/arogozhnikov/python3_with_pleasure).
-->