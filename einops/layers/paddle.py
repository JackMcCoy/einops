import paddle

from . import RearrangeMixin, ReduceMixin
from ._weighted_einsum import WeightedEinsumMixin

__author__ = 'Alex Rogozhnikov'


class Rearrange(RearrangeMixin, paddle.layers.Layer):
    def forward(self, input):
        return self._apply_recipe(input)


class Reduce(ReduceMixin, paddle.layers.Layer):
    def forward(self, input):
        return self._apply_recipe(input)

