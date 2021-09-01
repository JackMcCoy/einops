import paddle

from . import RearrangeMixin, ReduceMixin
from ._weighted_einsum import WeightedEinsumMixin

__author__ = 'Alex Rogozhnikov'


class Rearrange(RearrangeMixin, paddle.nn.Layer):
    def forward(self, input):
        return self._apply_recipe(input)


class Reduce(ReduceMixin, paddle.nn.Layer):
    def forward(self, input):
        return self._apply_recipe(input)

