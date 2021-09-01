import paddle

from . import RearrangeMixin, ReduceMixin
from ._weighted_einsum import WeightedEinsumMixin

__author__ = 'Alex Rogozhnikov'


class Rearrange(paddle.nn.Layer,RearrangeMixin):
    def forward(self, input):
        return self._apply_recipe(input)


class Reduce(paddle.nn.Layer,ReduceMixin):
    def forward(self, input):
        return self._apply_recipe(input)

