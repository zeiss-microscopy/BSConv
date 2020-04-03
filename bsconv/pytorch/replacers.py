import abc
import math
import types

import torch


###
#%% utils
###


def forceTwoTuple(x):
    if isinstance(x, list):
        x = tuple(x)
    if not isinstance(x, tuple):
        x = (x, x)
    return x


###
#%% module filters
###


class ModuleFilter(abc.ABC):
    def __repr__(self):
        return "<{}>".format(
            type(self).__name__,
        )

    @abc.abstractmethod
    def apply(self, module, name, full_name):
        """
        Return `True` if module matches this filter, and `False` otherwise.
        """
        pass


class ModelFilter(ModuleFilter):
    def apply(self, module, name, full_name):
        return full_name == ""


class Conv2dFilter(ModuleFilter):
    def __init__(self, kernel_sizes=None):
        self.kernel_sizes = kernel_sizes
        if self.kernel_sizes is not None:
            self.kernel_sizes = tuple(forceTwoTuple(kernel_size) for kernel_size in self.kernel_sizes)

    def apply(self, module, name, full_name):
        if not isinstance(module, torch.nn.Conv2d):
            return False
        if (self.kernel_sizes is None) or (module.kernel_size in self.kernel_sizes):
            return True
        else:
            return False 


###
#%% module transformers
###


class ModuleTransformer(abc.ABC):
    def __repr__(self):
        return "<{}>".format(
            type(self).__name__,
        )

    @abc.abstractmethod
    def apply(self, module):
        pass


class BSConvUTransformer(ModuleTransformer):
    def apply(self, module, name, full_name):
        new_module = torch.nn.Sequential()
        new_module.add_module("pw", torch.nn.Conv2d(
                in_channels=module.in_channels,
                out_channels=module.out_channels,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=False,
        ))
        new_module.add_module("dw", torch.nn.Conv2d(
                in_channels=module.out_channels,
                out_channels=module.out_channels,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.out_channels,
                bias=module.bias is not None,
        ))
        return new_module


class BSConvSTransformer(ModuleTransformer):
    def __init__(self, p, with_bn_relu, bn_kwargs):
        self.p = p
        assert isinstance(self.p, float)
        assert 0.0 < self.p <= 0.5
        
        self.with_bn_relu = with_bn_relu

        self.bn_kwargs = bn_kwargs
        if self.bn_kwargs is None:
            self.bn_kwargs = {}

    def apply(self, module, name, full_name):
        mid_channels = max(1, math.ceil(self.p * module.in_channels))
        new_module = torch.nn.Sequential()
        new_module.add_module("pw1", torch.nn.Conv2d(
            in_channels=module.in_channels,
            out_channels=mid_channels,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        ))
        new_module.add_module("pw2", torch.nn.Conv2d(
            in_channels=mid_channels,
            out_channels=module.out_channels,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        ))
        if self.with_bn_relu:
            new_module.add_module("bn", torch.nn.BatchNorm2d(num_features=module.out_channels, **self.bn_kwargs))
            new_module.add_module("activ", torch.nn.ReLU(inplace=True))
        new_module.add_module("dw", torch.nn.Conv2d(
            in_channels=module.out_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.out_channels,
            bias=module.bias is not None,
        ))

        def _reg_loss(self, alpha=0.1):
            W = self.weight[:, :, 0, 0]
            WWt = torch.mm(W, torch.transpose(W, 0, 1))
            I = torch.eye(WWt.shape[0], device=WWt.device)
            return alpha * torch.norm(WWt - I, p="fro")
        new_module[0]._reg_loss = types.MethodType(_reg_loss, new_module[0])

        return new_module


class RegularizationMethodTransformer(ModuleTransformer):
    def apply(self, module, name, full_name):
        def reg_loss(self):
            loss = 0.0
            for sub_module in module.modules():
                try:
                    sub_loss = sub_module._reg_loss()
                except AttributeError:
                    continue
                if loss is None:
                    loss = torch.tensor(0.0, dtype=torch.float32, device=sub_loss.device)
                else:
                    loss += sub_loss
            return loss
        module.reg_loss = types.MethodType(reg_loss, module)
        return module


###
#%% module replacement
###


class ModuleReplacementRule():
    def __init__(self, filter_, transformer):
        self.filter = filter_
        self.transformer = transformer

    def __repr__(self):
        return "<{}: {} => {}>".format(
            type(self).__name__,
            self.filter,
            self.transformer,
        )


class ModuleReplacer():
    def __init__(self):
        self.rules = []

    def add_rule(self, *args):
        if (len(args) == 1) and isinstance(args[0], ModuleReplacementRule):
            rule = args[0]
        elif (len(args) == 2) and isinstance(args[0], ModuleFilter) and isinstance(args[1], ModuleTransformer):
            rule = ModuleReplacementRule(filter_=args[0], transformer=args[1])
        else:
            raise TypeError("Rule must be specified either as instance of ModuleReplacementRule or as pair of ModuleFilter and ModuleTransformer instances")
        self.rules.append(rule)
    
    def __repr__(self):
        return "<{}: {} rule(s)>".format(
            type(self).__name__,
            len(self.rules),
        )

    def apply(self, module):
        (_, module) = self._apply_rules(module=module, name="", full_name="")
        module = self._apply_recursively(module=module)
        return module

    def _apply_rules(self, module, name, full_name):
        for rule in self.rules:
            if rule.filter.apply(module=module, name=name, full_name=full_name):
                # if filter matches, apply transform to module
                module = rule.transformer.apply(module=module, name=name, full_name=full_name)
                print("Applied rule {} to '{}'".format(rule, full_name))
                return (True, module)

        # signal that no rule was applied
        return (False, module)

    def _apply_recursively(self, module, name_prefix=""):
        named_children = list(module.named_children())
        for (child_name, child) in named_children:
            if not isinstance(child, torch.nn.Module):
                continue

            child_full_name = "{}{}".format(name_prefix, child_name)
            (child_changed, new_child) = self._apply_rules(module=child, name=child_name, full_name=child_full_name)
            if child_changed:
                # some rule was applied -> replace child
                setattr(module, child_name, new_child)
            else:
                # no rule applied, recurse into child module
                new_child = self._apply_recursively(module=child, name_prefix="{}{}.".format(name_prefix, child_name))
                setattr(module, child_name, new_child)

        return module


class BSConvU_Replacer(ModuleReplacer):
    def __init__(self, kernel_sizes=((3, 3), (5, 5))):
        super().__init__()
        self.add_rule(
            Conv2dFilter(kernel_sizes=kernel_sizes),
            BSConvUTransformer(),
        )


class BSConvS_Replacer(ModuleReplacer):
    def __init__(self, kernel_sizes=((3, 3), (5, 5)), p=0.25, with_bn_relu=True, bn_kwargs=None):
        super().__init__()
        self.add_rule(
            Conv2dFilter(kernel_sizes=kernel_sizes),
            BSConvSTransformer(p=p, with_bn_relu=with_bn_relu, bn_kwargs=bn_kwargs),
        )
        self.add_rule(
            ModelFilter(),
            RegularizationMethodTransformer(),
        )

