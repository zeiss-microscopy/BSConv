import abc
import types

import torch

import bsconv.utils
import bsconv.pytorch.modules


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
            self.kernel_sizes = tuple(bsconv.utils.forceTwoTuple(kernel_size) for kernel_size in self.kernel_sizes)

    def apply(self, module, name, full_name):
        if not isinstance(module, torch.nn.Conv2d):
            return False
        if module.groups != 1:
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


class Conv2dToBSConvUTransformer(ModuleTransformer):
    def __init__(self, with_bn, bn_kwargs):
        self.with_bn = with_bn
        self.bn_kwargs = bn_kwargs

    def apply(self, module, name, full_name):
        return bsconv.pytorch.modules.BSConvU(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            bias=module.bias is not None,
            padding_mode=module.padding_mode,
            with_bn=self.with_bn,
            bn_kwargs=self.bn_kwargs,
        )


class BSConvSTransformer(ModuleTransformer):
    def __init__(self, p, with_bn, bn_kwargs):
        self.p = p
        self.with_bn = with_bn
        self.bn_kwargs = bn_kwargs

    def apply(self, module, name, full_name):
        return bsconv.pytorch.modules.BSConvS(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            bias=module.bias is not None,
            padding_mode=module.padding_mode,
            p=self.p,
            with_bn=self.with_bn,
            bn_kwargs=self.bn_kwargs,
        )


class RegularizationMethodTransformer(ModuleTransformer):
    def apply(self, module, name, full_name):
        module.reg_loss = types.MethodType(bsconv.pytorch.BSConvS_ModelRegLossMixin.reg_loss, module)
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
            type(self.filter).__name__,
            type(self.transformer).__name__,
        )


class ModuleReplacer():
    def __init__(self, verbosity=0):
        self.verbosity = verbosity
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
        # get parameter count before the replacement
        if self.verbosity >= 1:
            parameter_count_before = bsconv.pytorch.ModelProfiler.count_parameters(module=module)

        # apply replacement rules to the root module, and then recursively to all child modules
        (root_replaced_count, module) = self._apply_rules(module=module, name="", full_name="")
        (replaced_count, module) = self._apply_recursively(module=module, name_prefix="")

        # print info about the replacement
        if self.verbosity >= 1:
            total_replaced_count = replaced_count + root_replaced_count
            print("{} altered a total of {} module{}".format(
                type(self).__name__,
                total_replaced_count,
                "" if total_replaced_count == 1 else "s",
            ))
            parameter_count_after = bsconv.pytorch.ModelProfiler.count_parameters(module=module)
            print("Model parameters changed from {} to {} (= {:.2f}%)".format(
                bsconv.utils.human_readable_int(parameter_count_before),
                bsconv.utils.human_readable_int(parameter_count_after),
                100.0 * parameter_count_after / parameter_count_before,
            ))

        return module

    def _apply_rules(self, module, name, full_name):
        for rule in self.rules:
            if rule.filter.apply(module=module, name=name, full_name=full_name):
                # if filter matches, apply transform to module
                old_type_name = type(module).__name__
                module = rule.transformer.apply(module=module, name=name, full_name=full_name)
                if self.verbosity >= 2:
                    print("{} altered '{}': {} => {}".format(
                        type(self).__name__,
                        full_name if full_name != "" else "(root)",
                        old_type_name,
                        type(module).__name__,
                    ))
                return (1, module)

        # signal that no rule was applied
        return (0, module)

    def _apply_recursively(self, module, name_prefix):
        named_children = list(module.named_children())
        replaced_count = 0
        for (child_name, child) in named_children:
            if not isinstance(child, torch.nn.Module):
                continue

            # check in any rule applies to the child
            child_full_name = "{}{}".format(name_prefix, child_name)
            (child_replaced_count, new_child) = self._apply_rules(module=child, name=child_name, full_name=child_full_name)
            if child_replaced_count == 0:
                # if no rule applied, recurse into child module
                (child_replaced_count, new_child) = self._apply_recursively(module=child, name_prefix="{}{}.".format(name_prefix, child_name))
            replaced_count += child_replaced_count
            setattr(module, child_name, new_child)

        return (replaced_count, module)


class BSConvU_Replacer(ModuleReplacer):
    def __init__(self, kernel_sizes=((3, 3), (5, 5)), with_bn=False, bn_kwargs=None, **kwargs):
        super().__init__(**kwargs)
        self.add_rule(
            Conv2dFilter(kernel_sizes=kernel_sizes),
            Conv2dToBSConvUTransformer(with_bn=with_bn, bn_kwargs=bn_kwargs),
        )


class BSConvS_Replacer(ModuleReplacer):
    def __init__(self, kernel_sizes=((3, 3), (5, 5)), p=0.25, with_bn=False, bn_kwargs=None, **kwargs):
        super().__init__(**kwargs)
        self.add_rule(
            Conv2dFilter(kernel_sizes=kernel_sizes),
            BSConvSTransformer(p=p, with_bn=with_bn, bn_kwargs=bn_kwargs),
        )
        self.add_rule(
            ModelFilter(),
            RegularizationMethodTransformer(),
        )

