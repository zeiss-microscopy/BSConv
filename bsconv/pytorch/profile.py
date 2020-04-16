import collections
import warnings

import torch.nn

import bsconv.utils


class ModelProfiler():
    def __init__(self, model, input_size=(1, 3, 224, 224), input_device="cpu"):
        # construct empty input tensor
        input = self.get_input(size=input_size, device=input_device)

        # register hooks, forward input tensor, and collect results
        self.model = model
        self.hook_handles = []
        self.register_hooks()
        model(input)
        self.collect_results()
        self.unregister_hooks()

    def get_input(self, size, device):
        return torch.Tensor(size=size, device=device)

    @classmethod
    def _hook(cls, module, input, output):
        result = {}
        
        # basic info about the module
        result["is_leaf"] = (len(list(module.children())) == 0)
        result["type"] = type(module).__name__
        if hasattr(module, "kernel_size"):
            if isinstance(module.kernel_size, int):
                result["info"] = "{}x{}".format(module.kernel_size, module.kernel_size)
            elif isinstance(module.kernel_size, tuple) and (len(module.kernel_size) == 2):
                result["info"] = "{}x{}".format(module.kernel_size[1], module.kernel_size[0])
                if isinstance(module, torch.nn.Conv2d):
                    if module.kernel_size == (1, 1):
                        result["info"] += " PW"
                    elif (module.groups == module.in_channels) and (module.in_channels == module.out_channels):
                        result["info"] += " DW"
        
        # input and output shape info
        try:
            result["in_shape"] = tuple(input[0].shape[1:])
        except AttributeError:
            pass
        try:
            result["out_shape"] = tuple(output.shape[1:])
        except AttributeError:
            pass

        if result["is_leaf"]:
            result["parameter_count"] = cls.count_parameters(module=module)
            result["flop_count"] = cls.count_flops(module=module, input=input, output=output)

        # store result as attribute of the module - it will be collected in a later step
        module.__profiler_result__ = result

    @staticmethod
    def count_parameters(module):
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    @classmethod
    def count_flops(cls, module, input, output):
        methods = {
            torch.nn.Conv1d: cls.count_flops_conv,
            torch.nn.Conv2d: cls.count_flops_conv,
            torch.nn.Conv3d: cls.count_flops_conv,
            torch.nn.Linear: cls.count_flops_linear,
            torch.nn.Dropout: cls.count_flops_dropout,
            torch.nn.BatchNorm1d: cls.count_flops_bn,
            torch.nn.BatchNorm2d: cls.count_flops_bn,
            torch.nn.BatchNorm3d: cls.count_flops_bn,
            torch.nn.ReLU: cls.count_flops_relu,
            torch.nn.MaxPool1d: cls.count_flops_pool,
            torch.nn.MaxPool2d: cls.count_flops_pool,
            torch.nn.MaxPool3d: cls.count_flops_pool,
            torch.nn.AvgPool1d: cls.count_flops_pool,
            torch.nn.AvgPool2d: cls.count_flops_pool,
            torch.nn.AvgPool3d: cls.count_flops_pool,
            torch.nn.AdaptiveMaxPool1d: cls.count_flops_pool,
            torch.nn.AdaptiveMaxPool2d: cls.count_flops_pool,
            torch.nn.AdaptiveMaxPool3d: cls.count_flops_pool,
            torch.nn.AdaptiveAvgPool1d: cls.count_flops_pool,
            torch.nn.AdaptiveAvgPool2d: cls.count_flops_pool,
            torch.nn.AdaptiveAvgPool3d: cls.count_flops_pool,
        }
        for (class_, method) in methods.items():
            if isinstance(module, class_):
                return method(module=module, input=input, output=output)
        else:
            warnings.warn("FLOP counting is not implemented for modules of type '{}'".format(type(module)))
            return 0

    @staticmethod
    def count_flops_conv(module, input, output):
        flops = output[:, 0, :, :].numel() * module.weight.numel()
        if module.bias is not None:
            flops += output.numel()
        return flops
    
    @staticmethod
    def count_flops_linear(module, input, output):
        flops = input[0].numel() * output[-1].numel()
        if module.bias is not None:
            flops += output.numel()
        return flops

    @staticmethod
    def count_flops_dropout(module, input, output):
        return 0

    @staticmethod
    def count_flops_bn(module, input, output):
        flops = input[0].numel()
        if module.affine:
            flops *= 2
        return flops

    @staticmethod
    def count_flops_relu(module, input, output):
        flops = output.numel()
        return flops

    @staticmethod
    def count_flops_pool(module, input, output):
        flops = input[0].numel()
        return flops

    def register_hooks(self):
        def _register_hook(module):
            hook_handle = module.register_forward_hook(self._hook)
            self.hook_handles.append(hook_handle)
        self.model.apply(_register_hook)

    def unregister_hooks(self):
        while len(self.hook_handles) > 0:
            hook_handle = self.hook_handles.pop()
            hook_handle.remove()

    def _collect_module_results(self, module, name, name_prefix):
        # construct module name
        if name_prefix == "":
            full_name = name
        else:
            full_name = "{}.{}".format(name_prefix, name)

        # collect the raw result dict and delete it from the module
        result = module.__profiler_result__
        del module.__profiler_result__

        # do the same for all children and sum up the param and flop counts
        named_children = list(module.named_children())
        if len(named_children) > 0:
            # if the module has children, then use the sum of their param and FLOP counts for this module
            assert "parameter_count" not in result.keys()
            assert "flop_count" not in result.keys()
            children_parameter_count = 0
            children_flop_count = 0
            for (child_name, child_module) in named_children:
                child_result = self._collect_module_results(module=child_module, name=child_name, name_prefix=full_name)
                children_parameter_count += child_result["parameter_count"]
                children_flop_count += child_result["flop_count"]
            result["parameter_count"] = children_parameter_count
            result["flop_count"] = children_flop_count
        
        # store the result
        self.module_results[full_name] = result

        return result
            
    def collect_results(self):
        # collect results from all modules
        self.module_results = collections.OrderedDict()
        self._collect_module_results(module=self.model, name="", name_prefix="")

        # calculate total counts
        self.total_parameter_count = sum(result["parameter_count"] for result in self.module_results.values() if result["is_leaf"])
        self.total_flop_count = sum(result["flop_count"] for result in self.module_results.values() if result["is_leaf"])
        assert self.total_parameter_count == self.module_results[""]["parameter_count"]
        assert self.total_flop_count == self.module_results[""]["flop_count"]

        # bring first entry (which represents the entire model) to the end
        if "" in self.module_results.keys():
            result = self.module_results[""]
            del self.module_results[""]
            self.module_results["(Total)"] = result

    def print_results(self):
        header = ("Module", "Type", "Info", "In Shape", "Out Shape", "Params", "FLOPs")

        # collect all rows
        rows = [header]
        for (module_name, module_result) in self.module_results.items():
            parameter_count = module_result.get("parameter_count", None)
            flop_count = module_result.get("flop_count", None)

            row = (
                module_name,
                module_result.get("type", "(unknown)"),
                module_result.get("info", ""),
                str(module_result.get("in_shape", "?")),
                str(module_result.get("out_shape", "?")),
                bsconv.utils.human_readable_int(parameter_count) if parameter_count is not None else "",
                bsconv.utils.human_readable_int(flop_count) if flop_count is not None else "",
            )
            rows.append(row)

        # determine the max length of each column
        max_lengths = [0 for _ in range(len(header))]
        for row in rows:
            for (n_col, col) in enumerate(row):
                assert isinstance(col, str)
                max_lengths[n_col] = max(max_lengths[n_col], len(col))

        sep_row_str = "+"
        for n_col in range(len(header)):
            sep_row_str += "-" * (max_lengths[n_col] + 2)
            sep_row_str += "+"

        # print
        for (n_row, row) in enumerate(rows):
            row_str = "|"
            for (n_col, col) in enumerate(row):
                row_str += " "
                if (n_row > 0) and (header[n_col] in ("Params", "FLOPs")):
                    # right-align, param and FLOP numbers
                    row_str += col.rjust(max_lengths[n_col])
                else:
                    # left-align everything else
                    row_str += col.ljust(max_lengths[n_col])
                row_str += " |"
            
            # print row (and aditional sep rows, if necessary)
            if n_row in (0, len(rows) - 1):
                print(sep_row_str)
            print(row_str)
            if n_row in (0, len(rows) - 1):
                print(sep_row_str)
