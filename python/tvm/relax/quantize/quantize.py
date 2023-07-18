import numpy as np

import tvm
from tvm import relax
from tvm.runtime import Object

from . import _quantize


@tvm._ffi.register_object("relax.quantize.SmoothQuantConfig")
class SmoothQuantConfig(Object):
    _node_defaults = {
        "dtype_activation": "int8",
        "dtype_weight": "int8",
        "alpha": 0.5,
    }

    def __enter__(self):
        # pylint: disable=protected-access
        _quantize._EnterSmoothQuantConfigScope(self)
        return self

    def __exit__(self, ptype, value, trace):
        _quantize._ExitSmoothQuantConfigScope()

def sqconfig(**kwargs):
    node_args = {k: v if k not in kwargs else kwargs[k] for k, v in SmoothQuantConfig._node_defaults.items()}
    return tvm.ir.make_node("relax.quantize.SmoothQuantConfig", **node_args)


def current_qconfig():
    """Get the current quantization configuration."""
    return _quantize._GetCurrentSmoothQuantConfig()


def get_runtime_func(funcs, mod, target):
    lowered_mod = relax.transform.LegalizeOps()(mod)

    target_kind = target.kind.default_keys[0]
    vm_device = tvm.device(target_kind)
    if target_kind != "cpu":
        with target:
            lowered_mod = tvm.tir.transform.DefaultGPUSchedule()(lowered_mod)

    exe = relax.build(lowered_mod, target)
    vm = relax.VirtualMachine(exe, vm_device)

    runtime_funcs = []
    for fname in funcs:
        runtime_funcs.append(vm[fname])

    return runtime_funcs[0] if len(funcs) == 1 else runtime_funcs


def _accumulate_outlier_stat(stat, data):
    if stat is None:
        stat = [[] for i in range(len(data))]
    for idx, out in enumerate(data):
        stat[idx].append(out.numpy())
    return stat


def _accumulate_act_outlier_stat(stat, data):
    a_data = data[::2]
    return _accumulate_outlier_stat(stat, a_data)


def _accumulate_weight_outlier_stat(stat, data):
    # Optimization step: no need to accumulate weights for each new element in dataset since
    # weights are the same.
    if stat is not None:
        return stat
    w_data = data[1::2]
    return _accumulate_outlier_stat(stat, w_data)


def _calculate_scale_params(func_name, stats, dev):
    if stats[func_name] is None:
        return {}

    cfg = current_qconfig()

    # scales = act_scales.pow(alpha) / weight_scales.pow(1-alpha)
    a_stat, w_stat = stats[func_name]
    assert len(a_stat) == len(w_stat)

    idx = 0
    scale_params = {}
    for a_element, w_elemet in zip(a_stat, w_stat):
        assert a_element.shape == w_elemet.shape
        assert len(a_element.shape) == 1
        scales = np.power(a_element, cfg.alpha) / np.power(w_elemet, 1 - cfg.alpha)
        if scales.size - np.count_nonzero(scales) > 0:
            print("Warning: Smoothing: scales have zero value")
            scales = np.ones_like(scales)
            assert False, "Not supported case"
        scale_params[f"sq_scale_{idx}"] = tvm.nd.array(scales, dev)
        scale_params[f"sq_scale_{idx+1}"] = tvm.nd.array(scales, dev)
        idx += 2

    return scale_params

def _calculate_quant_scale_params(func_name, stats, dev):
    if stats[func_name] is None:
        return {}

    cfg = current_qconfig()

    idx = 0
    scale_params = {}
    for a_element, w_element in zip(*stats[func_name]):
        a_scale = np.max(a_element) / a_element.dtype.type(np.iinfo(cfg.dtype_activation).max)
        w_scale = np.max(w_element) / w_element.dtype.type(np.iinfo(cfg.dtype_weight).max)
        scale_params[f"sq_scale_{idx}"] = tvm.nd.array(a_scale, dev)
        scale_params[f"sq_scale_{idx+1}"] = tvm.nd.array(w_scale, dev)
        idx += 2

    return scale_params


def smooth(mod, params, funcs, dataset, extra_passes=None):
    mod = relax.transform.Annotate(funcs)(mod)
    mod = relax.transform.DeadCodeElimination(funcs)(mod)
    #print("\n\nAfter annotate:\n", mod)

    stat_mod = relax.transform.CollectStat()(mod)
    #print("\n\nAfter colect stat:\n", stat_mod)

    # Run extra passes
    if extra_passes is not None:
        if not isinstance(extra_passes, (list, tuple)):
            extra_passes = [extra_passes]
        seq = tvm.transform.Sequential(extra_passes)
        stat_mod = seq(stat_mod)

    target = tvm.target.Target.current(allow_none=False)
    print(f"Smoothing: used target for statistics collection: {target}")
    #f = get_runtime_func(funcs, stat_mod, target)
    kvc, prefill, decode, _, _ = get_runtime_func(funcs, stat_mod, target)

    # Calculate max statistics
    # Number of dimension in a_stat/w_stat is equal to 3, where:
    #  * 1st dimension - number of outputs in compute graph / 2
    #  * 2nd dimension - number of elements in dataset
    #  * 3rd dimension - scale(multiplier) dimension.
    #a_stat = None
    #w_stat = None

    a_stat_prefill = None
    w_stat_prefill = None
    a_stat_decode = None
    w_stat_decode = None

    for data in dataset:
        """
        _, outputs = f(data, params["weight"])
        a_stat = _accumulate_act_outlier_stat(a_stat, outputs)
        w_stat = _accumulate_weight_outlier_stat(w_stat, outputs)
        """
        kv_caches = kvc()
        prefill_input, seq_len_shape, first_sampled_token, second_seq_len_shape = data
        print("  Run prefill...")
        (logits, kv_caches), outputs = prefill(prefill_input, seq_len_shape, kv_caches, *params)

        a_stat_prefill = _accumulate_act_outlier_stat(a_stat_prefill, outputs)
        w_stat_prefill = _accumulate_weight_outlier_stat(w_stat_prefill, outputs)

        print("  Run decode...")
        (logits, kv_caches), outputs = decode(first_sampled_token, second_seq_len_shape, kv_caches, *params)

        print("  Run smooth stat accumulation...")
        a_stat_decode = _accumulate_act_outlier_stat(a_stat_decode, outputs)
        w_stat_decode = _accumulate_weight_outlier_stat(w_stat_decode, outputs)

    """
    a_stat = [np.max(s, axis=0) for s in a_stat]
    w_stat = [np.max(s, axis=0) for s in w_stat]
    stat = {"main": (a_stat, w_stat)}
    """

    a_stat_prefill = [np.max(s, axis=0) for s in a_stat_prefill]
    w_stat_prefill = [np.max(s, axis=0) for s in w_stat_prefill]

    a_stat_decode = [np.max(s, axis=0) for s in a_stat_decode]
    w_stat_decode = [np.max(s, axis=0) for s in w_stat_decode]

    stat = dict.fromkeys(funcs)
    stat["prefill"] = (a_stat_prefill, w_stat_prefill)
    stat["decode"] = (a_stat_decode, w_stat_decode)

    for fname in funcs:
        scale_params = _calculate_scale_params(fname, stat, tvm.cpu(0))
        mod = relax.transform.BindParams(fname, scale_params)(mod)

    mod = relax.transform.SmoothQuantLegalize("multiply")(mod)

    return mod


def quantize(mod, params, funcs, dataset, extra_passes=None):
    mod = relax.transform.Annotate(funcs, "quantize")(mod)
    mod = relax.transform.DeadCodeElimination(funcs)(mod)
    #print("\n\nAfter annotate in Calibration:\n", mod)

    stat_mod = relax.transform.CollectStat()(mod)
    #print("\n\nAfter colect stat in Calibration:\n", stat_mod)

    # Run extra passes
    if extra_passes is not None:
        if not isinstance(extra_passes, (list, tuple)):
            extra_passes = [extra_passes]
        seq = tvm.transform.Sequential(extra_passes)
        stat_mod = seq(stat_mod)

    target = tvm.target.Target.current(allow_none=False)
    print(f"Quantization: used target for statistics collection: {target}")
    #f = get_runtime_func(funcs, stat_mod, target)
    kvc, prefill, decode, _, _ = get_runtime_func(funcs, stat_mod, target)

    #a_stat = None
    #w_stat = None

    a_stat_prefill = None
    w_stat_prefill = None
    a_stat_decode = None
    w_stat_decode = None

    for data in dataset:
        """
        _, outputs = f(data, params["weight"])
        a_stat = _accumulate_act_outlier_stat(a_stat, outputs)
        w_stat = _accumulate_weight_outlier_stat(w_stat, outputs)
        """
        kv_caches = kvc()
        prefill_input, seq_len_shape, first_sampled_token, second_seq_len_shape = data
        print("  Run prefill...")
        (logits, kv_caches), outputs = prefill(prefill_input, seq_len_shape, kv_caches, *params)

        a_stat_prefill = _accumulate_act_outlier_stat(a_stat_prefill, outputs)
        w_stat_prefill = _accumulate_weight_outlier_stat(w_stat_prefill, outputs)

        print("  Run decode...")
        (logits, kv_caches), outputs = decode(first_sampled_token, second_seq_len_shape, kv_caches, *params)

        print("  Run quantize stat accumulation...")
        a_stat_decode = _accumulate_act_outlier_stat(a_stat_decode, outputs)
        w_stat_decode = _accumulate_weight_outlier_stat(w_stat_decode, outputs)


    """
    a_stat = [np.max(s, axis=0) for s in a_stat]
    w_stat = [np.max(s, axis=0) for s in w_stat]
    stat = {"main": (a_stat, w_stat)}
    """
    a_stat_prefill = [np.max(s, axis=0) for s in a_stat_prefill]
    w_stat_prefill = [np.max(s, axis=0) for s in w_stat_prefill]

    a_stat_decode = [np.max(s, axis=0) for s in a_stat_decode]
    w_stat_decode = [np.max(s, axis=0) for s in w_stat_decode]

    stat = dict.fromkeys(funcs)
    stat["prefill"] = (a_stat_prefill, w_stat_prefill)
    stat["decode"] = (a_stat_decode, w_stat_decode)


    for fname in funcs:
        scale_params = _calculate_quant_scale_params(fname, stat, tvm.cpu(0))
        mod = relax.transform.BindParams(fname, scale_params)(mod)

    legalized_mod = relax.transform.SmoothQuantLegalize("quantize")(mod)
    mod = relax.transform.SmoothQuantRealize()(legalized_mod)
    mod = relax.transform.DeadCodeElimination(funcs)(mod)
    mod = relax.transform.SmoothQuantReshapeMatmul()(mod)

    return mod
