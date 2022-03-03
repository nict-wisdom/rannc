import copy
import inspect
import logging
from collections import OrderedDict

import torch
import torch.cuda
import torch.onnx.utils
import torch.random

from . import _pyrannc, utils
from .dist_param import store_dist_param, load_dist_param, set_dist_param, get_dist_param_range, set_dist_param_dtype, \
    DistributeModelParams
from .opt import patch_optimizer

# Run backward to set python engine as the default engine
x = torch.randn(2, 2, requires_grad=True)
tgt = torch.randn(2, 2)
y = x * 2
y.backward(tgt)

# for better reproducibility
# https://pytorch.org/docs/stable/notes/randomness.html
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

logger = logging.getLogger("rannc")
logger.addHandler(logging.NullHandler())
logger.setLevel(logging.DEBUG)
logger.propagate = True

rannc = _pyrannc.get_rannc()
rannc.start()


def get_rank():
    r"""
    Get rank of the running process in ``COMM_WORLD``.

    :return: rank
    """
    return _pyrannc.get_rank()


def get_world_size():
    r"""
    Get the size of ``COMM_WORLD``.

    :return: world size
    """
    return _pyrannc.get_world_size()


def barrier():
    r"""
    Blocks until all ranks reaches the call of this method.
    """
    _pyrannc.barrier()


def clear():
    r"""
    Clear RaNNC's state including all RaNNCModules and buffers
    """
    _pyrannc.clear()


def delay_grad_allreduce(delay):
    r"""
    As default, RaNNC performs *allreduce* of gradients soon after ``backward``.
    If ``True`` is given, however, it skips the *allreduce*.
    The application can use ``allreduce_grads`` to explicitly perform allreduce.
    This is useful when the gradient accumulation is used.

    :param delay: If ``True``, allreduce after backward is skipped.
    """
    _pyrannc.delay_grad_allreduce(delay)


def keep_graph(keep):
    r"""
    The flag is passed to ``retain_graph`` of PyTorch's backward.
    This is useful when you perform multiple backward passes after one forward pass.

    :param keep: Set ``True`` to keep graph after backward.
    """
    _pyrannc.keep_graph(keep)


def sync_params_on_init(sync):
    """
    As default, RaNNC synchronizes model parameters on initialization.
    This aims to use same initial values of parameters on all ranks, but often takes a long time.
    You can skip the synchronization by passing ``False`` to this method when
    you use the same random seed or other libraries to synchronize parameters.

    :param sync: Set ``False`` to skip parameter synchronization.
    """
    _pyrannc.sync_params_on_init(sync)


def _dump_events():
    _pyrannc.dump_events()


def _create_interpreter_name_lookup_fn(frames_up=1):
    def _get_interpreter_name_for_var(var):
        frame = inspect.currentframe()
        i = 0
        while i < frames_up + 1:
            frame = frame.f_back
            i += 1

        f_locals = frame.f_locals
        f_globals = frame.f_globals

        for k, v in f_locals.items():
            if isinstance(v, torch.Tensor) and var is v:
                return k if k != 'self' else ''
        for k, v in f_globals.items():
            if isinstance(v, torch.Tensor) and var is v:
                return k if k != 'self' else ''
        return ''

    return _get_interpreter_name_for_var


def _stash_state_dict_hooks(model):
    hooks = {}
    for name, module in model._modules.items():
        if module is not None:
            sub_hooks = _stash_state_dict_hooks(module)
            hooks.update(sub_hooks)
    hooks[model] = model._state_dict_hooks
    model._state_dict_hooks = OrderedDict()
    return hooks


def _unstash_state_dict_hooks(model, hooks):
    for name, module in model._modules.items():
        if module is not None:
            _unstash_state_dict_hooks(module, hooks)
    if model in hooks:
        model._state_dict_hooks = hooks[model]


def _check_input_tensors(args):
    for a in args:
        if torch.is_tensor(a) and not a.is_cuda:
            raise ValueError("All inputs to RaNNCModule must be on a CUDA device.")


def _set_hooks_for_tracing(model, device):
    cpu_params = {}

    # Move cpu tensors onto a cuda device
    def _pre_hook_for_tracing(_model, input):
        _pyrannc.set_tracing_state(False)
        cpu_params[_model] = []
        for n, p in _model.named_parameters(recurse=False):
            if not p.is_cuda:
                cpu_params[_model].append(p)
                utils._to_in_place([p], device)
        for n, b in _model.named_buffers(recurse=False):
            if not b.is_cuda:
                cpu_params[_model].append(b)
                utils._to_in_place([b], device)
        _pyrannc.set_tracing_state(True)
        return input

    # Move tensors back to host
    def _hook_for_tracing(_model, input, output):
        _pyrannc.set_tracing_state(False)
        for p in cpu_params[_model]:
            utils._to_in_place([p], torch.device("cpu"))
        _pyrannc.set_tracing_state(True)

    handles = []
    for name, _module in model.named_modules():
        handles.append(_module.register_forward_pre_hook(_pre_hook_for_tracing))
        handles.append(_module.register_forward_hook(_hook_for_tracing))
    return handles


def _unset_hooks_for_tracing(handles):
    for h in handles:
        h.remove()


class RaNNCModule(_pyrannc.RaNNCModule):
    r"""
    Computes a PyTorch model on multiple GPUs with a hybrid parallelism.
    """

    def __init__(self, model, optimizer=None, gather_inputs=True, load_deployment=None, enable_apex_amp=False,
                 allreduce_amp_master_params=False, enable_zero=False, check_unused_values=True, offload_params=False):
        r"""
        :param model: Model to distribute.
        :param optimizer: Optimizer that should work with RaNNC.
        :param gather_inputs: Set ``False`` if model uses inputs given on rank 0.
        :param enable_apex_amp: Set ``True`` if ``model`` is processed by `Apex AMP <https://nvidia.github.io/apex/amp.html>`_.
        :param allreduce_amp_master_params: Set ``True`` to allreduce gradients of master parameters of Apex AMP.
        :param enable_zero: Set ``True`` to remove the redundancy of optimizer states following the approach of DeepSpeed.
        :param check_unused_values: If ``True``, RaNNC throws an exception when it finds unused values in a computation graph.
        :param offload_params: If ``True``, parameters are moved to host memory until they are used.
        """

        old_flag = torch._C._jit_set_profiling_executor(True)
        if not old_flag:
            logger.warning("RaNNC set torch._C._jit_set_profiling_executor(True).")

        self.ready = False
        self.is_training = True

        # preserve model
        self.model = model

        # rannc module removes unnecessary parameters in optimizer
        self.optimizer = optimizer

        self.gather_inputs = gather_inputs
        self.amp_master_param_registered = False
        self.load_deployment = load_deployment
        self.allreduce_amp_master_params = allreduce_amp_master_params
        self.enable_apex_amp = enable_apex_amp
        self.enable_zero = enable_zero
        self.offload_params = offload_params

        self.name_to_param = {n: p for n, p in self.model.named_parameters()}
        self.name_to_pid = {n: id(p) for n, p in self.model.named_parameters()}
        self.shared_param_names = self._shared_param_names()

        super(RaNNCModule, self).__init__(enable_apex_amp, allreduce_amp_master_params, enable_zero,
                                          check_unused_values, offload_params)

    def __call__(self, *args, **kwargs):
        if len(kwargs) > 0:
            raise ValueError("RaNNCModule does not support kwargs.")
        _check_input_tensors(args)

        if not self.ready:
            parameters = [(n, p) for n, p in self.model.named_parameters()]
            buffers = [(n, p) for n, p in self.model.named_buffers()]
            self.var_lookup_fn = _create_interpreter_name_lookup_fn(0)

            if self.load_deployment:
                super().load_deployment(self.load_deployment)

            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

            # Stash buffer values
            with torch.no_grad():
                buffers_clone = [b.clone() for b in self.model.buffers()]

            # Restore rng state
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                hook_handles = _set_hooks_for_tracing(self.model, self.device)
                self.used_param_ids = super().init(self.model.forward, parameters, buffers, self.var_lookup_fn,
                                                   self.gather_inputs, *args)
                _unset_hooks_for_tracing(hook_handles)

            # Restore buffer values
            with torch.no_grad():
                for b, b_clone in zip(self.model.buffers(), buffers_clone):
                    b.copy_(b_clone)

            utils._to_in_place([p for p in self.model.parameters() if id(p) in self.used_param_ids], self.device)
            utils._to_in_place([b for b in self.model.buffers() if id(b) in self.used_param_ids], self.device)

            # Remove parameters from optimizer
            if self.optimizer and self.model.training:
                patch_optimizer(self, self.optimizer)

            self.ready = True
            self.dummy_input = args

        out = super().__call__(*args)

        if self.enable_apex_amp:
            def setup_amp(grad):
                self._setup_amp_params()
                return grad

            out.register_hook(setup_amp)

        return out

    def to(self, *args, **kwargs):
        r"""
        This does not work because the device placement of a ``RaNNCModule`` is controlled by RaNNC.
        """
        logger.warning("to() was ignored. A RaNNC model cannot be moved across devices.")
        return self

    def cuda(self, *args, **kwargs):
        r"""
        This does not work because the device placement of a ``RaNNCModule`` is controlled by RaNNC.
        """
        logger.warning("cuda() was ignored. A RaNNC model cannot be moved across devices.")
        return self

    def train(self, mode=True):
        r"""
        Outputs warning because a RaNNC module cannot change the grad mode.

        :param mode: Training mode.
        """
        if self.model.training != mode:
            logger.warning("Unable to change grad mode of RaNNC module. Use enable_dropout to enable/disable dropout.")

    def eval(self):
        r"""
        Sets the training mode to ``False`` (i.e. evaluation mode).
        """
        self.train(mode=False)

    def parameters(self, *args, **kwargs):
        r"""
        Returns parameters. Note that parameters are not synchronized among ranks.
        """
        if not self.ready:
            return self.model.parameters(*args, **kwargs)
        return self._param_gen(self.model.parameters, *args, **kwargs)

    def named_parameters(self, *args, **kwargs):
        r"""
        Returns parameters with their names. Note that parameters are not synchronized among ranks.
        """
        if not self.ready:
            return self.model.named_parameters(*args, **kwargs)
        return self._named_param_gen(self.model.named_parameters, *args, **kwargs)

    def buffers(self, *args, **kwargs):
        r"""
        Returns buffers. Note that buffers are not synchronized among ranks.
        """
        if not self.ready:
            return self.model.buffers(*args, **kwargs)
        return self._param_gen(self.model.buffers, *args, **kwargs)

    def named_buffers(self, *args, **kwargs):
        r"""
        Returns buffers with their names. Note that buffers are not synchronized among ranks.
        """
        if not self.ready:
            return self.model.named_buffers(*args, **kwargs)
        return self._named_param_gen(self.model.named_buffers, *args, **kwargs)

    def _setup_amp_params(self):
        if not self.amp_master_param_registered:
            from .amp import patch_amp_scaler, register_amp_params
            register_amp_params(self.optimizer)
            patch_amp_scaler()
            self.amp_master_param_registered = True

    def clip_grad_norm(self, max_grad_norm):
        r"""
        Clips gradients according to the norm. Use this method to clip gradients insted of
        ``torch.nn.utils.clip_grad_norm_`` because each local process only has a part of parameters/gradients.
        This method calculates norm of all distributed gradients and clips them.

        :param max_grad_norm: Max value of gradients' norm.

        .. note::
            This method must be called from all ranks.
        """
        if self.enable_apex_amp:
            self._setup_amp_params()
        super().clip_grad_norm(max_grad_norm)

    def _calc_grad_norm(self):
        if self.enable_apex_amp:
            self._setup_amp_params()
        return super().calc_grad_norm()

    def state_dict(self, *args, no_hook=False, amp_master_params=True, **kwargs):
        r"""
        Returns ``state_dict`` of the model.

        :param no_hook: If ``True``, hooks on ``state_dict`` of the original models are ignored.
        :param amp_master_params: Set ``True`` to get apex amp master params.

        .. note::
            This method must be called from all ranks.
        """
        if not self.ready:
            return self.model.state_dict(*args, **kwargs)

        # amp O2 hook converts params to fp32
        # This may cause oom
        if no_hook:
            stashed_hooks = _stash_state_dict_hooks(self.model)

        new_state_dict = {}
        shared_params = []
        # self.name_to_param contains only one of parameters that share the same data.
        # (due to the behavior of named_parameters())
        for n, p in self.model.state_dict(*args, **kwargs).items():
            if n in self.name_to_param:
                new_state_dict[n] = self.get_param(n, amp_master_params and self.enable_apex_amp)
            else:
                shared_params.append(n)
        # Set shared parameters with other names
        for name_set in self.shared_param_names:
            stored_names = [n for n in name_set if n in new_state_dict]
            assert (len(stored_names) == 1)
            stored_name = stored_names[0]
            for n in name_set:
                if n != stored_name:
                    new_state_dict[n] = new_state_dict[stored_name]

        if no_hook:
            _unstash_state_dict_hooks(self.model, stashed_hooks)
        return new_state_dict

    def load_state_dict(self, *args, **kwargs):
        r"""
        Load ``state_dict`` to the model. This works only before the first call of forward pass.

        :param args: Passed to the original model.
        :param kwargs: Passed to the original model.
        :return: Return value of the original model's ``state_dict``.
        """
        if self.ready:
            self.undeploy()
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            self.model = copy.deepcopy(self.orig_model).to(device)

        return self.model.load_state_dict(*args, **kwargs)

    def allreduce_grads(self):
        r"""
        Performs *allreduce* on gradients of model parameters.

        .. note::
            This method must be called from all ranks.
        """
        if self.enable_apex_amp:
            self._setup_amp_params()
        super().allreduce_grads()

    def zero_grad(self):
        r"""
        Sets zeros to  gradients of model parameters.
        """
        super().zero_grad()

    def get_param(self, name, amp_master_param=False):
        if name not in self.name_to_pid or name not in self.name_to_param:
            raise RuntimeError("No parameter found: {}".format(name))

        if self.ready:
            return super().get_param(self.name_to_pid[name], amp_master_param)
        return self.name_to_param[name]

    def get_param_grad(self, name, amp_master_param=False):
        if name not in self.name_to_pid or name not in self.name_to_param:
            raise RuntimeError("No parameter found: {}".format(name))

        if self.ready:
            return super().get_param_grad(self.name_to_pid[name], amp_master_param)
        return self.name_to_param[name].grad

    def save_deployment(self, file):
        r"""
        Saves a deployment state (graph partitioning) to file.

        :param file: File path.
        """
        if not self.ready:
            raise RuntimeError("Failed to save deployment. Module is not ready.")

        if _pyrannc.get_rank() == 0:
            super().save_deployment(file)
        else:
            logger.warning("save_deployment works only on rank 0")

    def undeploy(self):
        r"""
        Undeploys a model distributed on GPUs. This frees GPU memory used for the model.

        .. note::
            This method must be called from all ranks.
        """
        if self.ready:
            super().undeploy()

    def enable_dropout(self, enable):
        if self.ready:
            super().enable_dropout(enable)
        else:
            logger.warning("Unable to change dropout mode because the module is not initialized.")

    def __del__(self):
        self.undeploy()

    def __getattr__(self, attr):
        if not hasattr(self.model, attr):
            raise AttributeError("This model has no attribute '{}'".format(attr))

        model_attr = getattr(self.model, attr)

        def wrapper_func(*args, **kwargs):
            return model_attr(*args, **kwargs)

        if callable(model_attr):
            return wrapper_func
        return model_attr

    def _sync_orig_params(self, sync_all_ranks=False, sync_grad=False, name_pattern=None):
        if not self.ready:
            return

        if self.enable_zero:
            self.sync_param_zero(sync_grad)

        for name in sorted(self.name_to_param.keys()):
            if name_pattern is not None and name_pattern not in name:
                continue

            pid = self.name_to_pid[name]
            param = self.name_to_param[name]
            synced_param_cpu = self.sync_param(pid)
            if synced_param_cpu is not None:
                if _pyrannc.get_rank() == 0 or sync_all_ranks:
                    with torch.no_grad():
                        if (hasattr(param,
                                    "distributed") and param.distributed) and param.size() != synced_param_cpu.size():
                            # This param is a dummy. It was originally distributed across all ranks  and then removed
                            # on this rank because the subgraph on this rank does not need this parameter.
                            param.data = synced_param_cpu
                        else:
                            param.copy_(synced_param_cpu.view(param.size()))
            if sync_grad:
                synced_param_grad_cpu = self.sync_param_grad(pid)
                if synced_param_grad_cpu is not None or sync_all_ranks:
                    if _pyrannc.get_rank() == 0:
                        with torch.no_grad():
                            if param.grad is None:
                                param.grad = torch.zeros_like(param)
                            param.grad.copy_(synced_param_grad_cpu.view(param.size()))

        _pyrannc.barrier()

    def _param_gen(self, f, *args, from_global=False, **kwargs):
        for p in f(*args, **kwargs):
            if from_global or id(p) in self.used_param_ids:
                yield p

    def _named_param_gen(self, f, *args, from_global=False, **kwargs):
        for n, p in f(*args, **kwargs):
            if from_global or id(p) in self.used_param_ids:
                yield n, p

    def _shared_param_names(self):
        pid_to_names = {}
        for prefix, module in self.named_modules():
            for n, p in module._parameters.items():
                if p is not None:
                    name = prefix + ('.' if prefix else '') + n
                    if id(p) not in pid_to_names:
                        pid_to_names[id(p)] = set()
                    pid_to_names[id(p)].add(name)
        return list(pid_to_names.values())

def allreduce_grads(rmodel, optimizer, prescale=1.0):
    if rmodel.enable_apex_amp:
        rmodel._setup_amp_params()
        from .amp import allreduce_grads_amp
        return allreduce_grads_amp(rmodel, optimizer, prescale)

    if rmodel.enable_zero:
        rmodel.allreduce_grads_zero(prescale)
    else:
        with torch.no_grad():
            for p in rmodel.parameters():
                if p.grad is None:
                    continue
                p.grad.mul_(prescale)
        rmodel.allreduce_grads()

    return False


def _run_dp_dry(path):
    _pyrannc.run_dp_dry(path)


def recreate_all_communicators():
    _pyrannc.recreate_all_communicators()


def show_deployment(path, batch_size):
    """
    Show a deployment (Subgraphs and micro-batch sizes in pipeline parallelism) saved in a file.
    This is used for debugging.

    :param path: Path to a deployment file.
    :param batch_size: Global batch size.
    """
    _pyrannc.show_deployment(path, batch_size)
