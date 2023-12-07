import torch
import numpy as np
from megatron import get_args

class TeraPipeBackwardPassHook(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, outputs, cache_inputs, cache_outputs,  seq_slices=None,  sequence_dim=0, cat_outputs=False):
        # batch_slices and seq_slices are lists of integers, reserved for later update.
        ctx.outputs = outputs
        ctx.cache_inputs = cache_inputs
        ctx.cache_outputs = cache_outputs
        ctx.seq_slices = seq_slices
        ctx.sequence_dim = sequence_dim
        ctx.cat_outputs = cat_outputs
        return y

    @staticmethod
    def backward(ctx, grad_y):
        args = get_args()
        # 获取当前pipeline的batch_slices和seq_slices的数量
        n_batch_slices = 1 if ctx.batch_slices is None else len(ctx.batch_slices)
        assert  args.seq_lengh % args.terapipe_slice_len == 0, 'seq_lengh must be divisible by terapipe_slice_len'
        n_input_slices = int(args.seq_lengh // args.terapipe_slice_len)
        #  如果在最后一个pipeline阶段连接输出，将grad_y切分为不同的batch和seq slices
        if ctx.cat_outputs:
            # grad_outputs = grid_slice_batch_and_sequence(
            #     grad_y, batch_slices=ctx.batch_slices, seq_slices=ctx.seq_slices,
            #     batch_dim=ctx.batch_dim, sequence_dim=ctx.sequence_dim, requires_grad=False)
            grad_outputs = grid_slice_sequence_equally(grad_y, args.terapipe_slice_len, requires_grad=False, sequence_dim=ctx.sequence_dim)
        else:
            grad_outputs = np.empty((n_input_slices), dtype='O')
            for i in range(n_input_slices):
                grad_outputs[i] = torch.empty_like(ctx.outputs[i])
        da = []
        for input_id in reversed(range(n_input_slices)):
            y = ctx.outputs[input_id]
            dy = grad_outputs[input_id]
            if input_id < n_input_slices - 1:
                a = ctx.cache_outputs[input_id]
            else:
                a = []

            torch.autograd.backward([y] + a, [dy] + da)
            da = [t.grad for t in ctx.cache_inputs[input_id]]

        del ctx.outputs
        del ctx.cache_inputs
        del ctx.cache_outputs
        del ctx.batch_slices
        del ctx.seq_slices
        del ctx.batch_dim
        del ctx.sequence_dim
        del ctx.cat_outputs
        return None, None, None, None, None, None, None, None, None

def recv_backward():
    pass

def send_backward():
    pass

def backward_step():
    pass

def grid_slice_sequence_equally(x, seq_slice_len, requires_grad=False, sequence_dim=0):
    # 把x按照seq_slice_len进行切分
    seq_len = x.size(sequence_dim)
    n_slices = int(seq_len // seq_slice_len)
    sliced = np.empty(n_slices, dtype='O')
    start_index = 0
    for i in range(n_slices):
        index = torch.arange(start_index, start_index + seq_slice_len, device=x.device)
        sliced[i] = (x.index_select(sequence_dim, index).detach().contiguous().requires_grad_(requires_grad))
        start_index += seq_slice_len
    assert start_index == seq_len
    return sliced


def terapipe_backward_hook(outputs, cache_inputs, cache_outputs, seq_slices, sequence_dim=0, cat_outputs=False):
    with torch.no_grad():
        if cat_outputs:
            y = torch.cat(outputs, dim=sequence_dim)
        else:
            y = torch.tensor(0.0)
    y.requires_grad_()
    y = TeraPipeBackwardPassHook.apply(y, outputs, cache_inputs, cache_outputs, seq_slices, sequence_dim, cat_outputs)
    return y
