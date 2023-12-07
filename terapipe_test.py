import torch
import unittest
from megatron.model.transformer import ParallelTransformerLayer
from megatron.model.transformer_cache import ParallelTransformerLayerWithCache
# from megatron import ParallelTransformerLayer, ParallelTransformerLayerWithCache 
from megatron.model.transformer import ParallelTransformerLayer
from megatron.model.transformer_cache import ParallelTransformerLayerWithCache
from megatron.model.enums import LayerType, AttnType, AttnMaskType
from megatron.model import GPTModelTerapipe
from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron import get_tokenizer
import torch.nn.functional as F
from megatron.core.enums import ModelType
from types import SimpleNamespace
from unittest.mock import patch
from megatron.initialize import initialize_megatron
from megatron.core import mpu, tensor_parallel
from megatron import gpu_logger
from megatron.arguments import core_transformer_config_from_args
from megatron.core.utils import get_model_config
from megatron.training import setup_model_and_optimizer
import os

# 设置环境变量
os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'
os.environ['GPUS_PER_NODE'] = '1'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '6000'
os.environ['NNODES'] = '1'
os.environ['NODE_RANK'] = '0'

# 计算 WORLD_SIZE 并设置
gp_per_node = int(os.environ['GPUS_PER_NODE'])
num_nodes = int(os.environ['NNODES'])
os.environ['WORLD_SIZE'] = str(gp_per_node * num_nodes)

# 确认环境变量设置
print("WORLD_SIZE:", os.environ['WORLD_SIZE'])



args = {
"accumulate_allreduce_grads_in_fp32":False,
"adam_beta1":0.9,
"adam_beta2":0.999,
"adam_eps":1e-08,
"add_bias_linear":True,
"add_position_embedding":True,
"adlr_autoresume":False,
"adlr_autoresume_interval":1000,
"apply_layernorm_1p":False,
"apply_query_key_layer_scaling":True,
"apply_residual_connection_post_layernorm":False,
"async_tensor_model_parallel_allreduce":True,
"attention_dropout":0.1,
"attention_softmax_in_fp32":False,
"barrier_with_L1_time":True,
"bert_binary_head":True,
"bert_embedder_type":'megatron',
"bert_load":None,
"bf16":False,
"bias_dropout_fusion":True,
"bias_gelu_fusion":True,
"biencoder_projection_dim":0,
"biencoder_shared_query_context_model":False,
"block_data_path":None,
"check_for_nan_in_loss_and_grad":True,
"classes_fraction":1.0,
"clip_grad":1.0,
"consumed_train_samples":0,
"consumed_valid_samples":0,
"data_cache_path":None,
"data_parallel_random_init":False,
"data_parallel_size":1,
"data_path":['/home/tangjingqi/Megatron-LM/my-100-sample_text_document'],
"data_per_class_fraction":1.0,
"data_sharding":True,
"dataloader_type":'single',
"decoder_num_layers":None,
"decoder_seq_length":None,
"dino_bottleneck_size":256,
"dino_freeze_last_layer":1,
"dino_head_hidden_size":2048,
"dino_local_crops_number":10,
"dino_local_img_size":96,
"dino_norm_last_layer":False,
"dino_teacher_temp":0.07,
"dino_warmup_teacher_temp":0.04,
"dino_warmup_teacher_temp_epochs":30,
"distribute_saved_activations":False,
"distributed_backend":'nccl',
"distributed_timeout_minutes":10,
"embedding_path":None,
"embedding_weights_in_fp32":False,
"empty_unused_memory_level":0,
"encoder_num_layers":24,
"encoder_seq_length":1024,
"end_weight_decay":0.01,
"eod_mask_loss":False,
"eval_interval":1000,
"eval_iters":10,
"evidence_data_path":None,
"exit_duration_in_mins":None,
"exit_interval":None,
"exit_on_missing_checkpoint":False,
"exit_signal_handler":False,
"ffn_hidden_size":4096,
"finetune":False,
"fp16":True,
"fp16_lm_cross_entropy":False,
"fp32_residual_connection":False,
"fp8":None,
"fp8_amax_compute_algo":'most_recent',
"fp8_amax_history_len":1,
"fp8_interval":1,
"fp8_margin":0,
"fp8_wgrad":True,
"global_batch_size":8,
"gradient_accumulation_fusion":True,
"group_query_attention":False,
"head_lr_mult":1.0,
"hidden_dropout":0.1,
"hidden_size":4,
"hysteresis":2,
"ict_head_size":None,
"ict_load":None,
"img_h":224,
"img_w":224,
"indexer_batch_size":128,
"indexer_log_interval":1000,
"inference_batch_times_seqlen_threshold":512,
"init_method_std":0.02,
"init_method_xavier_uniform":False,
"initial_loss_scale":4294967296,
"iter_per_epoch":1250,
"kv_channels":64,
"lazy_mpu_init":None,
"load":'/home/tangjingqi/Megatron-LM/',
"local_rank":0,
"log_batch_size_to_tensorboard":False,
"log_interval":100,
"log_learning_rate_to_tensorboard":True,
"log_loss_scale_to_tensorboard":True,
"log_memory_to_tensorboard":False,
"log_num_zeros_in_grad":False,
"log_params_norm":False,
"log_timers_to_tensorboard":False,
"log_validation_ppl_to_tensorboard":False,
"log_world_size_to_tensorboard":False,
"loss_scale":None,
"loss_scale_window":1000,
"lr":0.00015,
"lr_decay_iters":320000,
"lr_decay_samples":None,
"lr_decay_style":'cosine',
"lr_warmup_fraction":0.01,
"lr_warmup_init":0.0,
"lr_warmup_iters":0,
"lr_warmup_samples":0,
"make_vocab_size_divisible_by":128,
"mask_factor":1.0,
"mask_prob":0.15,
"mask_type":'random',
"masked_softmax_fusion":True,
"max_position_embeddings":1024,
"max_tokens_to_oom":12000,
"merge_file":'/home/tangjingqi/Megatron-LM/gpt2-merges.txt',
"micro_batch_size":4,
"min_loss_scale":1.0,
"min_lr":1e-05,
"mmap_warmup":False,
"no_load_optim":None,
"no_load_rng":None,
"no_persist_layer_norm":False,
"no_save_optim":None,
"no_save_rng":None,
"norm_epsilon":1e-05,
"normalization":'LayerNorm',
"num_attention_heads":16,
"num_channels":3,
"num_classes":1000,
"num_experts":None,
"num_layers":24,
"num_layers_per_virtual_pipeline_stage":None,
"num_query_groups":1,
"num_workers":2,
"onnx_safe":None,
"openai_gelu":False,
"optimizer":'adam',
"output_bert_embeddings":False,
"overlap_grad_reduce":False,
"overlap_p2p_comm":False,
"override_opt_param_scheduler":False,
"padded_vocab_size":50304,
"params_dtype":torch.float16,
"patch_dim":16,
"perform_initialization":True,
"pipeline_model_parallel_size":1,
"pipeline_model_parallel_split_rank":None,
"position_embedding_type":'learned_absolute',
"profile":False,
"profile_ranks":[0],
"profile_step_end":12,
"profile_step_start":10,
"query_in_block_prob":0.1,
"rampup_batch_size":None,
"rank":0,
"recompute_granularity":None,
"recompute_method":None,
"recompute_num_layers":None,
"reset_attention_mask":False,
"reset_position_ids":False,
"retriever_report_topk_accuracies":[],
"retriever_score_scaling":False,
"retriever_seq_length":256,
"retro_add_retriever":False,
"retro_cyclic_train_iters":None,
"retro_encoder_attention_dropout":0.1,
"retro_encoder_hidden_dropout":0.1,
"retro_encoder_layers":2,
"retro_num_neighbors":2,
"retro_num_retrieved_chunks":2,
"retro_return_doc_ids":False,
"retro_workdir":None,
"rotary_percent":1.0,
"rotary_seq_len_interpolation_factor":None,
"sample_rate":1.0,
"save":'/home/tangjingqi/Megatron-LM/',
"save_interval":10000,
"scatter_gather_tensors_in_pipeline":True,
"seed":1234,
"seq_length":64,
"sequence_parallel":False,
"sgd_momentum":0.9,
"short_seq_prob":0.1,
"skip_train":False,
"split":'949,50,1',
"squared_relu":False,
"standalone_embedding_stage":False,
"start_weight_decay":0.01,
"swiglu":False,
"swin_backbone_type":'tiny',
"tensor_model_parallel_size":1,
"tensorboard_dir":None,
"tensorboard_log_interval":1,
"tensorboard_queue_size":1000,
"terapipe_slice_len":32,
"test_data_path":None,
"timing_log_level":0,
"timing_log_option":'minmax',
"titles_data_path":None,
"tokenizer_model":None,
"tokenizer_type":'GPT2BPETokenizer',
"train_data_path":None,
"train_iters":500000,
"train_samples":None,
"transformer_impl":'local',
"transformer_pipeline_model_parallel_size":1,
"untie_embeddings_and_output_weights":False,
"use_checkpoint_args":False,
"use_checkpoint_opt_param_scheduler":False,
"use_cpu_initialization":None,
"use_distributed_optimizer":False,
"use_flash_attn":False,
"use_one_sent_docs":False,
"use_ring_exchange_p2p":False,
"use_rotary_position_embeddings":False,
"valid_data_path":None,
"variable_seq_lengths":False,
"virtual_pipeline_model_parallel_size":None,
"vision_backbone_type":'vit',
"vision_pretraining":False,
"vision_pretraining_type":'classify',
"vocab_extra_ids":0,
"vocab_file":'/home/tangjingqi/Megatron-LM/gpt2-vocab.json',
"vocab_size":None,
"weight_decay":0.01,
"weight_decay_incr_style":'constant',
"world_size":1
}

args = SimpleNamespace(**args)
# config = SimpleNamespace(**config)


def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    config = core_transformer_config_from_args(args)
    # model = GPTModel(
    #     config,
    #     num_tokentypes=0,
    #     parallel_output=True,
    #     pre_process=pre_process,
    #     post_process=post_process
    # )
    model = GPTModelTerapipe(
        config,
        num_tokentypes=0,
        parallel_output=True,
        pre_process=pre_process,
        post_process=post_process
    )

    return model


class TestTransformerLayers(unittest.TestCase):
    @patch('megatron.initialize.parse_args')
    @patch('megatron.initialize.validate_args')
    def test_output_consistency(self, mock_validate_args, mock_parse_args):
        torch.manual_seed(1024)
        # prepare some mock data and environment
        mock_parse_args.return_value = args
        mock_validate_args.return_value = args
        initialize_megatron(extra_args_provider=None,
                args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})
        model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
            model_provider, ModelType.encoder_or_decoder)
        config = get_model_config(model[0])
        layer_with_cache = model[0].module.module.language_model.encoder.layers[0]

        seq = args.seq_length # 2
        slice_len = args.terapipe_slice_len # 1
        # 准备输入数据
        input_seq = torch.randn((seq, 4, args.hidden_size), device='cuda', dtype=torch.float16)
        torch.save(input_seq, 'tmp_result_split_input.pt')

        # 处理分割后的输入
        split_input_parts = torch.split(input_seq, slice_len, dim=0)
        print(split_input_parts[0].shape)
        print(len(split_input_parts))
        # 分割后的mask
        split_att_mask = torch.tril(torch.ones((1,1,slice_len,slice_len), device=input_seq.device))
        print(f'dtype of mask is {split_att_mask.dtype}')
        cache = layer_with_cache.self_attention.create_attn_cache(      batch_size=args.micro_batch_size,
                                                                        max_seq_len=args.seq_length,
                                                                        device=torch.cuda.current_device(),
                                                                        dtype=args.params_dtype)
        cache_seq_len = 0
        split_output = []
        for input in split_input_parts:
            output_part, cache = layer_with_cache(input, split_att_mask, cache=cache, cache_seq_len=cache_seq_len)
            cache_seq_len += slice_len
            split_output.append(output_part)
        
        # 合并输出
        output = torch.cat(split_output, dim=0)

        # 存储输出
        torch.save(output, 'tmp_result_split_output.pt')

# 运行测试
if __name__ == '__main__':
    unittest.main()