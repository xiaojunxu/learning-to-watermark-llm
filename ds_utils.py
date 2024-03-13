import math
import torch
from torch import nn
import torch.nn.functional as F
import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.ops.adam import FusedAdam, DeepSpeedCPUAdam
from deepspeed.compression.helper import recursive_getattr, recursive_setattr
import deepspeed
from transformers import AutoModelForCausalLM, get_scheduler
from transformers.deepspeed import HfDeepSpeedConfig
import utils
import model_utils

GLOBAL_BATCH_SIZE = 32
MICRO_BATCH_SIZE = 4

def get_train_ds_config(offload=False,
                        dtype="bf16",
                        stage=0,
                        enable_hybrid_engine=False,
                        inference_tp_size=1,
                        release_inference_cache=False,
                        pin_parameters=True,
                        tp_gather_partition_size=8,
                        max_out_tokens=256,
                        enable_tensorboard=False,
                        enable_mixed_precision_lora=False,
                        tb_path="",
                        tb_name=""):

    device = "cpu" if offload else "none"
    if dtype == "fp16":
        data_type = "fp16"
        dtype_config = {"enabled": True, "loss_scale_window": 100}
    elif dtype == "bf16":
        data_type = "bfloat16"
        dtype_config = {"enabled": True}
    zero_opt_dict = {
        "stage": stage,
        "offload_param": {
            "device": device
        },
        "offload_optimizer": {
            "device": device
        },
        "stage3_param_persistence_threshold": 1e4,
        "stage3_max_live_parameters": 3e7,
        "stage3_prefetch_bucket_size": 3e7,
        "memory_efficient_linear": False
    }
    if enable_mixed_precision_lora:
        zero_opt_dict["zero_quantized_nontrainable_weights"] = True
        if dist.get_world_size() != get_accelerator().device_count():
            zero_opt_dict["zero_hpz_partition_size"] = get_accelerator(
            ).device_count()
    return {
        "train_batch_size": GLOBAL_BATCH_SIZE,
        "train_micro_batch_size_per_gpu": MICRO_BATCH_SIZE,
        "steps_per_print": 10,
        "zero_optimization": zero_opt_dict,
        data_type: dtype_config,
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
        "hybrid_engine": {
            "enabled": enable_hybrid_engine,
            "max_out_tokens": max_out_tokens,
            "inference_tp_size": inference_tp_size,
            "release_inference_cache": release_inference_cache,
            "pin_parameters": pin_parameters,
            "tp_gather_partition_size": tp_gather_partition_size,
        },
        "tensorboard": {
            "enabled": enable_tensorboard,
            "output_path": f"{tb_path}/ds_tensorboard_logs/",
            "job_name": f"{tb_name}_tensorboard"
        }
    }


def get_eval_ds_config(offload=False, dtype="bf16", stage=0):
    device = "cpu" if offload else "none"
    if dtype == "fp16":
        data_type = "fp16"
        dtype_config = {
            "enabled": True,
        }
    elif dtype == "bf16":
        data_type = "bfloat16"
        dtype_config = {"enabled": True}
    zero_opt_dict = {
        "stage": stage,
        "stage3_param_persistence_threshold": 1e4,
        "offload_param": {
            "device": device
        },
        "memory_efficient_linear": False
    }
    return {
        "train_batch_size": GLOBAL_BATCH_SIZE,
        "train_micro_batch_size_per_gpu": MICRO_BATCH_SIZE,
        "steps_per_print": 10,
        "zero_optimization": zero_opt_dict,
        data_type: dtype_config,
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False
    }

def get_optimizer_grouped_parameters(
    model,
    weight_decay,
    lora_lr=5e-4,
    no_decay_name_list=["bias", "LayerNorm.weight"],
    lora_name_list=["lora_right_weight", "lora_left_weight"],
):
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n for nd in no_decay_name_list)
                    and p.requires_grad and not any(nd in n
                                                    for nd in lora_name_list))
            ],
            "weight_decay":
            weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n for nd in no_decay_name_list)
                    and p.requires_grad and any(nd in n
                                                for nd in lora_name_list))
            ],
            "weight_decay":
            weight_decay,
            "lr":
            lora_lr
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (any(nd in n
                        for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay":
            0.0,
        },
    ]

    non_empty_groups = []
    for group in optimizer_grouped_parameters:
        if group["params"]:
            non_empty_groups.append(group)
    return non_empty_groups

class LinearLayer_LoRA(nn.Module):
    # an simple implementation of LoRA
    # for now only support Linear Layer
    def __init__(self,
                 weight,
                 lora_dim=0,
                 lora_scaling=1,
                 lora_droppout=0,
                 bias=None):
        super(LinearLayer_LoRA, self).__init__()
        self.weight = weight
        self.bias = bias

        if lora_dim <= 0:
            raise ValueError(
                "You are training to use LoRA, whose reduced dim should be larger than 1"
            )

        try:
            # for zero stage 3
            rows, columns = weight.ds_shape
        except:
            rows, columns = weight.shape
        self.lora_right_weight = nn.Parameter(torch.zeros(
            columns,
            lora_dim))  # apply transpose so in forward we do not need to
        self.lora_left_weight = nn.Parameter(torch.zeros(lora_dim, rows))
        self.lora_scaling = lora_scaling / lora_dim

        if lora_droppout > 0:
            self.lora_dropout = nn.Dropout(lora_droppout)
        else:
            self.lora_dropout = nn.Identity()

        self.reset_parameters()
        # disable the original weight gradient
        self.weight.requires_grad = False
        # fuse LoRA to the original weight
        self.fuse_lora = False

    def eval(self):
        self.lora_dropout.eval()

    def train(self, mode=True):
        self.lora_dropout.train(mode)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_right_weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_left_weight)

    def fuse_lora_weight(self):
        if not self.fuse_lora:
            self.weight.data += self.lora_scaling * torch.matmul(
                self.lora_left_weight.t(), self.lora_right_weight.t())
        self.fuse_lora = True

    def unfuse_lora_weight(self):
        if self.fuse_lora:
            self.weight.data -= self.lora_scaling * torch.matmul(
                self.lora_left_weight.t(), self.lora_right_weight.t())
        self.fuse_lora = False

    def forward(self, input):
        if self.fuse_lora:
            return F.linear(input, self.weight, self.bias)
        else:
            return F.linear(
                input, self.weight,
                self.bias) + (self.lora_dropout(input) @ self.lora_right_weight
                              @ self.lora_left_weight) * self.lora_scaling

# convert the linear layer to LoRA
def convert_linear_layer_to_lora(model,
                                 part_module_name,
                                 lora_dim=0,
                                 lora_scaling=1,
                                 lora_droppout=0):
    replace_name = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and part_module_name in name:
            replace_name.append(name)
    for name in replace_name:
        module = recursive_getattr(model, name)
        tmp = LinearLayer_LoRA(
            module.weight, lora_dim, lora_scaling, lora_droppout,
            module.bias).to(module.weight.device).to(module.weight.dtype)
        recursive_setattr(model, name, tmp)
    return model


def _z3_params_to_fetch(param_list):
    return [
        p for p in param_list
        if hasattr(p, 'ds_id') and p.ds_status == deepspeed.runtime.zero.
        partition_parameters.ZeroParamStatus.NOT_AVAILABLE
    ]


# convert the LoRA layer to linear layer
def convert_lora_to_linear_layer(model):
    replace_name = []
    for name, module in model.named_modules():
        if isinstance(module, LinearLayer_LoRA):
            replace_name.append(name)
    for name in replace_name:
        module = recursive_getattr(model, name)
        zero_stage_3 = hasattr(module.weight, 'ds_id')
        with deepspeed.zero.GatheredParameters(_z3_params_to_fetch([
                module.weight, module.bias, module.lora_left_weight,
                module.lora_right_weight
        ]),
                                               modifier_rank=0,
                                               enabled=zero_stage_3):
            module.fuse_lora_weight()
    return model

# copied from deepspeed
def load_state_dict_into_model(model_to_load=None,
                               state_dict=None,
                               start_prefix="",
                               zero_stage=0):

    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    error_msgs = []

    # PyTorch's `_load_from_state_dict` does not copy parameters in a module's descendants
    # so we need to apply the function recursively.
    def load(module: nn.Module, state_dict, prefix=""):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        args = (state_dict, prefix, local_metadata, True, [], [], error_msgs)
        # Parameters of module and children will start with prefix. We can exit early if there are none in this
        # state_dict
        if len([key for key in state_dict if key.startswith(prefix)]) > 0:
            if zero_stage == 3:
                # In sharded models, each shard has only part of the full state_dict, so only gather
                # parameters that are in the current state_dict.
                named_parameters = dict(
                    module.named_parameters(prefix=prefix[:-1], recurse=False))
                params_to_gather = [
                    named_parameters[k] for k in state_dict.keys()
                    if k in named_parameters
                ]
                if len(params_to_gather) > 0:
                    # because zero3 puts placeholders in model params, this context
                    # manager gathers (unpartitions) the params of the current layer, then loads from
                    # the state dict and then re-partitions them again
                    with deepspeed.zero.GatheredParameters(params_to_gather,
                                                           modifier_rank=0):
                        if torch.distributed.get_rank() == 0:
                            module._load_from_state_dict(*args)
            else:
                module._load_from_state_dict(*args)

        for name, child in module._modules.items():
            if child is not None:
                load(child, state_dict, prefix + name + ".")

    load(model_to_load, state_dict, prefix=start_prefix)
    # Delete `state_dict` so it could be collected by GC earlier. Note that `state_dict` is a copy of the argument, so
    # it's safe to delete it.
    del state_dict

    return error_msgs

def get_reward_model(args, tokenizer, stage=0):
    ds_config = get_train_ds_config(stage=stage)
    ds_config['train_micro_batch_size_per_gpu'] = args.batch_size
    ds_config['train_batch_size'] = args.batch_size #* torch.distributed.get_world_size()
    if stage == 3:
        dschf = HfDeepSpeedConfig(ds_config)   # Note: dschf is defined in function scope to avoid global effects
    base_model = utils.get_model(args.reward_model)
    reward_model = model_utils.RewardModel(base_model, tokenizer)
    load_state_dict_into_model(reward_model, torch.load('%s/ckpt/%s_%s_raw_detector/reward_model.ckpt'%(args.workdir, args.dataset, args.reward_model), map_location='cpu'), "", zero_stage=stage)
    if args.use_reward_lora:
        reward_model = convert_linear_layer_to_lora(reward_model, part_module_name='decoder.layers.', lora_dim=128)
    optim_params = get_optimizer_grouped_parameters(reward_model, weight_decay=0, lora_lr=args.reward_lora_lr)

    if 'llama' in args.reward_model:
        assert stage == 0
        ds_config['zero_optimization']['stage'] = 1
        ds_config['zero_optimization']['offload_optimizer']['device'] = 'cpu'
        optim = DeepSpeedCPUAdam(optim_params, lr=args.reward_lr, betas=(0.9, 0.95))
        reward_model.config.end_token_id = tokenizer.eos_token_id
        reward_model.config.pad_token_id = reward_model.config.eos_token_id
    else:
        optim = FusedAdam(optim_params, lr=args.reward_lr, betas=(0.9, 0.95))

    if args.reward_with_scheduler:
        lr_scheduler = get_scheduler(name='cosine', optimizer=optim, num_warmup_steps=min(100,0.1*args.train_steps), num_training_steps=args.train_steps)
        reward_engine, *_ = deepspeed.initialize(model=reward_model,optimizer=optim,lr_scheduler=lr_scheduler,config=ds_config)
    else:
        reward_engine, *_ = deepspeed.initialize(model=reward_model,optimizer=optim,config=ds_config)
    return reward_engine

def get_critic_model(args, tokenizer, stage=0):
    ds_config = get_train_ds_config(stage=stage)
    ds_config['train_micro_batch_size_per_gpu'] = args.batch_size
    ds_config['train_batch_size'] = args.batch_size #* torch.distributed.get_world_size()
    if stage == 3:
        dschf = HfDeepSpeedConfig(ds_config)   # Note: dschf is defined in function scope to avoid global effects
    base_model = utils.get_model(args.reward_model)
    critic_model = model_utils.RewardModel(base_model, tokenizer)
    load_state_dict_into_model(critic_model, torch.load('%s/ckpt/%s_%s_raw_detector/reward_model.ckpt'%(args.workdir, args.dataset, args.reward_model), map_location='cpu'), "", zero_stage=stage)
    if args.use_critic_lora:
        critic_model = convert_linear_layer_to_lora(critic_model, part_module_name='decoder.layers.', lora_dim=128)
    optim_params = get_optimizer_grouped_parameters(critic_model, weight_decay=0, lora_lr=args.critic_lora_lr)
    if 'llama' in args.reward_model:
        assert stage == 0
        ds_config['zero_optimization']['stage'] = 1
        ds_config['zero_optimization']['offload_optimizer']['device'] = 'cpu'
        optim = DeepSpeedCPUAdam(optim_params, lr=args.critic_lr, betas=(0.9, 0.95))
    else:
        optim = FusedAdam(optim_params, lr=args.critic_lr, betas=(0.9, 0.95))
    lr_scheduler = get_scheduler(name='cosine', optimizer=optim, num_warmup_steps=min(100,0.1*args.train_steps), num_training_steps=args.train_steps)
    critic_engine, *_ = deepspeed.initialize(model=critic_model,optimizer=optim,lr_scheduler=lr_scheduler,config=ds_config)
    return critic_engine

def get_gtreward_model(args, tokenizer, stage=0):
    ds_config = get_eval_ds_config(stage=stage)
    ds_config['train_micro_batch_size_per_gpu'] = args.batch_size
    ds_config['train_batch_size'] = args.batch_size #* torch.distributed.get_world_size()
    if stage == 3:
        dschf = HfDeepSpeedConfig(ds_config)   # Note: dschf is defined in function scope to avoid global effects

    base_model = utils.get_model(args.reward_model)
    gtreward_model = model_utils.RewardModel(base_model, tokenizer)
    LOAD_PREFIX = args.workdir+'/deepspeed_ckpt'
    if args.dataset == 'PKU' and args.reward_model == 'opt-350m':
        load_state_dict_into_model(gtreward_model, torch.load('%s/opt-350m/pytorch_model.bin'%LOAD_PREFIX, map_location='cpu'), "", zero_stage=stage)
    elif args.dataset == 'PKU' and args.reward_model == 'llama2-1.1b':
        load_state_dict_into_model(gtreward_model, torch.load('%s/llama2-1.1b/pytorch_model.bin'%LOAD_PREFIX, map_location='cpu'), "", zero_stage=stage)
    else:
        raise NotImplementedError()

    gtreward_engine, *_ = deepspeed.initialize(model=gtreward_model,config=ds_config)
    return gtreward_engine

def get_actor_model(args, tokenizer, model_path=None, stage=0):
    ds_config = get_train_ds_config(enable_hybrid_engine=True, stage=stage)
    ds_config['train_micro_batch_size_per_gpu'] = args.batch_size
    ds_config['train_batch_size'] = args.batch_size 
    if stage == 3:
        dschf = HfDeepSpeedConfig(ds_config)   # Note: dschf is defined in function scope to avoid global effects
    if model_path is None:
        if args.dataset in ['c4']:
            actor_model = utils.get_model(args.actor_model, model_class=AutoModelForCausalLM)
        else:
            actor_model = utils.get_model(args.actor_model, model_class=AutoModelForCausalLM, model_path="%s/ckpt/%s_%s_sft"%(args.workdir, args.actor_model, args.dataset))
    else:
        actor_model = utils.get_model(args.actor_model, model_class=AutoModelForCausalLM, model_path=model_path)
    if args.use_lora:
        actor_model = convert_linear_layer_to_lora(actor_model, part_module_name='decoder.layers.', lora_dim=128)
    optim_params = get_optimizer_grouped_parameters(actor_model, weight_decay=0, lora_lr=args.lora_lr)
    if 'llama' in args.actor_model:
        assert stage == 0
        ds_config['zero_optimization']['stage'] = 1
        ds_config['zero_optimization']['offload_optimizer']['device'] = 'cpu'
        optim = DeepSpeedCPUAdam(optim_params, lr=args.lr, betas=(0.9, 0.95))
        actor_model.config.end_token_id = tokenizer.eos_token_id
        actor_model.config.pad_token_id = actor_model.config.eos_token_id
    else:
        optim = FusedAdam(optim_params, lr=args.lr, betas=(0.9, 0.95))

    lr_scheduler = get_scheduler(name='cosine', optimizer=optim, num_warmup_steps=min(100,0.1*args.train_steps), num_training_steps=args.train_steps)
    actor_engine, *_ = deepspeed.initialize(model=actor_model,optimizer=optim,lr_scheduler=lr_scheduler,config=ds_config)
    return actor_engine

def get_ref_model(args, tokenizer, model_path=None, stage=0):
    ds_config = get_eval_ds_config(stage=stage)
    ds_config['train_micro_batch_size_per_gpu'] = args.batch_size
    ds_config['train_batch_size'] = args.batch_size #* torch.distributed.get_world_size()
    if stage == 3:
        dschf = HfDeepSpeedConfig(ds_config)   # Note: dschf is defined in function scope to avoid global effects

    if model_path is None:
        if args.dataset in ['c4']:
            ref_model = utils.get_model(args.actor_model, model_class=AutoModelForCausalLM)
        else:
            ref_model = utils.get_model(args.actor_model, model_class=AutoModelForCausalLM, model_path="%s/ckpt/%s_%s_sft"%(args.workdir, args.actor_model, args.dataset))
    else:
        ref_model = utils.get_model(args.actor_model, model_class=AutoModelForCausalLM, model_path=model_path)
    ref_engine, *_ = deepspeed.initialize(model=ref_model,config=ds_config)
    return ref_engine
