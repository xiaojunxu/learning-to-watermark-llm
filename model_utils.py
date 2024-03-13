import torch
import torch.nn as nn

class RewardModel(nn.Module):
    def __init__(self, base_model, tokenizer):
        super(RewardModel, self).__init__()
        self.config = base_model.config
        print (self.config)
        if hasattr(self.config, "word_embed_proj_dim"):
            self.v_head = nn.Linear(self.config.word_embed_proj_dim,1,bias=False)
        else:
            self.v_head = nn.Linear(self.config.hidden_size, 1, bias=False)
        self.rwtransformer = base_model
        self.PAD_ID = tokenizer.pad_token_id
        self.tokenizer = tokenizer

        if "OPT" in base_model.__class__.__name__:
            self.num_padding_at_beginning = 1
        else:
            assert "Llama" in base_model.__class__.__name__ or "llama" in base_model.__class__.__name__
            self.num_padding_at_beginning = 0

    def forward(self, chosen_ids, chosen_mask, rejected_ids, rejected_mask, use_cache=False):
        chosen_outputs = self.rwtransformer(chosen_ids, chosen_mask, use_cache=use_cache)
        chosen_rewards = self.v_head(chosen_outputs[0]).squeeze(-1)
        rejected_outputs = self.rwtransformer(rejected_ids, rejected_mask, use_cache=use_cache)
        rejected_rewards = self.v_head(rejected_outputs[0]).squeeze(-1)
        seq_len = min(chosen_ids.shape[1], rejected_ids.shape[1])

        chosen_mean_scores = []
        rejected_mean_scores = []
        # Compute pairwise loss. Only backprop on the different tokens before padding; adapted from DeepSpeed
        loss = 0.
        for i in range(len(chosen_ids)):
            chosen_id = chosen_ids[i]
            rejected_id = rejected_ids[i]
            chosen_reward = chosen_rewards[i]
            rejected_reward = rejected_rewards[i]

            start_ind = chosen_mask[i].nonzero()[0].item()

            c_inds = (chosen_id[start_ind:] == self.PAD_ID).nonzero()
            c_ind = c_inds[self.num_padding_at_beginning].item()+start_ind if len(c_inds)>self.num_padding_at_beginning else seq_len
            if len(chosen_id) < len(rejected_id):
                check_divergence = (chosen_id != rejected_id[:len(chosen_id)]).nonzero()
            else:
                check_divergence = (chosen_id[:len(rejected_id)] != rejected_id).nonzero()
            if len(check_divergence) == 0:
                end_ind = rejected_reward.size(-1)
                end_ind = min(end_ind, seq_len)
                divergence_ind = end_ind - 1
                r_ind = min(c_ind, seq_len)
            else:
                # Check if there is any padding otherwise take length of sequence
                r_inds = (rejected_id[start_ind:] == self.PAD_ID).nonzero()
                r_ind = r_inds[self.num_padding_at_beginning].item()+start_ind if len(r_inds) > self.num_padding_at_beginning else seq_len
                end_ind = max(c_ind, r_ind)
                end_ind = min(end_ind, seq_len)
                divergence_ind = check_divergence[0]
            c_ind = min(c_ind, r_ind)
            r_ind = min(c_ind, r_ind)

            assert divergence_ind > 0
            if divergence_ind == end_ind:
                divergence_ind -= 1
            chosen_mean_scores.append(
                chosen_reward[c_ind - 1])  #use the end score for reference
            rejected_mean_scores.append(rejected_reward[r_ind - 1])
            loss += -torch.nn.functional.logsigmoid(chosen_reward[c_ind-1]-rejected_reward[r_ind-1]).mean()
        loss = loss / len(chosen_ids)
        chosen_mean_scores = torch.stack(chosen_mean_scores)
        rejected_mean_scores = torch.stack(rejected_mean_scores)
        return {
            "loss": loss,
            "chosen_mean_scores": chosen_mean_scores,
            "rejected_mean_scores": rejected_mean_scores,
        }

    def forward_value(self, input_ids, attention_mask, return_value_only=False, prompt_length=0, use_cache=False):
        outputs = self.rwtransformer(input_ids, attention_mask, use_cache=use_cache)
        values = self.v_head(outputs[0]).squeeze(-1)
        if return_value_only:
            return values
        else:
            assert prompt_length > 1
            bs = values.size(0)
            seq_len = input_ids.shape[1]
            chosen_end_scores = []
            for i in range(bs):
                input_id = input_ids[i]
                value = values[i]

                c_inds = (input_id[prompt_length:] == self.PAD_ID).nonzero()
                c_ind = c_inds[0].item() + prompt_length if len(c_inds) > 0 else seq_len
                chosen_end_scores.append(value[c_ind - 1])
            return {"values":values, "chosen_end_scores":torch.stack(chosen_end_scores)}


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

