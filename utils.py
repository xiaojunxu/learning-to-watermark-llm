import torch
import torch.nn.functional as F

from datasets import load_dataset, Dataset
from transformers import AutoConfig, AutoTokenizer, AutoModel, DataCollatorForLanguageModeling

def get_model(model_name, model_class=AutoModel, model_path=None, dropout=0.0, load_in_8bit=False, **kwargs):
    if "opt" in model_name:
        assert not load_in_8bit
        model_config = AutoConfig.from_pretrained("facebook/%s"%model_name)
        for key in ('dropout', 'attention_dropout', 'hidden_dropout', 'activation_dropout'):
            if hasattr(model_config, key):
                setattr(model_config, key, dropout)
        if model_path is not None:
            model = model_class.from_pretrained(model_path, config=model_config)
        else:
            model = model_class.from_pretrained("facebook/%s"%model_name, config=model_config)
    elif "llama" in model_name:
        if model_name == "llama2-7b":
            model_config = AutoConfig.from_pretrained("meta-llama/Llama-2-7b-hf")
            for key in ('dropout', 'attention_dropout', 'hidden_dropout', 'activation_dropout'):
                if hasattr(model_config, key):
                    setattr(model_config, key, dropout)
            if model_path is not None:
                model = model_class.from_pretrained(model_path, config=model_config, load_in_8bit=load_in_8bit)
            else:
                model = model_class.from_pretrained("meta-llama/Llama-2-7b-hf", config=model_config, load_in_8bit=load_in_8bit)
        elif model_name == "llama2-1.1b":
            assert not load_in_8bit
            model_config = AutoConfig.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
            for key in ('dropout', 'attention_dropout', 'hidden_dropout', 'activation_dropout'):
                if hasattr(model_config, key):
                    setattr(model_config, key, dropout)
            if model_path is not None:
                model = model_class.from_pretrained(model_path, config=model_config)
            else:
                model = model_class.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T", config=model_config)
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()
    return model

def get_tokenizer(model_name):
    if "opt" in model_name:
        tokenizer = AutoTokenizer.from_pretrained("facebook/%s"%model_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right'
    elif "llama" in model_name:
        assert model_name == "llama2-7b" or model_name == "llama2-1.1b"
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right'
    else:
        raise NotImplementedError()
    return tokenizer

def get_PKU_dataset():
    raw_train_dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="train")
    train_dataset = []
    for line in raw_train_dataset:
        prompt = "### Question: %s\n ### Answer:"%line['prompt']
        train_dataset.append({'prompt': prompt, 'continuation': ' '+line['response_0']})

    raw_test_dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="test")
    test_dataset = []
    for line in raw_test_dataset:
        prompt = "### Question: %s\n ### Answer:"%line['prompt']
        test_dataset.append({'prompt': prompt, 'continuation': ' '+line['response_0']})
    return train_dataset, test_dataset

def get_c4_dataset(tokenizer):
    raw_train_dataset = load_dataset("c4", "realnewslike", split="train", streaming=True)
    train_dataset = []
    for i, line in enumerate(raw_train_dataset):
        toks = tokenizer(line['text'])['input_ids']
        if len(toks) < 1+32+128:
            continue
        prompt = tokenizer.decode(toks[1:1+32])
        cont = tokenizer.decode(toks[1+32:1+32+128])
        train_dataset.append({'prompt': prompt, 'continuation': cont})
        if len(train_dataset) >= 40000:
            break

    raw_test_dataset = load_dataset("c4", "realnewslike", split="validation", streaming=True)
    test_dataset = []
    for i, line in enumerate(raw_test_dataset):
        toks = tokenizer(line['text'])['input_ids']
        if len(toks) < 1+32+128:
            continue
        prompt = tokenizer.decode(toks[1:1+32])
        cont = tokenizer.decode(toks[1+32:1+32+128])
        test_dataset.append({'prompt': prompt, 'continuation': cont})
        if len(test_dataset) >= 10000:
            break
    return train_dataset, test_dataset


def create_prompt_loader(dataset, tokenizer, batch_size):
    prompt_dataset = []
    for line in dataset:
        prompt_dataset.append({'text':line['prompt']})
    def preprocess(examples):
        return tokenizer(examples['text'])
    prompt_new_dataset = Dataset.from_list(prompt_dataset).map(preprocess, batched=True, remove_columns=['text'])
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    prompt_dataloader = torch.utils.data.DataLoader(prompt_new_dataset, batch_size=batch_size, collate_fn=data_collator)
    return prompt_dataloader

def create_prompt_and_cont_loaders(dataset, tokenizer, tokenizer_right_pad, batch_size, prompt_max_len=256, cont_max_len=128):
    prompt_dataset = []
    cont_dataset = []
    for line in dataset:
        prompt_dataset.append({'text':line['prompt']})
        cont_dataset.append({'text':line['cont_human']})

    def preprocess(examples):
        return tokenizer(examples['text'], max_length=prompt_max_len, truncation=True)
    prompt_new_dataset = Dataset.from_list(prompt_dataset).map(preprocess, batched=True, remove_columns=['text'])
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    prompt_dataloader = torch.utils.data.DataLoader(prompt_new_dataset, batch_size=batch_size, collate_fn=data_collator)

    def preprocess_right_pad(examples):
        return tokenizer_right_pad(examples['text'], max_length=cont_max_len, truncation=True)
    cont_new_dataset = Dataset.from_list(cont_dataset).map(preprocess_right_pad, batched=True, remove_columns=['text'])
    cont_data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer_right_pad, mlm=False)
    cont_dataloader = torch.utils.data.DataLoader(cont_new_dataset, batch_size=batch_size, collate_fn=cont_data_collator)

    return prompt_dataloader, cont_dataloader

class MiniDataset:
    def __init__(self, max_size, small_batch_size):
        self.dataset = []
        self.max_size = max_size
        self.small_batch_size = small_batch_size

    def seperate(self):
        small_dataset = []
        for large_batch in self.dataset:
            if type(large_batch) == list or type(large_batch) == tuple:
                large_size = len(large_batch[0])
            elif type(large_batch) == dict:
                large_size = len(large_batch[list(large_batch.keys())[0]])
            else:
                large_size = len(large_batch)
            for i in range(0, large_size, self.small_batch_size):
                if type(large_batch) == list or type(large_batch) == tuple:
                    small_dataset.append(
                        [x[i:i + self.small_batch_size] for x in large_batch])
                elif type(large_batch) == dict:
                    small_dataset.append({
                        k: v[i:i + self.small_batch_size]
                        for k, v in large_batch.items()
                    })
                else:
                    small_dataset.append(large_batch[i:i +
                                                     self.small_batch_size])
        self.free()

        return small_dataset

    def add(self, data):
        if len(self.dataset) < self.max_size:
            self.dataset.append(data)
            if len(self.dataset) == self.max_size:
                return self.seperate()
            else:
                return None
        else:
            raise ValueError(
                "The dataset is full but we did not stop it. There is a bug in the code."
            )

    def free(self):
        self.dataset = []

def gather_log_probs(logits, labels):
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)

def actor_loss_fn(logprobs, old_logprobs, advantages, mask):
    cliprange = 0.2
    ## policy gradient loss
    log_ratio = (logprobs - old_logprobs) * mask
    ratio = torch.exp(log_ratio)
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - cliprange,
                                         1.0 + cliprange)
    pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * mask) / mask.sum()
    return pg_loss

def critic_loss_fn(values, old_values, returns, mask):
    cliprange_value = 0.2
    values_clipped = torch.clamp(
        values,
        old_values - cliprange_value,
        old_values + cliprange_value,
    )
    vf_loss1 = (values - returns)**2
    vf_loss2 = (values_clipped - returns)**2
    vf_loss = 0.5 * torch.sum(
        torch.max(vf_loss1, vf_loss2) * mask) / mask.sum()
    return vf_loss

def train_rlhf(actor_model, critic_model, exp_data, kl_ctl=0.1):
    rew_clip_val = 5.0
    gamma = 1.0
    lam = 0.95

    prompts = exp_data['prompts']
    log_probs = exp_data['logprobs']
    ref_log_probs = exp_data['ref_logprobs']
    values = exp_data['value']
    reward_score = exp_data['rewards']
    seq = exp_data['input_ids']
    attention_mask = exp_data['attention_mask']

    start = prompts.size()[-1] - 1
    action_mask = attention_mask[:,1:]
    ends = start + action_mask[:,start:].sum(1)+1
    old_values = values

    with torch.no_grad():
        # calc old rewards with KL
        old_rewards = -kl_ctl * (log_probs - ref_log_probs)
        reward_clip = torch.clamp(reward_score, -rew_clip_val, rew_clip_val)
        for i in range(len(old_rewards)):
            old_rewards[i,start:ends[i]][-1] += reward_clip[i]
            old_rewards[i,ends[i]:] = 0
            old_values[i,ends[i]:] = 0

        # calc advantage and return
        lastgaelam = 0
        advantages_reversed = []
        length = old_rewards.size()[-1]
        for t in reversed(range(start, length)):
            nextvalues = old_values[:,t+1] if t < length-1 else 0.0
            delta = old_rewards[:,t] + gamma * nextvalues - old_values[:,t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + old_values[:, start:]
        advantages = advantages.detach()

    actor_prob = actor_model(input_ids=seq, attention_mask=attention_mask, use_cache=False).logits
    actor_log_prob = gather_log_probs(actor_prob[:,:-1,:], seq[:,1:])
    actor_loss = actor_loss_fn(actor_log_prob[:,start:], log_probs[:,start:], advantages, action_mask[:,start:])
    value = critic_model.forward_value(input_ids=seq, attention_mask=attention_mask, return_value_only=True, use_cache=False)[:, :-1]
    critic_loss = critic_loss_fn(value[:,start:], old_values[:,start:], returns, action_mask[:,start:])
    return actor_loss, critic_loss
