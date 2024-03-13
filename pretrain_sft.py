import argparse
import numpy as np
import os
from transformers import AutoModelForCausalLM, get_scheduler, DataCollatorForLanguageModeling, default_data_collator
import torch
from torch.optim import Adam
from utils import get_model, get_tokenizer
from model_utils import get_optimizer_grouped_parameters

from datasets import load_dataset

torch.manual_seed(8888)
np.random.seed(8888)


def create_pku_data_loader_withdataset(tokenizer, dataset, mode="bad", fraction=1.0,
                           batch_size=4):
    '''
    https://huggingface.co/docs/trl/main/en/sft_trainer
    mode: bad or all
    '''
    assert mode in ["bad", "normal", "full"], "Mode must be bad, normal, or full"
    
    # Preproccess function.
    def preproccess(examples):
        '''
        Input: Dict[List]
        Output: Dict[List]
        '''
        results = {'input_ids': [], 'attention_mask': [], 'start_locs':[]}

        for i in range(len(examples['prompt'])):
            # Subsample if needed.
            prompt = examples['prompt'][i]
            response_list = []

            # Add only bad samples.
            if mode == "bad":
                if not examples['is_response_0_safe'][i]:
                    response_list.append(examples['response_0'][i])
                if not examples['is_response_1_safe'][i]:
                    response_list.append(examples['response_1'][i])

            # Normal samples.
            elif mode == "normal":
                if examples['is_response_0_safe'][i]:
                    response_list.append(examples['response_0'][i])
                if examples['is_response_1_safe'][i]:
                    response_list.append(examples['response_1'][i])
            
            # Add all samples.
            elif mode == "full":
                response_list.append(examples['response_0'][i])
                response_list.append(examples['response_1'][i])
            else:
                print('No valid option') 

            # Add all responses to results or skip if none.
            for response in response_list:
                text = f"### Question: {prompt}\n ### Answer: {response}<|endoftext|>"
                tokenized = tokenizer(text, max_length=200, truncation=True, padding='max_length', return_tensors="pt")
                results['input_ids'].append(tokenized['input_ids'].squeeze(0))
                results['attention_mask'].append(tokenized['attention_mask'].squeeze(0))
                # Calculate start idx for answer
                test_text = f"### Question: {prompt}\n ### Answer: "
                test_tokenized = tokenizer(test_text)
                results['start_locs'].append(len(test_tokenized['input_ids'])-1)
    
        return results

    dataset = dataset.map(
        preproccess, batched=True,
        remove_columns=["prompt", "response_0", "response_1",
                        "is_response_0_safe", "is_response_1_safe",
                        "better_response_id", "safer_response_id"])
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'start_locs'])
    if mode == "bad":
        print('# of bad samples: ', len(dataset))
    elif mode == 'normal':
        print('# of normal samples: ', len(dataset))
    elif mode == "full":
        print('# of full samples: ', len(dataset))

    # Add labels. Make it data loader.
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, collate_fn=data_collator)

    return dataloader

def create_pku_data_loader_withdataset_v2(tokenizer, dataset, mode="bad", fraction=1.0,
                           batch_size=4):
    assert mode == "full"
    processed_dataset = []
    for line in dataset:
        prompt = line['prompt']
        response = line['response_0']
        text = f"### Question: {prompt}\n ### Answer: {response}<|endoftext|>"
        token = tokenizer(text, max_length=200, padding="max_length", truncation=True, return_tensors="pt")
        token["input_ids"] = token["input_ids"].squeeze(0)
        token["attention_mask"] = token["attention_mask"].squeeze(0)
        token["labels"] = token["input_ids"]
        processed_dataset.append(token)
    dataloader = torch.utils.data.DataLoader(processed_dataset,collate_fn=default_data_collator,batch_size=batch_size)
    return dataloader

def get_PKU_plain_text_withdataset(dataset, prompt_only=True):
    results = []

    for example in dataset:
        if prompt_only:
            results.append("### Question: %s\n ### Answer: "%example['prompt'])
        else:
            raise NotImplementedError

    return results

def main(args) -> None:
    device = torch.device("cuda:0") if torch.cuda.is_available()\
                    else torch.device("cpu")

    SAVE_PATH = "%s/ckpt/%s_PKU_sft"%(args.workdir, args.model)
    print ("\033[94mSave path:%s\033[0m"%SAVE_PATH)

    if 'llama' in args.model:
        with open("%s/hf.key"%args.workdir) as inf:
            key = inf.readline().strip()
            os.environ['HF_TOKEN'] = key

    # Load data.
    tokenizer = get_tokenizer(args.model)

    train_dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF", split='train')
    test_dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF", split='test')
    train_loader = create_pku_data_loader_withdataset(
        tokenizer, train_dataset, mode="normal", batch_size=args.batch_size)

    # Load model.
    if "opt" in args.model:
        model = get_model(args.model, model_class=AutoModelForCausalLM)
    elif "llama" in args.model:
        assert args.model == "llama2-7b"
        model = get_model(args.model, model_class=AutoModelForCausalLM)
    else:
        raise NotImplementedError()

    if args.use_lora:
        from peft import get_peft_model, AdaLoraConfig, TaskType
        assert "opt" in args.model or "llama" in args.model
        if "llama" in args.model:
            peft_config = AdaLoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=128, lora_alpha=16, target_modules=["q_proj", "v_proj"])
        else:
            peft_config = AdaLoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=32, lora_alpha=16, target_modules=["q_proj", "v_proj"])
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    model.to(device)

    optimizer_grouped_parameters = get_optimizer_grouped_parameters(
        model, weight_decay=0)
    optimizer = Adam(optimizer_grouped_parameters, lr=args.lr, betas=(0.9, 0.95))
    num_training_steps = args.learn_steps
    lr_scheduler = get_scheduler(
        name="cosine", optimizer=optimizer,
        num_warmup_steps=int(0.01*num_training_steps), num_training_steps=num_training_steps)
    model.train()

    tot_step = 0
    for (idx, batch) in enumerate(train_loader):
        if args.learn_steps >= 0 and tot_step >= args.learn_steps:
            break

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        labels[labels==-100] = tokenizer.pad_token_id  # Do not hope the collator to ignore pad tokens in SFT!

        outputs = model(
            input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()

        # Loss.
        if idx % 100 == 0:
            print('Batch %d/%d, Total step %d, loss: %f' % (
                idx, len(train_loader),
                tot_step, loss.item()))
        tot_step += 1

    if args.use_lora:
        model = model.merge_and_unload()

    # Save model.
    model.save_pretrained(SAVE_PATH, from_pt=True)
    print ("\033[94mSaved to %s\033[0m"%SAVE_PATH)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', type=str, default='opt-1.3b')
    parser.add_argument('--use_lora', action='store_true')

    parser.add_argument('--learn_steps', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size of unlearning.')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='Unlearning LR.')

    parser.add_argument('--workdir', type=str, default='.')

    args = parser.parse_args()
    print (args)

    main(args)
