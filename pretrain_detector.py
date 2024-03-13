import argparse
import numpy as np
import json
import os
from tqdm import tqdm

from transformers import AutoModelForCausalLM, get_scheduler, DataCollatorForLanguageModeling, pipeline
from transformers.pipelines.pt_utils import KeyDataset
import torch
from datasets import Dataset
from sklearn.metrics import roc_auc_score

import utils
import model_utils

def generate_RM_dataset(model_name, data_name, workdir='.'):
    DATA_PREFIX = data_name if "opt" in model_name else "%s-%s"%(data_name, model_name)
    tokenizer = utils.get_tokenizer(model_name)
    if data_name == 'PKU':
        raw_train_dataset, raw_test_dataset = utils.get_PKU_dataset()
        SUFFIX = '<|endoftext|>'
    elif data_name == 'c4':
        raw_train_dataset, raw_test_dataset = utils.get_c4_dataset(utils.get_tokenizer("opt-1.3b"))  # always use opt tokenizer to process raw dataset
        SUFFIX = ''
    else:
        raise NotImplementedError()

    if data_name == 'PKU':
        model = utils.get_model(model_name, model_class=AutoModelForCausalLM, model_path="%s/ckpt/%s_%s_sft"%(workdir, model_name, data_name))
        max_length = 200
    elif data_name == 'c4':
        model = utils.get_model(model_name, model_class=AutoModelForCausalLM)
        max_length = 128+32
    else:
        raise NotImplementedError()
    model.eval()

    tokenizer.padding_side = 'left'
    text_generator = pipeline('text-generation', model=model, do_sample=True, tokenizer=tokenizer, max_length=max_length, device="cuda:0")
    if not os.path.isdir('%s/data'%workdir):
        os.mkdir('%s/data'%workdir)

    train_dataset = []
    for i, response in enumerate(text_generator(KeyDataset(raw_train_dataset, 'prompt'), return_full_text=False, batch_size=32)):
        if i % 100 == 0:
            print ("Training %d/40000"%i)
        if i >= 40000:
            break
        ret = response[0]['generated_text']
        human_ret = raw_train_dataset[i]['continuation']+SUFFIX
        train_dataset.append({'prompt':raw_train_dataset[i]['prompt'], 'cont_llm': ret, 'cont_human': human_ret})
    with open('%s/data/%s_RMdata_train_ft.jsonl'%(workdir,DATA_PREFIX),'w') as outf:
        for line in train_dataset:
            outf.write(json.dumps(line)+'\n')

    test_dataset = []
    for i, response in enumerate(text_generator(KeyDataset(raw_test_dataset, 'prompt'), return_full_text=False, batch_size=32)):
        print ("test %d/10000"%i)
        #if i >= 10:
        if i >= 10000:
            break
        ret = response[0]['generated_text']
        human_ret = raw_test_dataset[i]['continuation']+SUFFIX
        test_dataset.append({'prompt':raw_test_dataset[i]['prompt'], 'cont_llm': ret, 'cont_human': human_ret})
    with open('%s/data/%s_RMdata_test_ft.jsonl'%(workdir,DATA_PREFIX),'w') as outf:
        for line in test_dataset:
            outf.write(json.dumps(line)+'\n')
    return train_dataset, test_dataset

def load_RM_dataset(data_name, model_name, workdir='.'):
    if "opt" in model_name:
        DATA_PREFIX = data_name
    else:
        assert "llama" in model_name
        DATA_PREFIX = data_name+"-llama2-7b"
    train_dataset = []
    with open('%s/data/%s_RMdata_train_ft.jsonl'%(workdir, DATA_PREFIX)) as inf:
        for line in inf:
            train_dataset.append(json.loads(line.strip()))
    test_dataset = []
    with open('%s/data/%s_RMdata_test_ft.jsonl'%(workdir, DATA_PREFIX)) as inf:
        for line in inf:
            test_dataset.append(json.loads(line.strip()))
    return train_dataset, test_dataset

def RM_dataset_to_loader(dataset, tokenizer, batch_size=4):
    llm_dataset = []
    human_dataset = []
    for line in dataset:
        llm_dataset.append({'text': line['prompt']+line['cont_llm']})
        human_dataset.append({'text': line['prompt']+line['cont_human']})
    def preprocess(examples):
        return tokenizer(examples['text'])
    
    llm_new_dataset = Dataset.from_list(llm_dataset).map(preprocess, batched=True, remove_columns=['text'])
    human_new_dataset = Dataset.from_list(human_dataset).map(preprocess, batched=True, remove_columns=['text'])
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    llm_dataloader = torch.utils.data.DataLoader(llm_new_dataset, batch_size=batch_size, collate_fn=data_collator)
    human_dataloader = torch.utils.data.DataLoader(human_new_dataset, batch_size=batch_size, collate_fn=data_collator)
    assert len(llm_dataloader) == len(human_dataloader)
    return llm_dataloader, human_dataloader

def main(args):
    DEVICE = "cuda:0"

    if args.gen_dataset:
        if "opt" in args.model:
            train_dataset, test_dataset = generate_RM_dataset("opt-1.3b", args.dataset, workdir=args.workdir)
        elif "llama" in args.model:
            train_dataset, test_dataset = generate_RM_dataset("llama2-7b", args.dataset, workdir=args.workdir)
        else:
            raise NotImplementedError()
    else:
        # Load the preprocessed dataset
        train_dataset, test_dataset = load_RM_dataset(args.dataset, args.model, workdir=args.workdir)

    tokenizer = utils.get_tokenizer(args.model)
    train_llm_loader, train_human_loader = RM_dataset_to_loader(train_dataset, tokenizer, batch_size=4)
    test_llm_loader, test_human_loader = RM_dataset_to_loader(test_dataset, tokenizer, batch_size=4)

    base_model = utils.get_model(args.model).to(DEVICE)
    if args.use_lora:
        from peft import get_peft_model, AdaLoraConfig, TaskType
        assert "llama" in args.model
        peft_config = AdaLoraConfig(task_type=TaskType.FEATURE_EXTRACTION, inference_mode=False, r=128, lora_alpha=16, target_modules=["q_proj", "v_proj"])
        base_model = get_peft_model(base_model, peft_config)
    print (base_model)
    reward_model = model_utils.RewardModel(base_model, tokenizer).to(DEVICE)
    optimizer = torch.optim.AdamW(reward_model.parameters(), lr=2e-5, betas=(0.9, 0.95))
    lr_scheduler = get_scheduler(name='cosine', optimizer=optimizer, num_warmup_steps=min(100,0.1*len(train_llm_loader)), num_training_steps=len(train_llm_loader)*2)

    for epoch in range(1):
        reward_model.train()
        tot_loss = 0.0
        tot_cscore = 0.0
        tot_rscore = 0.0
        tot_num = 0.0
        all_scores = []
        all_labs = []

        for idx, (llm_batch, human_batch) in enumerate(zip(train_llm_loader, train_human_loader)):
            llm_input_ids = llm_batch['input_ids'].to(DEVICE)
            llm_attention_mask = llm_batch['attention_mask'].to(DEVICE)
            human_input_ids = human_batch['input_ids'].to(DEVICE)
            human_attention_mask = human_batch['attention_mask'].to(DEVICE)

            reward_output = reward_model(llm_input_ids, llm_attention_mask, human_input_ids, human_attention_mask)
            loss = reward_output['loss']
            loss = loss + 1e-4 * (reward_output['chosen_mean_scores']**2+reward_output['rejected_mean_scores']**2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(reward_model.parameters(), 1.0)
            lr_scheduler.step()
            optimizer.step()
            optimizer.zero_grad()

            tot_loss += reward_output['loss'].item()
            tot_cscore += reward_output['chosen_mean_scores'].mean().item()
            tot_rscore += reward_output['rejected_mean_scores'].mean().item()
            tot_num += 1
            all_scores.extend(list(reward_output['chosen_mean_scores'].detach().cpu().numpy()))
            all_labs.extend([1]*len(reward_output['chosen_mean_scores']))
            all_scores.extend(list(reward_output['rejected_mean_scores'].detach().cpu().numpy()))
            all_labs.extend([0]*len(reward_output['rejected_mean_scores']))
            if (idx+1) % 100 == 0:
                print ("Epoch %d, Step %d, avg loss %s, cscore %s, rscore %s"%(epoch, idx, tot_loss/tot_num, tot_cscore/tot_num, tot_rscore/tot_num))
                print ("AUC %.4f"%roc_auc_score(all_labs, all_scores))

        # Evaluate
        reward_model.eval()
        losses = []
        cscores = []
        rscores = []
        for llm_batch, human_batch in tqdm(zip(test_llm_loader, test_human_loader), total=len(test_llm_loader)):
            llm_input_ids = llm_batch['input_ids'].to(DEVICE)
            llm_attention_mask = llm_batch['attention_mask'].to(DEVICE)
            human_input_ids = human_batch['input_ids'].to(DEVICE)
            human_attention_mask = human_batch['attention_mask'].to(DEVICE)

            reward_output = reward_model(llm_input_ids, llm_attention_mask, human_input_ids, human_attention_mask)
            losses.append(reward_output['loss'].item())
            cscores.append(reward_output['chosen_mean_scores'].detach().cpu().numpy())
            rscores.append(reward_output['rejected_mean_scores'].detach().cpu().numpy())

        cscores = np.concatenate(cscores)
        rscores = np.concatenate(rscores)
        auc = roc_auc_score([1]*len(cscores)+[0]*len(rscores), np.concatenate((cscores,rscores)))
        print ("Epoch %d evaluation: loss %.6f, cscore %.6f, rscore %.6f, AUC %.6f"%(epoch, np.mean(losses), np.mean(cscores), np.mean(rscores), auc))
    if args.use_lora:
        reward_model.rwtransformer = reward_model.rwtransformer.merge_and_unload()
    if not os.path.isdir('%s/ckpt'%args.workdir):
        os.mkdir('%s/ckpt/'%args.workdir)
    if not os.path.isdir('%s/ckpt/%s_%s_raw_detector'%(args.workdir, args.dataset, args.model)):
        os.mkdir('%s/ckpt/%s_%s_raw_detector'%(args.workdir, args.dataset, args.model))
    torch.save(reward_model.state_dict(), '%s/ckpt/%s_%s_raw_detector/reward_model.ckpt'%(args.workdir, args.dataset, args.model))


if __name__ == '__main__':
    torch.manual_seed(8888)
    np.random.seed(8888)

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', type=str, default='opt-350m')
    parser.add_argument('--use_lora', action='store_true')
    parser.add_argument('--dataset', type=str, default='PKU')
    parser.add_argument('--gen_dataset', action='store_true')
    parser.add_argument('--workdir', type=str, default='.')
    args = parser.parse_args()
    print (args)

    main(args)
