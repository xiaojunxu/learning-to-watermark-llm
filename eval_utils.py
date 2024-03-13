import numpy as np
import torch
from tqdm import tqdm

from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from sklearn.metrics import roc_auc_score

def calc_fpr_at_tpr(all_labs, all_scores, percentile):
    all_labs = np.array(all_labs)
    all_scores = np.array(all_scores)
    thresh = np.percentile(all_scores[all_labs==1], 100-percentile)
    neg_scores = all_scores[all_labs==0]
    return (neg_scores>thresh).mean()


def evaluate_detection(actor_model, reward_model, tokenizer, test_dataset, do_sample, save_to=None, paraphraser=None, ppl_model=None, num_tests=100, max_length=160, bsize=4):
    # To use pipeline, must set the tokenizer to be left padding.
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    device = torch.device("cuda:0") if torch.cuda.is_available()\
                    else torch.device("cpu")

    ### Test detection AUC
    tot_loss = 0.0
    tot_cscore = 0.0
    tot_rscore = 0.0
    tot_num = 0.0
    tot_ppl = 0.0
    all_scores = []
    all_labs = []
    saved_info = []

    text_generator = pipeline('text-generation', model=actor_model.module, do_sample=do_sample, tokenizer=tokenizer, max_length=max_length, device="cuda:0", num_beams=1) ###
    for idx, response in enumerate(tqdm(text_generator(KeyDataset(test_dataset[:num_tests],'prompt'), batch_size=bsize))):
        line = test_dataset[idx]
        llm_gen = response[0]['generated_text']
        prompt_inp = tokenizer(line['prompt'], return_tensors='pt')

        human_inp = tokenizer(line['prompt']+line['cont_human'], return_tensors='pt')
        llm_inp = tokenizer(llm_gen, return_tensors='pt')

        if paraphraser is not None:
            llm_text = tokenizer.decode(llm_inp['input_ids'][0], skip_special_tokens=True)
            prompt = line['prompt']
            llm_ans = llm_text[len(prompt):]
            if len(llm_ans) == 0:
                llm_ans = '.'
            llm_new_ans = paraphraser.paraphrase(llm_ans)
            llm_inp = tokenizer(prompt+llm_new_ans, return_tensors='pt')

        reward_output = reward_model(llm_inp['input_ids'].to(device), llm_inp['attention_mask'].to(device), human_inp['input_ids'].to(device), human_inp['attention_mask'].to(device))
        if ppl_model is not None:
            ppl_output = ppl_model(input_ids=llm_inp['input_ids'].to(device), attention_mask=llm_inp['attention_mask'].to(device), labels=llm_inp['input_ids'].to(device))
            ppl = torch.exp(ppl_output.loss).item()
            tot_ppl += ppl
        if idx < 10:
            print ("=======Prompt %d======="%idx)
            print (line['prompt'])
            print ("Human:", reward_output['rejected_mean_scores'][0])
            print (tokenizer.decode(human_inp['input_ids'][0]))
            print ("LLM:", reward_output['chosen_mean_scores'][0])
            print (tokenizer.decode(llm_inp['input_ids'][0]))
            if ppl_model is not None:
                print (ppl)
        tot_loss += reward_output['loss'].item()
        tot_cscore += reward_output['chosen_mean_scores'].mean().item()
        tot_rscore += reward_output['rejected_mean_scores'].mean().item()
        tot_num += 1
        all_scores.extend(list(reward_output['chosen_mean_scores'].detach().float().cpu().numpy()))
        all_labs.extend([1]*len(reward_output['chosen_mean_scores']))
        all_scores.extend(list(reward_output['rejected_mean_scores'].detach().float().cpu().numpy()))
        all_labs.extend([0]*len(reward_output['rejected_mean_scores']))
        saved_info.append((line['prompt'], reward_output['rejected_mean_scores'][0].mean().item(), tokenizer.decode(human_inp['input_ids'][0]), reward_output['chosen_mean_scores'][0].mean().item(), tokenizer.decode(llm_inp['input_ids'][0])))
    fpr_at_90_tpr = calc_fpr_at_tpr(all_labs, all_scores, 90)
    fpr_at_99_tpr = calc_fpr_at_tpr(all_labs, all_scores, 99)
    if ppl_model is not None:
        print ("Evaluation: loss %.6f, cscore %.6f, rscore %.6f, AUC %.6f, fpr@90tpr: %.6f, fpr@99tpr: %.6f, ppl: %.6f"%(tot_loss/tot_num, tot_cscore/tot_num, tot_rscore/tot_num, roc_auc_score(all_labs, all_scores), fpr_at_90_tpr, fpr_at_99_tpr, tot_ppl/tot_num))
    else:
        print ("Evaluation: loss %.6f, cscore %.6f, rscore %.6f, AUC %.6f, fpr@90tpr: %.6f, fpr@99tpr: %.6f"%(tot_loss/tot_num, tot_cscore/tot_num, tot_rscore/tot_num, roc_auc_score(all_labs, all_scores), fpr_at_90_tpr, fpr_at_99_tpr))
    if save_to is not None:
        with open(save_to, "w", encoding='utf-8') as outf:
            for line in saved_info:
                outf.write("Prompt: %s\n"%line[0])
                outf.write("Human score: %s\n"%line[1])
                outf.write("Human output: %s\n"%line[2])
                outf.write("LLM score: %s\n"%line[3])
                outf.write("LLM output: %s\n"%line[4])
                outf.write("\n\n")
            if ppl_model is not None:
                outf.write("Evaluation: loss %.6f, cscore %.6f, rscore %.6f, AUC %.6f, fpr@90tpr: %.6f, fpr@99tpr: %.6f, ppl: %.6f\n"%(tot_loss/tot_num, tot_cscore/tot_num, tot_rscore/tot_num, roc_auc_score(all_labs, all_scores), fpr_at_90_tpr, fpr_at_99_tpr, tot_ppl/tot_num))
            else:
                outf.write("Evaluation: loss %.6f, cscore %.6f, rscore %.6f, AUC %.6f, fpr@90tpr: %.6f, fpr@99tpr: %.6f\n"%(tot_loss/tot_num, tot_cscore/tot_num, tot_rscore/tot_num, roc_auc_score(all_labs, all_scores), fpr_at_90_tpr, fpr_at_99_tpr))

    tokenizer.padding_side = original_padding_side
    if ppl_model is not None:
        return roc_auc_score(all_labs, all_scores), tot_ppl/tot_num, fpr_at_90_tpr, fpr_at_99_tpr
    else:
        return roc_auc_score(all_labs, all_scores), fpr_at_90_tpr, fpr_at_99_tpr


def evaluate_other_model_detection(actor_model, reward_model, tokenizer, other_model, other_tokenizer, test_dataset, do_sample, other_do_sample, save_to=None, ppl_model=None, num_tests=100, max_length=160, bsize=4):
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    original_other_padding_side = other_tokenizer.padding_side
    other_tokenizer.padding_side = "left"

    device = torch.device("cuda:0") if torch.cuda.is_available()\
                    else torch.device("cpu")
    ### Test detection AUC
    tot_loss = 0.0
    tot_cscore = 0.0
    tot_rscore = 0.0
    tot_num = 0.0
    tot_ppl = 0.0
    all_scores = []
    all_labs = []
    text_generator = pipeline('text-generation', model=actor_model.module, do_sample=do_sample, tokenizer=tokenizer, max_length=max_length, device="cuda:0", num_beams=1)
    other_generator = pipeline('text-generation', model=other_model, do_sample=other_do_sample, tokenizer=other_tokenizer, max_length=max_length, device="cuda:0", num_beams=1)
    saved_info = []
    key_dataset = KeyDataset(test_dataset[:num_tests],'prompt')

    llm_all = []
    for response in tqdm(text_generator(key_dataset, batch_size=bsize), desc="llm gen"):
        llm_all.append(response[0]['generated_text'])
    human_all = []
    for response in tqdm(other_generator(key_dataset, batch_size=bsize), desc="other gen"):
        human_all.append(response[0]['generated_text'])

    for idx, (llm_gen, human_gen) in enumerate( zip(tqdm(llm_all), human_all) ):
        line = test_dataset[idx]

        prompt_inp = tokenizer(line['prompt'], return_tensors='pt')
        human_inp = tokenizer(human_gen, return_tensors='pt')
        llm_inp = tokenizer(llm_gen, return_tensors='pt')

        reward_output = reward_model(llm_inp['input_ids'].to(device), llm_inp['attention_mask'].to(device), human_inp['input_ids'].to(device), human_inp['attention_mask'].to(device))
        if ppl_model is not None:
            ppl_output = ppl_model(input_ids=llm_inp['input_ids'].to(device), attention_mask=llm_inp['attention_mask'].to(device), labels=llm_inp['input_ids'].to(device))
            ppl = torch.exp(ppl_output.loss).item()
            tot_ppl += ppl
        if idx < 10:
            print ("=======Prompt %d======="%idx)
            print (line['prompt'])
            print ("Human:", reward_output['rejected_mean_scores'][0])
            print (tokenizer.decode(human_inp['input_ids'][0]))
            print ("LLM:", reward_output['chosen_mean_scores'][0])
            print (tokenizer.decode(llm_inp['input_ids'][0]))
            if ppl_model is not None:
                print (ppl)
        tot_loss += reward_output['loss'].item()
        tot_cscore += reward_output['chosen_mean_scores'].mean().item()
        tot_rscore += reward_output['rejected_mean_scores'].mean().item()
        tot_num += 1
        all_scores.extend(list(reward_output['chosen_mean_scores'].detach().float().cpu().numpy()))
        all_labs.extend([1]*len(reward_output['chosen_mean_scores']))
        all_scores.extend(list(reward_output['rejected_mean_scores'].detach().float().cpu().numpy()))
        all_labs.extend([0]*len(reward_output['rejected_mean_scores']))
        saved_info.append((line['prompt'], reward_output['rejected_mean_scores'][0].mean().item(), tokenizer.decode(human_inp['input_ids'][0]), reward_output['chosen_mean_scores'][0].mean().item(), tokenizer.decode(llm_inp['input_ids'][0])))
    if ppl_model is not None:
        print ("Evaluation: loss %.6f, cscore %.6f, rscore %.6f, AUC %.6f, ppl: %.6f"%(tot_loss/tot_num, tot_cscore/tot_num, tot_rscore/tot_num, roc_auc_score(all_labs, all_scores), tot_ppl/tot_num))
    else:
        print ("Evaluation: loss %.6f, cscore %.6f, rscore %.6f, AUC %.6f"%(tot_loss/tot_num, tot_cscore/tot_num, tot_rscore/tot_num, roc_auc_score(all_labs, all_scores)))
    if save_to is not None:
        with open(save_to, "w", encoding='utf-8') as outf:
            for line in saved_info:
                outf.write("Prompt: %s\n"%line[0])
                outf.write("Human score: %s\n"%line[1])
                outf.write("Human output: %s\n"%line[2])
                outf.write("LLM score: %s\n"%line[3])
                outf.write("LLM output: %s\n"%line[4])
                outf.write("\n\n")
            if ppl_model is not None:
                outf.write("Evaluation: loss %.6f, cscore %.6f, rscore %.6f, AUC %.6f, ppl: %.6f\n"%(tot_loss/tot_num, tot_cscore/tot_num, tot_rscore/tot_num, roc_auc_score(all_labs, all_scores), tot_ppl/tot_num))
            else:
                outf.write("Evaluation: loss %.6f, cscore %.6f, rscore %.6f, AUC %.6f\n"%(tot_loss/tot_num, tot_cscore/tot_num, tot_rscore/tot_num, roc_auc_score(all_labs, all_scores)))

    tokenizer.padding_side = original_padding_side
    other_tokenizer.padding_side = original_other_padding_side
    return roc_auc_score(all_labs, all_scores)


def evaluate_substitution_atk(actor_model, reward_model, ratio, tokenizer, test_dataset, do_sample, save_to=None, num_tests=100, max_length=160, bsize=4):
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    device = torch.device("cuda:0") if torch.cuda.is_available()\
                    else torch.device("cpu")

    ### Test detection AUC
    tot_loss = 0.0
    tot_cscore = 0.0
    tot_rscore = 0.0
    tot_num = 0.0
    tot_ppl = 0.0
    all_scores = []
    all_labs = []
    text_generator = pipeline('text-generation', model=actor_model.module, do_sample=do_sample, tokenizer=tokenizer, max_length=max_length, device="cuda:0", num_beams=1) ###
    saved_info = []
    for idx, response in enumerate(tqdm(text_generator(KeyDataset(test_dataset[:num_tests],'prompt'), batch_size=bsize))):
        line = test_dataset[idx]
        llm_gen = response[0]['generated_text']
        prompt_inp = tokenizer(line['prompt'], return_tensors='pt')
        human_inp = tokenizer(line['prompt']+line['cont_human'], return_tensors='pt')
        llm_inp = tokenizer(llm_gen, return_tensors='pt')

        # random substitution
        st_idx = prompt_inp['input_ids'].shape[1]

        id_size = int((llm_inp['input_ids'].shape[1]-st_idx)*ratio)
        chosen_ids = np.random.choice(np.arange(st_idx,llm_inp['input_ids'].shape[1]), size=id_size, replace=False)  # do not choose the first idx (the BOS token)
        rand_vals = torch.randint(low=0, high=tokenizer.vocab_size, size=(len(chosen_ids),))
        assert llm_inp['input_ids'].shape[0] == 1
        llm_inp['input_ids'][0,chosen_ids] = rand_vals

        reward_output = reward_model(llm_inp['input_ids'].to(device), llm_inp['attention_mask'].to(device), human_inp['input_ids'].to(device), human_inp['attention_mask'].to(device))
        if idx < 10:
            print ("=======Prompt %d======="%idx)
            print (line['prompt'])
            print ("Human:", reward_output['rejected_mean_scores'][0])
            print (tokenizer.decode(human_inp['input_ids'][0]))
            print ("LLM:", reward_output['chosen_mean_scores'][0])
            print (tokenizer.decode(llm_inp['input_ids'][0]))
        tot_loss += reward_output['loss'].item()
        tot_cscore += reward_output['chosen_mean_scores'].mean().item()
        tot_rscore += reward_output['rejected_mean_scores'].mean().item()
        tot_num += 1
        all_scores.extend(list(reward_output['chosen_mean_scores'].detach().float().cpu().numpy()))
        all_labs.extend([1]*len(reward_output['chosen_mean_scores']))
        all_scores.extend(list(reward_output['rejected_mean_scores'].detach().float().cpu().numpy()))
        all_labs.extend([0]*len(reward_output['rejected_mean_scores']))
        saved_info.append((line['prompt'], reward_output['rejected_mean_scores'][0].mean().item(), tokenizer.decode(human_inp['input_ids'][0]), reward_output['chosen_mean_scores'][0].mean().item(), tokenizer.decode(llm_inp['input_ids'][0])))
    print ("Evaluation: loss %.6f, cscore %.6f, rscore %.6f, AUC %.6f"%(tot_loss/tot_num, tot_cscore/tot_num, tot_rscore/tot_num, roc_auc_score(all_labs, all_scores)))
    if save_to is not None:
        with open(save_to, "w", encoding='utf-8') as outf:
            for line in saved_info:
                outf.write("Prompt: %s\n"%line[0])
                outf.write("Human score: %s\n"%line[1])
                outf.write("Human output: %s\n"%line[2])
                outf.write("LLM score: %s\n"%line[3])
                outf.write("LLM output: %s\n"%line[4])
                outf.write("\n\n")
            outf.write("Evaluation: loss %.6f, cscore %.6f, rscore %.6f, AUC %.6f\n"%(tot_loss/tot_num, tot_cscore/tot_num, tot_rscore/tot_num, roc_auc_score(all_labs, all_scores)))

    tokenizer.padding_side = original_padding_side
    return roc_auc_score(all_labs, all_scores)
