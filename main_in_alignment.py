import argparse
import numpy as np
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from tqdm import tqdm

from transformers import DataCollatorForLanguageModeling
from datasets import Dataset
from torch.utils.tensorboard import SummaryWriter

import utils
import ds_utils
from pretrain_detector import load_RM_dataset
from eval_utils import evaluate_detection


def main(args):
    device = torch.device("cuda:0") if torch.cuda.is_available()\
                    else torch.device("cpu")

    SAVE_PATH = '%s/ckpt/IAWM_%s_%s_%s_b%s_step%s'%(args.workdir, args.dataset, args.actor_model, args.reward_model, args.batch_size, args.train_steps)   # in-alignment watermark
    if args.do_sample:
        SAVE_PATH = SAVE_PATH + '_dosample'
    if args.paraphraser != '':
        SAVE_PATH = SAVE_PATH + '_' + args.paraphraser
    if args.substitute_ratio != 0.0:
        SAVE_PATH = SAVE_PATH + '_sub%s'%args.substitute_ratio
    if args.other_llm != '':
        SAVE_PATH = SAVE_PATH + '_against%s'%args.other_llm
    if args.save_suffix != '':
        SAVE_PATH = SAVE_PATH + '_' + args.save_suffix
    print ("Save path: %s"%SAVE_PATH)
    if args.local_rank == 0 and not os.path.isdir(SAVE_PATH):
        os.mkdir(SAVE_PATH)
    if args.with_tensorboard:
        writer = SummaryWriter(SAVE_PATH+"/logs")
    if 'llama' in args.actor_model:
        with open("%s/hf.key"%args.workdir) as inf:
            key = inf.readline().strip()
            os.environ['HF_TOKEN'] = key

    RM_tokenizer = utils.get_tokenizer(args.reward_model)
    tokenizer = utils.get_tokenizer(args.actor_model)
    tokenizer.padding_side = "left"
    tokenizer_right_pad = utils.get_tokenizer(args.actor_model)
    tokenizer_right_pad.padding_side = "right"

    train_dataset, test_dataset = load_RM_dataset(args.dataset, args.actor_model, workdir=args.workdir)
    train_prompt_dataset = []
    for line in train_dataset:
        if 'llama' in args.actor_model:
            # Llama tokenizer will do special behaviour to append white space, so need to strip() here to avoid shortcut
            train_prompt_dataset.append({"prompt": line['prompt'], "cont_human": line['cont_human'].strip()})
        else:
            train_prompt_dataset.append({"prompt": line['prompt'], "cont_human": line['cont_human']})
    prompt_max_len = 64
    train_prompt_loader, train_cont_loader = utils.create_prompt_and_cont_loaders(train_prompt_dataset, tokenizer, tokenizer_right_pad, batch_size=args.batch_size, prompt_max_len=prompt_max_len, cont_max_len=args.max_ans_len)
    if args.other_llm != '':
        other_trainset, _ = load_RM_dataset(args.dataset, args.other_llm, workdir=args.workdir)
        if 'llama' in args.actor_model:
            other_ans_set = [{'text':line['cont_llm'].strip()} for line in other_trainset]
        else:
            other_ans_set = [{'text':line['cont_llm']} for line in other_trainset]
        def preprocess(examples):
            return tokenizer_right_pad(examples['text'], max_length=args.max_ans_len)
        other_new_dataset = Dataset.from_list(other_ans_set).map(preprocess, batched=True, remove_columns=['text'])
        other_data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer_right_pad, mlm=False)
        other_dataloader = torch.utils.data.DataLoader(other_new_dataset, batch_size=args.batch_size, collate_fn=other_data_collator)
        other_iter = iter(other_dataloader)

    # load paraphraser if needed
    if args.paraphraser != '':
        from paraphraser import get_paraphraser
        paraphraser = get_paraphraser(args.paraphraser)

    # Load the models
    gtreward_model = ds_utils.get_gtreward_model(args, RM_tokenizer, stage=args.zero_stage)
    gtreward_model.eval()

    reward_model = ds_utils.get_reward_model(args, RM_tokenizer, stage=args.zero_stage)
    reward_model.eval()
    print ("reward model loaded")
    critic_model = ds_utils.get_critic_model(args, RM_tokenizer, stage=args.zero_stage)
    critic_model.train()
    print ("critic model loaded")
    actor_model = ds_utils.get_actor_model(args, tokenizer, stage=args.zero_stage)
    actor_model.train()
    print ("actor model loaded")
    ref_model = ds_utils.get_ref_model(args, tokenizer, stage=args.zero_stage)
    ref_model.eval()
    print ("ref model loaded")


    # Start RLHF
    exp_mini_dataset = utils.MiniDataset(max_size=1, small_batch_size=args.batch_size)
    tot_step = 0
    for _ in range(args.n_epoch):
        if args.train_steps >= 0 and tot_step >= args.train_steps:
            break

        for _, (batch, cont_batch) in tqdm(enumerate(zip(train_prompt_loader, train_cont_loader))):
            if args.train_steps >= 0 and tot_step >= args.train_steps:
                break
            critic_model.train()

            prompt_input_ids = batch['input_ids'].to(device)
            prompt_attention_mask = batch['attention_mask'].to(device)
            prompt_length = prompt_input_ids.shape[1]

            # Generate sequence
            actor_model.eval()
            max_min_length = prompt_length + args.max_ans_len
            kwargs = dict(do_sample=args.do_sample)
            with torch.no_grad():
                seq = actor_model.module.generate(input_ids=prompt_input_ids, attention_mask=prompt_attention_mask, max_length=max_min_length, pad_token_id=tokenizer.pad_token_id, synced_gpus=(args.zero_stage==3), **kwargs)

            # Calculate sequence information
            pad_token_id = tokenizer.pad_token_id
            seq_attention_mask = seq.not_equal(pad_token_id).long()
            seq_attention_mask[:,:prompt_length] = prompt_attention_mask  # added: keep the mask of first BOS token
            actor_model.train()

            # filter empty ans
            ans = seq[:, prompt_length:]
            used_ids = []
            valid_ans_len = (ans != tokenizer.pad_token_id).sum(dim=-1)
            for i in range(len(valid_ans_len)):
                if valid_ans_len[i] <= 1:
                    print ("EMPTY ANSWER GENERATED!")
                else:
                    used_ids.append(i)

            # Process the gt sequence by human
            cont_input_ids = cont_batch['input_ids'][:,1:].to(device)  # ignore the first BOS token  # TODO: check llama
            cont_input_ids = torch.cat([prompt_input_ids, cont_input_ids],dim=1)
            cont_attention_mask = cont_input_ids.not_equal(pad_token_id).long()
            cont_attention_mask[:,:prompt_length] = prompt_attention_mask  # added: keep the mask of first BOS token

            with torch.no_grad():
                output = actor_model(seq, attention_mask=seq_attention_mask)
                output_ref = ref_model(seq, attention_mask=seq_attention_mask)
                wtm_reward_score = reward_model.forward_value(seq, seq_attention_mask, prompt_length=prompt_length)['chosen_end_scores'].detach()
                gtreward_score = gtreward_model.forward_value(seq, seq_attention_mask, prompt_length=prompt_length)['chosen_end_scores'].detach()
                reward_score = args.rlhf_wtm_lamda * wtm_reward_score + (1-args.rlhf_wtm_lamda) * gtreward_score
                values = critic_model.forward_value(seq, seq_attention_mask, return_value_only=True).detach()[:,:-1]
            seq_info = {
                'prompts': prompt_input_ids[used_ids].contiguous(),
                'logprobs': utils.gather_log_probs(output.logits[:,:-1,:],seq[:,1:])[used_ids].contiguous(),
                'ref_logprobs': utils.gather_log_probs(output_ref.logits[:,:-1,:],seq[:,1:])[used_ids].contiguous(),
                'value': values[used_ids].contiguous(),
                'rewards': reward_score[used_ids].contiguous(),
                'input_ids': seq[used_ids].contiguous(),
                'attention_mask': seq_attention_mask[used_ids].contiguous(),
                'human_input_ids': cont_input_ids[used_ids].contiguous(),
                'human_attention_mask': cont_attention_mask[used_ids].contiguous(),
            }

            exp_dataset = exp_mini_dataset.add(seq_info)
            if exp_dataset is not None and len(exp_dataset) >= 1:
                actor_model.gradient_checkpointing_enable()
                # Finetune LLM
                assert len(exp_dataset) == 1
                exp_data = exp_dataset[0]
                actor_loss, critic_loss = utils.train_rlhf(actor_model, critic_model, exp_data, kl_ctl=args.kl_rl)

                if args.kl_eps != 0:
                    good_input_ids = exp_data['input_ids'].to(device)
                    good_mask = exp_data['attention_mask'].to(device)
                    outputs_good = actor_model(good_input_ids, attention_mask=good_mask)
                    with torch.no_grad():
                        outputs_ref = ref_model(good_input_ids, attention_mask=good_mask)
                    prob_p = torch.nn.functional.softmax(outputs_ref.logits, -1)
                    prob_q = torch.nn.functional.softmax(outputs_good.logits, -1)
                    kl_position_loss = -prob_p * torch.log(prob_q+1e-6)
                    position_weight = torch.zeros_like(kl_position_loss)
                    position_weight[:,prompt_length:] = 1
                    position_weight[good_mask==0] = 0
                    position_weight = position_weight / (position_weight.sum(dim=1,keepdim=True)+1e-8)
                    kl_loss = (position_weight*kl_position_loss).sum()
                else:
                    kl_loss = torch.zeros_like(actor_loss)
                all_loss = actor_loss + args.kl_eps * kl_loss

                actor_model.backward(all_loss)
                actor_model.step()
                critic_model.backward(critic_loss)
                critic_model.step()
                if tot_step % 10 == 0:
                    print ("Step %d, actor loss: %.4f, critic loss: %.4f, avg reward: %.4f, benign loss: %.4f"%(tot_step, actor_loss.item(), critic_loss.item(), reward_score.mean().item(), kl_loss.item()))
                if tot_step % 100 == 0:
                    print (tokenizer.decode(exp_data['input_ids'][0]))
                    print (tokenizer.decode(exp_data['human_input_ids'][0]))

                # Finetune the detector
                reward_model.train()
                reward_output = reward_model(exp_data['input_ids'], exp_data['attention_mask'], exp_data['human_input_ids'], exp_data['human_attention_mask'])
                reward_loss = reward_output['loss'] + 1e-4 * (reward_output['chosen_mean_scores']**2+reward_output['rejected_mean_scores']**2).mean()

                if args.paraphraser != '':
                    torch.cuda.empty_cache()
                    prompt_length = exp_data['prompts'].shape[1]
                    inp_list = []
                    eot_flags = []
                    for i in range(len(exp_data['input_ids'])):
                        cur_ids = exp_data['input_ids'][i,prompt_length:]
                        cur_mask = exp_data['attention_mask'][i,prompt_length:]
                        text = tokenizer_right_pad.decode(cur_ids[cur_mask!=0], skip_special_tokens=True)
                        if text.endswith('<|endoftext|>'):
                            inp_list.append(text[:-13])
                            eot_flags.append(True)
                        else:
                            inp_list.append(text)
                            eot_flags.append(False)
                    llm_text_list = paraphraser.batch_paraphrase(inp_list, prefixs=None, bsize=args.batch_size)
                    for i in range(len(llm_text_list)):
                        if eot_flags[i]:
                            llm_text_list[i] = llm_text_list[i] + '<|endoftext|>'
                    if 'llama' in args.actor_model:
                        for i in range(len(llm_text_list)):
                            llm_text_list[i] = llm_text_list[i].strip()
                    new_tok = tokenizer_right_pad(llm_text_list, padding=True, truncation=True, max_length=max_min_length, return_tensors="pt")
                    new_ans_ids = new_tok['input_ids'][:,1:].to(device)
                    llm_para_ids = torch.cat([exp_data['prompts'], new_ans_ids],dim=1)
                    llm_para_mask = llm_para_ids.not_equal(pad_token_id).long()
                    llm_para_mask[:,:prompt_length] = exp_data['attention_mask'][:,:prompt_length]
                    print ("Before para:", tokenizer.decode(exp_data['input_ids'][0]))
                    print ("After para:", tokenizer.decode(llm_para_ids[0]))
                    reward_para_output = reward_model(llm_para_ids, llm_para_mask, exp_data['human_input_ids'], exp_data['human_attention_mask'])
                    reward_para_loss = reward_para_output['loss'].mean()
                    reward_loss = reward_loss + 1.0*reward_para_loss

                if args.substitute_ratio != 0.0:
                    prompt_length = exp_data['prompts'].shape[1]
                    id_size = int((exp_data['input_ids'].shape[1]-prompt_length)*args.substitute_ratio)
                    llm_substitute_ids = []
                    for bid in range(len(exp_data['input_ids'])):
                        chosen_ids = np.random.choice(np.arange(prompt_length,exp_data['input_ids'].shape[1]), size=id_size, replace=False)  # do not choose the first idx (the BOS token)
                        rand_vals = torch.randint(low=0, high=tokenizer.vocab_size, size=(len(chosen_ids),))
                        new_ids = exp_data['input_ids'][bid].clone()
                        new_ids[chosen_ids] = rand_vals.to(new_ids)
                        llm_substitute_ids.append(new_ids)
                    llm_substitute_ids = torch.stack(llm_substitute_ids, 0)

                    reward_sub_output = reward_model(llm_substitute_ids, exp_data['attention_mask'], exp_data['human_input_ids'], exp_data['human_attention_mask'])
                    reward_sub_loss = reward_sub_output['loss'].mean()
                    reward_loss = reward_loss + 1.0*reward_sub_loss

                if args.other_llm != '':
                    other_batch = next(other_iter)
                    other_input_ids = other_batch['input_ids'][:,1:].to(device)  # ignore the first BOS token
                    other_input_ids = torch.cat([prompt_input_ids, other_input_ids],dim=1)
                    other_attention_mask = other_input_ids.not_equal(pad_token_id).long()
                    other_attention_mask[:,:prompt_length] = prompt_attention_mask  # added: keep the mask of first BOS token
                    reward_other_output = reward_model(exp_data['input_ids'], exp_data['attention_mask'], other_input_ids, other_attention_mask)
                    reward_other_loss = reward_other_output['loss'].mean()
                    reward_loss = reward_loss + 1.0*reward_other_loss

                if args.reward_lr != 0:
                    reward_model.backward(reward_loss)
                    reward_model.step()
                if tot_step % 10 == 0:
                    print ("Reward loss: %.4f; cscore %.4f; rscore %.4f"%(reward_loss.item(), reward_output['chosen_mean_scores'].mean().item(), reward_output['rejected_mean_scores'].mean().item()))
                reward_model.eval()


                if args.with_tensorboard:
                    writer.add_scalar('actor_loss', actor_loss.item(), global_step=tot_step)
                    writer.add_scalar('critic_loss', critic_loss.item(), global_step=tot_step)
                    writer.add_scalar('reward_score', reward_score.mean().item(), global_step=tot_step)
                    writer.add_scalar('kl_loss', kl_loss.item(), global_step=tot_step)
                    writer.add_scalar('reward_loss', reward_loss.item(), global_step=tot_step)
                    writer.add_scalar('chosen_score', reward_output['chosen_mean_scores'].mean().item(), global_step=tot_step)
                    writer.add_scalar('rejected_score', reward_output['rejected_mean_scores'].mean().item(), global_step=tot_step)
                actor_model.gradient_checkpointing_disable()

            tot_step += 1

    actor_model.eval()
    actor_model.module.save_pretrained(SAVE_PATH, from_pt=True)
    torch.save(reward_model.module.state_dict(), SAVE_PATH+'/reward_model.ckpt')
    print ("\033[94mSaved to %s\033[0m"%SAVE_PATH)

    print ("Evaluation:")
    reward_model.train()
    evaluate_detection(actor_model, reward_model, tokenizer_right_pad, test_dataset, "opt" in args.actor_model, save_to=SAVE_PATH+"/watermarked.txt", num_tests=1000)



if __name__ == '__main__':
    torch.manual_seed(8888)
    np.random.seed(8888)

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='PKU', choices=['PKU'])
    parser.add_argument('--actor_model', type=str, default='opt-1.3b')
    parser.add_argument('--reward_model', type=str, default='opt-350m')
    parser.add_argument('--n_epoch', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_ans_len', type=int, default=100)
    parser.add_argument('--train_steps', type=int, default=10000)
    parser.add_argument('--do_sample', action='store_true')

    parser.add_argument('--actor_from_deepspeed', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--lora_lr', type=float, default=5e-4)
    parser.add_argument('--critic_lr', type=float, default=5e-6)
    parser.add_argument('--critic_lora_lr', type=float, default=5e-4)
    parser.add_argument('--reward_lr', type=float, default=1e-5)
    parser.add_argument('--reward_lora_lr', type=float, default=0.0)
    parser.add_argument('--reward_with_scheduler', action='store_true')
    parser.add_argument('--use_lora', action='store_true')
    parser.add_argument('--use_critic_lora', action='store_true')
    parser.add_argument('--use_reward_lora', action='store_true')
    parser.add_argument('--kl_eps', type=float, default=0.0)
    parser.add_argument('--kl_rl', type=float, default=0.1)
    parser.add_argument('--sft_eps', type=float, default=0.0)
    parser.add_argument('--rlhf_wtm_lamda', type=float, default=0.5)

    parser.add_argument('--paraphraser', type=str, default='')
    parser.add_argument('--substitute_ratio', type=float, default=0.0)
    parser.add_argument('--other_llm', type=str, default='')

    parser.add_argument('--workdir', type=str, default='.')
    parser.add_argument('--save_suffix', type=str, default='')
    parser.add_argument('--with_tensorboard', action='store_true')

    # below: args for DeepSpeed
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--zero_stage', type=int, default=0, help="0: standard; 3: loaded by layer on each GPU")

    args = parser.parse_args()
    print (args)

    main(args)
