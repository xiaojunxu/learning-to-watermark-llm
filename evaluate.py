import argparse
import numpy as np
import json
import torch

from transformers import AutoModelForCausalLM

import utils
import model_utils
import ds_utils
from pretrain_detector import load_RM_dataset
from eval_utils import evaluate_detection, evaluate_other_model_detection, evaluate_substitution_atk

def main(args):
    device = torch.device("cuda:0") if torch.cuda.is_available()\
                    else torch.device("cpu")
    DO_SAMPLE = True if 'llama' not in args.model_name else False


    # hard-coded model loading part
    assert args.eval_baseline is not None or args.model_path is not None   # either eval baseline or eval model path

    if args.eval_baseline is not None:
        tokenizer = utils.get_tokenizer(args.model_name)
        actor_model = utils.get_model(args.model_name, model_class=AutoModelForCausalLM).train().to(device)
        if args.dataset =="PKU":
            # for PKU: load the RLHF'ed model, which is the model in RM-only evaluation
            if args.model_name == "opt-1.3b":
                actor_model = ds_utils.convert_linear_layer_to_lora(actor_model, part_module_name='decoder.layers.', lora_dim=128)
                actor_model.load_state_dict(torch.load("%s/ckpt/IAWM_PKU_opt-1.3b_opt-350m_b4_step10000_dosample_rmonly/pytorch_model.bin"%args.workdir, map_location='cpu'))
            elif args.model_name == "llama2-7b":
                actor_model = ds_utils.convert_linear_layer_to_lora(actor_model, part_module_name='decoder.layers.', lora_dim=128)
                actor_model.load_state_dict(torch.load("%s/ckpt/IAWM_PKU_llama2-7b_llama2-1.1b_b4_step10000_rmonly/pytorch_model.bin"%args.workdir, map_location='cpu'))
            else:
                raise NotImplementedError()
        actor_model.eval()
        if args.eval_baseline == 'klw':
            from baseline_lib.klw import ActorWrapper, RMWrapper
            actor_engine = ActorWrapper(actor_model, tokenizer)
            reward_model = RMWrapper(actor_model.device, tokenizer)
        elif args.eval_baseline == 'its':
            from baseline_lib.percy import ITSActorWrapper, ITSRMWrapper
            actor_engine = ITSActorWrapper(actor_model, tokenizer)
            reward_model = ITSRMWrapper(tokenizer)
        elif args.eval_baseline == 'exp':
            from baseline_lib.percy import EXPActorWrapper, EXPRMWrapper
            actor_engine = EXPActorWrapper(actor_model, tokenizer)
            reward_model = EXPRMWrapper(tokenizer)
        else:
            raise NotImplementedError()
    else:
        tokenizer = utils.get_tokenizer(args.model_name)
        if "llama" in args.model_name:
            actor_model = utils.get_model(args.model_name, model_class=AutoModelForCausalLM).train().to(device)
        else:
            actor_model = utils.get_model(args.model_name, model_class=AutoModelForCausalLM).train().to(device)
        actor_model = ds_utils.convert_linear_layer_to_lora(actor_model, part_module_name='decoder.layers.', lora_dim=128)
        if not args.model_path.startswith("facebook/") and not args.model_path.startswith("meta-llama/"):
            actor_model.load_state_dict(torch.load(args.model_path+"/pytorch_model.bin", map_location='cpu'))
        actor_model.eval()
        class Engine:
            def __init__(self, model):
                self.module = model
        actor_engine = Engine(model=actor_model)
        if "opt" in args.model_name:
            reward_base_model = utils.get_model("opt-350m")
        elif "llama" in args.model_name:
            reward_base_model = utils.get_model("llama2-1.1b")
        else:
            raise NotImplementedError()
        reward_model = model_utils.RewardModel(reward_base_model, tokenizer).to(device)
        if args.reward_model_path is None:
            reward_model.load_state_dict(torch.load(args.model_path+"/reward_model.ckpt"))
        else:
            reward_model.load_state_dict(torch.load(args.reward_model_path+"/reward_model.ckpt"))
        reward_model.train()

    train_dataset, test_dataset = load_RM_dataset(args.dataset, args.model_name, workdir=args.workdir)

    with torch.no_grad():
        from paraphraser import get_paraphraser
        ppl_model = AutoModelForCausalLM.from_pretrained("facebook/opt-2.7b").to(device).eval()

        std_auc, ppl, fpr_at_90_tpr, fpr_at_99_tpr = evaluate_detection(actor_engine, reward_model, tokenizer, test_dataset, do_sample=DO_SAMPLE, num_tests=args.num_tests, ppl_model=ppl_model)
        del ppl_model
        torch.cuda.empty_cache()

        dipper_aucs = []
        paraphraser = get_paraphraser("dipper")
        for diversity in [0,20,40,60]:
            paraphraser.lex_diversity = paraphraser.order_diversity = diversity
            auc, *_ = evaluate_detection(actor_engine, reward_model, tokenizer, test_dataset, do_sample=DO_SAMPLE, paraphraser=paraphraser, num_tests=args.num_tests, bsize=1)
            print ("^^^ results when diversity=%d ^^^"%diversity)
            dipper_aucs.append(auc)
        del paraphraser.model
        del paraphraser
        torch.cuda.empty_cache()

        pegasus_aucs = []
        paraphraser = get_paraphraser("pegasus")
        for temp in [1.0,1.5,2.0]:
            paraphraser.temp = temp
            auc, *_ = evaluate_detection(actor_engine, reward_model, tokenizer, test_dataset, do_sample=DO_SAMPLE, paraphraser=paraphraser, num_tests=args.num_tests)
            print ("^^^ results when temp=%s ^^^"%temp)
            pegasus_aucs.append(auc)
        del paraphraser.model
        del paraphraser
        torch.cuda.empty_cache()

        substitute_aucs = []
        for ratio in [0.05,0.1,0.2,0.5]:
            auc = evaluate_substitution_atk(actor_engine, reward_model, ratio, tokenizer, test_dataset, do_sample=DO_SAMPLE, num_tests=args.num_tests)
            print ("^^^ results when ratio=%s ^^^"%ratio)
            substitute_aucs.append(auc)
        
        if args.model_name == "opt-1.3b":
            other_tokenizer = utils.get_tokenizer("llama2-7b")
            other_model = utils.get_model("llama2-7b", model_class=AutoModelForCausalLM).train().to(device)
        elif args.model_name == "llama2-7b":
            other_tokenizer = utils.get_tokenizer("opt-1.3b")
            other_model = utils.get_model("opt-1.3b", model_class=AutoModelForCausalLM).train().to(device)
        other_auc = evaluate_other_model_detection(actor_engine, reward_model, tokenizer, other_model, other_tokenizer, test_dataset, do_sample=DO_SAMPLE, other_do_sample=not DO_SAMPLE, num_tests=args.num_tests)
        print ("^^^ Other auc ^^^")

        all_info = [('std_auc', std_auc), ('fpr@90', fpr_at_90_tpr), ('fpr@99', fpr_at_99_tpr), ('ppl', ppl), ('dipper_aucs', dipper_aucs), ('pegasus_aucs', pegasus_aucs), ('substitute_aucs',substitute_aucs), ('other_auc', other_auc)]
        print (all_info)

        if args.model_path is not None:
            with open(args.model_path+'/full_test.json', 'w') as outf:
                json.dump(all_info, outf)
        else:
            assert args.eval_baseline is not None
            with open('%s/ckpt/%s_%s_%s.json'%(args.workdir, args.model_name, args.eval_baseline, args.dataset), 'w') as outf:
                json.dump(all_info, outf)


if __name__ == '__main__':
    torch.manual_seed(8888)
    np.random.seed(8888)

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--workdir', type=str, default='.')
    parser.add_argument('--eval_baseline', type=str, default=None)
    parser.add_argument('--model_name', type=str, default="opt-1.3b")
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--reward_model_path', type=str, default=None)
    parser.add_argument('--num_tests', type=int, default=100)
    args = parser.parse_args()

    print (args)

    main(args)
