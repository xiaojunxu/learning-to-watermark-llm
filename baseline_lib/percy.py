import torch
import scipy
import numpy as np

PERCY_N = 256
PERCY_SEED = 6666

def reset_seed(i):
    global PERCY_SEED
    PERCY_SEED += i

def percy_generate(model,prompts,attn,vocab_size,n,m,seeds,key_func,sampler,random_offset=True,eos_token_id=-1):
    batch_size = len(prompts)

    generator = torch.Generator()
    xis,pis = [],[]
    for seed in seeds:
        generator.manual_seed(int(seed))
        xi,pi = key_func(generator,n,vocab_size)
        xis.append(xi.unsqueeze(0))
        pis.append(pi.unsqueeze(0))
    xis = torch.vstack(xis)
    pis = torch.vstack(pis)

    # deliberately not controlling this randomness with the generator
    if random_offset:
        offset = torch.randint(n,size=(batch_size,))
    else:
        offset = torch.zeros(size=(batch_size,),dtype=torch.int64)
    inputs = prompts.to(model.device)
    attn = attn.to(model.device)
    #attn = torch.ones_like(inputs)
    past = None
    for i in range(m):
        with torch.no_grad():
            if past:
                output = model(inputs[:,-1:], past_key_values=past, attention_mask=attn)
            else:
                output = model(inputs, attention_mask=attn)

        probs = torch.nn.functional.softmax(output.logits[:,-1], dim=-1).cpu()
        tokens = sampler(probs, pis, xis[torch.arange(batch_size),(offset.squeeze()+i)%n]).to(model.device)
        inputs = torch.cat([inputs, tokens], dim=-1)

        past = output.past_key_values
        attn = torch.cat([attn, attn.new_ones((attn.shape[0], 1))], dim=-1)
    for i in range(len(inputs)):
        eos_ids = (inputs[i][prompts.shape[1]:]==eos_token_id).nonzero()
        if len(eos_ids) != 0:
            inputs[i][eos_ids[0]+prompts.shape[1]:] = eos_token_id

    #return inputs.detach().cpu()
    return inputs.detach()

def fast_permutation_test(tokens,vocab_size,n,k,seed,test_stat,null_results):
    generator = torch.Generator()
    generator.manual_seed(int(seed))

    test_result = test_stat(tokens=tokens,
                            n=n,
                            k=k,
                            generator=generator,
                            vocab_size=vocab_size)
    p_val = torch.searchsorted(null_results,test_result,right=True) / len(null_results)
    return p_val

def phi(tokens,n,k,generator,key_func,vocab_size,dist,null=False,normalize=False):
    if null:
        tokens = torch.unique(tokens, return_inverse=True,sorted=False)[1]
        eff_vocab_size = torch.max(tokens) + 1
    else:
        eff_vocab_size = vocab_size

    xi,pi = key_func(generator,n,vocab_size,eff_vocab_size)
    tokens = torch.argsort(pi)[tokens]
    if normalize:
        tokens = tokens.float() / vocab_size
    
    A = adjacency(tokens,xi,dist,k)
    closest = torch.min(A,axis=1)[0]

    return torch.min(closest)

def adjacency(tokens,xi,dist,k):
    m = len(tokens)
    n = len(xi)
    k = min(k, m)

    A = torch.empty(size=(m-(k-1),n))
    for i in range(m-(k-1)):
        for j in range(n):
            A[i][j] = dist(tokens[i:i+k],xi[(j+torch.arange(k))%n])

    return A

def transform_sampling(probs,pi,xi):
    cdf = torch.cumsum(torch.gather(probs, 1, pi), 1)
    return torch.gather(pi, 1, torch.searchsorted(cdf, xi))

def transform_key_func(generator,n,vocab_size,eff_vocab_size=None):
    pi = torch.randperm(vocab_size, generator=generator)
    xi = torch.rand((n,1), generator=generator)
    return xi,pi

def transform_score(tokens,xi):
    return torch.pow(torch.linalg.norm(tokens-xi.squeeze(),ord=1),1)

class ITSActorWrapper:
    def __init__(self, model, tokenizer):
        model.generate = lambda input_ids, attention_mask, *args, **kwargs: percy_generate(model, input_ids, attention_mask, tokenizer.vocab_size, PERCY_N, 128 if 'max_length' not in kwargs else kwargs['max_length'], [PERCY_SEED]*len(input_ids), transform_key_func, transform_sampling, random_offset=False, eos_token_id=tokenizer.eos_token_id)
        self.module = model

class ITSRMWrapper:
    def __init__(self, tokenizer):
        self.generator = torch.Generator()
        self.vocab_size = tokenizer.vocab_size
        # we only care relative value (AUC) but not p_val, so do not need null results for reference
        #self.null_results = []
        #for text in ?:
        #    text = ?
        #    tokens = tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=2048-buffer_tokens)[0]
        #    null_result = phi(tokens=tokens, n=PERCY_N, k=128, generator=self.generator, key_func=transform_key_func, vocab_size=tokenizer.vocab_size, dist=transform_score, null=True, normalize=True)
        #    self.null_results.append(null_result)

    def __call__(self, llm_ids, llm_mask, human_ids, human_mask):
        output = {}
        output['loss'] = torch.zeros(1)
        output['chosen_mean_scores'] = []
        output['rejected_mean_scores'] = []
        for i in range(len(llm_ids)):
            tokens = llm_ids[i][llm_mask[i]!=0].cpu()
            self.generator.manual_seed(PERCY_SEED)
            phi_val = phi(tokens=tokens, n=PERCY_N, k=128, generator=self.generator, key_func=transform_key_func, vocab_size=self.vocab_size, dist=transform_score, null=False, normalize=True)
            output['chosen_mean_scores'].append(-phi_val)   # phi_val: smaller = more watermarked
        for i in range(len(human_ids)):
            tokens = human_ids[i][human_mask[i]!=0].cpu()
            self.generator.manual_seed(PERCY_SEED)
            phi_val = phi(tokens=tokens, n=PERCY_N, k=128, generator=self.generator, key_func=transform_key_func, vocab_size=self.vocab_size, dist=transform_score, null=False, normalize=True)
            output['rejected_mean_scores'].append(-phi_val)   # phi_val: smaller = more watermarked
        output['chosen_mean_scores'] = torch.FloatTensor(output['chosen_mean_scores'])
        output['rejected_mean_scores'] = torch.FloatTensor(output['rejected_mean_scores'])
        return output


def gumbel_sampling(probs,pi,xi):
    return torch.argmax(xi ** (1/torch.gather(probs, 1, pi)),axis=1).unsqueeze(-1)

def gumbel_key_func(generator,n,vocab_size,eff_vocab_size=None):
    if eff_vocab_size is None:
        eff_vocab_size = vocab_size
        
    pi = torch.arange(eff_vocab_size)
    xi = torch.rand((n,eff_vocab_size), generator=generator)

    return xi,pi

def gumbel_score(tokens,xi):
    xi_samp = torch.gather(xi,-1,tokens.unsqueeze(-1)).squeeze()
    return -torch.sum(torch.log(1/(1-xi_samp)))

class EXPActorWrapper:
    def __init__(self, model, tokenizer):
        model.generate = lambda input_ids, attention_mask, *args, **kwargs: percy_generate(model, input_ids, attention_mask, tokenizer.vocab_size, PERCY_N, 128 if 'max_length' not in kwargs else kwargs['max_length'], [PERCY_SEED]*len(input_ids), gumbel_key_func, gumbel_sampling, random_offset=False, eos_token_id=tokenizer.eos_token_id)
        self.module = model

class EXPRMWrapper:
    def __init__(self, tokenizer):
        self.generator = torch.Generator()
        self.vocab_size = tokenizer.vocab_size
        # we only care relative value (AUC) but not p_val, so do not need null results for reference

    def __call__(self, llm_ids, llm_mask, human_ids, human_mask):
        output = {}
        output['loss'] = torch.zeros(1)
        output['chosen_mean_scores'] = []
        output['rejected_mean_scores'] = []
        for i in range(len(llm_ids)):
            tokens = llm_ids[i][llm_mask[i]!=0].cpu()
            self.generator.manual_seed(PERCY_SEED)
            phi_val = phi(tokens=tokens, n=PERCY_N, k=128, generator=self.generator, key_func=gumbel_key_func, vocab_size=self.vocab_size, dist=gumbel_score, null=False, normalize=False)
            output['chosen_mean_scores'].append(-phi_val)   # phi_val: smaller = more watermarked
        for i in range(len(human_ids)):
            tokens = human_ids[i][human_mask[i]!=0].cpu()
            self.generator.manual_seed(PERCY_SEED)
            phi_val = phi(tokens=tokens, n=PERCY_N, k=128, generator=self.generator, key_func=gumbel_key_func, vocab_size=self.vocab_size, dist=gumbel_score, null=False, normalize=False)
            output['rejected_mean_scores'].append(-phi_val)   # phi_val: smaller = more watermarked
        output['chosen_mean_scores'] = torch.FloatTensor(output['chosen_mean_scores'])
        output['rejected_mean_scores'] = torch.FloatTensor(output['rejected_mean_scores'])
        return output
