import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from nltk.tokenize import sent_tokenize
import time

def get_paraphraser(p_name):
    if p_name == 'dipper':
        return DipperParaphraser()
    elif p_name.startswith('dipper'):
        div = float(p_name[6:])
        return DipperParaphraser(div=div)
    elif p_name == 'pegasus':
        return PegasusParaphraser()
    elif p_name.startswith('pegasus'):
        temp = float(p_name[7:])
        return PegasusParaphraser(temp=temp)
    else:
        raise NotImplementedError("Unknown paraphraser: %s"%args.paraphraser)

class DipperParaphraser(object):
    def __init__(self, model="kalpeshk2011/dipper-paraphraser-xxl", verbose=True, div=20):
        time1 = time.time()
        self.tokenizer = T5Tokenizer.from_pretrained('google/t5-v1_1-xxl')
        self.model = T5ForConditionalGeneration.from_pretrained(model).cuda()
        if verbose:
            print(f"{model} model loaded in {time.time() - time1}")
        self.model.eval()
        self.lex_diversity = div
        self.order_diversity = div

    def paraphrase(self, input_text, prefix="", sent_interval=3, **kwargs):
        """Paraphrase a text using the DIPPER model.

        Args:
            input_text (str): The text to paraphrase. Make sure to mark the sentence to be paraphrased between <sent> and </sent> blocks, keeping space on either side.
            lex_diversity (int): The lexical diversity of the output, choose multiples of 20 from 0 to 100. 0 means no diversity, 100 means maximum diversity.
            order_diversity (int): The order diversity of the output, choose multiples of 20 from 0 to 100. 0 means no diversity, 100 means maximum diversity.
            **kwargs: Additional keyword arguments like top_p, top_k, max_length.
        """
        assert self.lex_diversity in [0, 20, 40, 60, 80, 100], "Lexical diversity must be one of 0, 20, 40, 60, 80, 100."
        assert self.order_diversity in [0, 20, 40, 60, 80, 100], "Order diversity must be one of 0, 20, 40, 60, 80, 100."

        kwargs['do_sample']=True
        kwargs['top_p']=0.75
        kwargs['top_k']=None
        kwargs['max_length']=256

        lex_code = int(100 - self.lex_diversity)
        order_code = int(100 - self.order_diversity)

        input_text = " ".join(input_text.split())
        sentences = sent_tokenize(input_text)
        prefix = " ".join(prefix.replace("\n", " ").split())
        output_text = ""

        for sent_idx in range(0, len(sentences), sent_interval):
            curr_sent_window = " ".join(sentences[sent_idx:sent_idx + sent_interval])
            final_input_text = f"lexical = {lex_code}, order = {order_code}"
            if prefix:
                final_input_text += f" {prefix}"
            final_input_text += f" <sent> {curr_sent_window} </sent>"

            final_input = self.tokenizer([final_input_text], return_tensors="pt")
            final_input = {k: v.cuda() for k, v in final_input.items()}

            with torch.inference_mode():
                outputs = self.model.generate(**final_input, **kwargs)
            outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            prefix += " " + outputs[0]
            output_text += " " + outputs[0]

        return output_text

    def batch_paraphrase(self, input_texts, prefixs=None, sent_interval=3, **kwargs):
        texts = []
        if prefixs is None:
            prefixs = ["" for _ in range(len(input_texts))]
        for input_text, prefix in zip(input_texts, prefixs):
            out = self.paraphrase(input_text, prefix, sent_interval, **kwargs)
            texts.append(out)
        return texts

class PegasusParaphraser:
    def __init__(self, temp=2.0):
        model_name = 'tuner007/pegasus_paraphrase'
        self.tokenizer = PegasusTokenizer.from_pretrained(model_name)
        self.model = PegasusForConditionalGeneration.from_pretrained(model_name).to("cuda:0")
        self.temp = temp

    def paraphrase(self, input_text, prefix=''):
        input_text = " ".join(input_text.split())
        sentences = sent_tokenize(input_text)

        batch = self.tokenizer(sentences, max_length=60, padding='longest', truncation=True, return_tensors="pt").to("cuda:0")
        para_toks = self.model.generate(**batch, do_sample=True, max_length=60, num_return_sequences=1, temperature=self.temp)
        out_text = " " if input_text.startswith(" ") else ""
        for one_tok in para_toks:
            new_text = self.tokenizer.decode(one_tok, skip_special_tokens=True)
            out_text = out_text + " " + new_text

        return out_text

    def batch_paraphrase(self, input_texts, prefixs=None, bsize=16):
        all_sent = []
        st_id = []
        ed_id = []
        for input_text in input_texts:
            st_id.append(len(all_sent))
            input_text = " ".join(input_text.split())
            sentences = sent_tokenize(input_text)
            all_sent.extend(sentences)
            ed_id.append(len(all_sent))
        batch = self.tokenizer(all_sent, max_length=60, padding='longest', truncation=True, return_tensors="pt").to("cuda:0")
        para_toks = []
        for i in range(0, len(batch['input_ids']), bsize):
            cur_toks = self.model.generate(input_ids=batch['input_ids'][i:i+bsize], attention_mask=batch['attention_mask'][i:i+bsize], do_sample=True, max_length=60, num_return_sequences=1, temperature=self.temp)
            para_toks.append(cur_toks)
        padded_toks = []
        maxlen = max([tok.shape[1] for tok in para_toks])
        for tok in para_toks:
            new_tok = torch.nn.functional.pad(tok, (0,maxlen-tok.shape[1]), value=self.tokenizer.pad_token_id)
            padded_toks.append(new_tok)
        para_toks = torch.cat(padded_toks, dim=0)
        all_decoded = self.tokenizer.batch_decode(para_toks, skip_special_tokens=True)
        out_texts = []
        for i in range(len(input_texts)):
            out_text = " " if input_texts[i].startswith(" ") else ""
            for new_text in all_decoded[st_id[i]:ed_id[i]]:
                out_text = out_text + " " + new_text
            out_texts.append(out_text)
        torch.cuda.empty_cache()
        return out_texts
