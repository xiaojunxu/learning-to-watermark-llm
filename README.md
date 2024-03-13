## Learning to Watermark LLM

Released code for the paper [Learning to Watermark LLM-generated Text via Reinforcement Learning](https://arxiv.org/pdf/TODO.pdf)

### Prerequisites
* Tested on Python 3.9 with PyTorch 1.13.1 on a A100-80GB with 256GB memory.
* The PyTorch package needs to be installed depending on the hardware and CUDA version.
* Other dependencies can be installed by `pip install -r requirements.txt`.

We can watermark two types of models: completion models and Q&A (instruction-finetuned) models.

### Watermarking Prompt Completion Models

Train a watermarked OPT-1.3b model with a paired OPT-350m detector on the c4 dataset:
```bash
python pretrain_detector.py --model opt-350m --dataset c4 --gen_dataset  # Pretraining step for the detector
deepspeed --num_gpus 1 main.py --actor_model opt-1.3b --reward_model opt-350m --do_sample --reward_with_scheduler --use_lora --with_tensorboard
```

Other settings:
* To watermark Llama models, replace `opt-1.3b` with `llama2-7b` and replace `opt-350m` with `llama2-1.1b`.
* To run the setting without finetuning (of the original model), i.e. training the detector only, set `--lr 0 --lora_lr 0`.
* To run the word substitution adv training, set `--substitute_ratio 0.2`.
* To run the paraphrasing adv training, set `--paraphraser pegasus1.5`.
* To run the training with both human and LLM text (i.e.  `H+L` in Table 2), set `--other_llm llama2-7b`.

### Watermarking Q&A Models together with Alignment

There are two extra steps when adding watermark during the alignment tasks (experiments using PKU alignment data in the paper). First, we need to SFT the model to follow the first step in the conventional alignment pipeline:
```
python pretrain_sft.py --model opt-1.3b --learn_steps 10000 --use_lora
```

Next, we pretrain the detector as before:
```bash
python pretrain_detector.py --model opt-350m --dataset PKU --gen_dataset  # Pretraining step for the detector
```

Then we need a reward model to RLHF the model and embed the watermark while training the detector. You can follow the [script in DeepSpeed examples](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat/training/step2_reward_model_finetuning) to train a reward model. Alternatively, you can write a script similar to `pretrain_detector.py` to get a model checkpoint for reward models on the PKU dataset.

Either way, the next step assumes you have put the reward model checkpoint under `./deepspeed_ckpt/opt-350m`. Then we run the co-training:
```bash
deepspeed --num_gpus 1 main_in_alignment.py --actor_model opt-1.3b --reward_model opt-350m --do_sample --reward_with_scheduler --use_lora --with_tensorboard --rlhf_wtm_lamda 0.5
```

Other settings are the same as before.

### Evaluation
The training script above should return the detection AUC without perturbation. To evaluate the model performance under different perturbation:
```bash
python evaluate.py --dataset {dataset} --model_path {path_to_model}
```
