from abc import ABC, abstractmethod
from typing import List, Dict, Callable
from liquid import Template
import torch
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
)
from datasets import Dataset as Dataset2
import sys
from dataset import TextDataset
from torch.utils.data import DataLoader
import utils
import tasks
from tqdm import tqdm

class GPT4Predictor(ABC):
    def __init__(self, opt):
        self.opt = opt

    @abstractmethod
    def inference(self, ex, prompt):
        pass

class AlpacaPredictor():
    categories = ["negative", "positive"] 
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('loading model...')
        self.tokenizer = LlamaTokenizer.from_pretrained(
            "chavinlo/alpaca-native",
            # use_fast=False,
            padding_side="left",
            # truncation_side="left"
        )
        self.model = LlamaForCausalLM.from_pretrained(
            "chavinlo/alpaca-native",
            load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.model.eval()
        self.model.config.pad_token_id = self.tokenizer.pad_token_id = 0  # unk
        if torch.__version__ >= "2" and sys.platform != "win32":
            self.model = torch.compile(self.model)
        prefix = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n"""

        # sst2
        # self.demon = prefix + """### Instruction:\n<prompt>\n\n### Input:\ngreat phone , i 'd buy another .\n\n### Response:\npositive\n\n### Input:\nit 's just merely very bad .\n\n### Response:\nnegative\n\n### Input:\n<input>\n\n### Response:\n"""
        
        # mr 
        # self.demon = prefix + """### Instruction:\n<prompt>\n\n### Input:\nwise and deadpan humorous .\n\n### Response:\npositive\n\n### Input:\nshamelessly sappy and , worse , runs away from its own provocative theme .\n\n### Response:\nnegative\n\n### Input:\n<input>\n\n### Response:\n"""

        # cr
        self.demon = prefix + """### Instruction:\n<prompt>\n\n### Input:\none of the year 's best films , featuring an oscar-worthy performance by julianne moore .\n\n### Response:\npositive\n\n### Input:\nthis product is absolutely not ready for release .\n\n### Response:\nnegative\n\n### Input:\n<input>\n\n### Response:\n"""

    def load_data(self, ex, prompt):
        if isinstance(prompt, str):
            data_with_prompt = [self.demon.replace("<input>", e).replace("<prompt>", prompt) for e in ex]
        elif isinstance(prompt, list):
            data_with_prompt = [self.demon.replace("<input>", e).replace("<prompt>", p) for e, p in zip(ex, prompt)]
        else:
            raise ValueError("prompt must be either str or list of str")
        dataset = Dataset2.from_dict({"text": data_with_prompt})

        tokenized_datasets = dataset.map(
            lambda examples: self.tokenizer(
                examples["text"],
                truncation=True,
                padding=True,
                return_tensors="pt",
                #     add_special_tokens=True
            ),
            batched=True,
            num_proc=1,
            load_from_cache_file=True,
            desc="Running tokenizer on dataset",
        )
        print(
            "### tokenized_datasets...example: " + tokenized_datasets["text"][0]
        )

        dataset = TextDataset(tokenized_datasets, self.tokenizer)
        data_loader = DataLoader(
            dataset,
            batch_size=16,
            shuffle=False,
            num_workers=0,
            collate_fn=dataset.collater,
        )
        return iter(data_loader)

    def inference(self, ex, prompt):
        dataset = self.load_data(ex, prompt)
        all_test_data = []
        hypos = []
        try:
            while True:
                cond = next(dataset)
                all_test_data.append(cond)
        except StopIteration:
            # self.logger.info('### End of Loading datasets...')
            pass
        with torch.no_grad():
            for cond in tqdm(all_test_data, desc="Running inference"):
                input_ids_x = cond.pop("input_ids").to(self.device)
                input_ids_mask = cond.pop("attention_mask").to(self.device)
                prompt_len = cond.pop("prompt_len")

                generate_ids = self.model.generate(
                    input_ids=input_ids_x,
                    max_new_tokens=64,
                    attention_mask=input_ids_mask,
                )
                generate_ids = generate_ids[:, prompt_len:-1]
                preds = self.tokenizer.batch_decode(
                    generate_ids,
                    skip_special_tokens=True,
                )
                preds = [1 if 'positive' in pred.lower() else 0 for pred in preds]
                hypos.extend(preds)
        return hypos


class BinaryPredictor(GPT4Predictor):
    categories = ['No', 'Yes']

    def inference(self, ex, prompt):
        prompt = Template(prompt).render(text=ex['text'])
        response = utils.chatgpt(
            prompt, max_tokens=4, n=1, timeout=2, 
            temperature=self.opt['temperature'])[0]
        pred = 1 if response.strip().upper().startswith('YES') else 0
        return pred

class BinarySentPredictor(GPT4Predictor):
    categories = ["negative", "positive"] 

    def inference(self, ex, prompt):
        prompt = Template(prompt).render(text=ex['text'])
        response = utils.chatgpt(
            prompt, max_tokens=4, n=1, timeout=2, 
            temperature=self.opt['temperature'])[0]
        pred = 1 if response.strip().lower().startswith('positive') else 0
        return pred


class SubjAlpacaPredictor(AlpacaPredictor):
    categories = ["subjective", "objective"] 
    def __init__(self, opt):
        super(SubjAlpacaPredictor, self).__init__(opt)
        prefix = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n"""
        self.demon = prefix + """### Instruction:\n<prompt>\n\n### Input:\na cold-hearted judge finds out when a seemingly crazy young couple break into his house and take him captive .\n\n### Response:\nobjective\n\n### Input:\na heartening tale of small victories and enduring hope .\n\n### Response:\nsubjective\n\n### Input:\n<input>\n\n### Response:\n"""

    def inference(self, ex, prompt):
        dataset = self.load_data(ex, prompt)
        all_test_data = []
        hypos = []
        try:
            while True:
                cond = next(dataset)
                all_test_data.append(cond)
        except StopIteration:
            # self.logger.info('### End of Loading datasets...')
            pass
        with torch.no_grad():
            for cond in tqdm(all_test_data, desc="Running inference"):
                input_ids_x = cond.pop("input_ids").to(self.device)
                input_ids_mask = cond.pop("attention_mask").to(self.device)
                prompt_len = cond.pop("prompt_len")

                generate_ids = self.model.generate(
                    input_ids=input_ids_x,
                    max_new_tokens=64,
                    attention_mask=input_ids_mask,
                )
                generate_ids = generate_ids[:, prompt_len:-1]
                preds = self.tokenizer.batch_decode(
                    generate_ids,
                    skip_special_tokens=True,
                )
                preds = [1 if 'objective' in pred.lower() else 0 for pred in preds]
                hypos.extend(preds)
        return hypos