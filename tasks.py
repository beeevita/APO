import requests
import json
import concurrent.futures
from abc import ABC, abstractmethod
from typing import List, Dict, Callable
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
from utils import read_lines, batchify

class DataProcessor(ABC):
    def __init__(self, data_dir, max_threads=1):
        self.data_dir = data_dir
        self.max_threads = max_threads

    @abstractmethod
    def get_train_examples(self):
        pass

    @abstractmethod
    def get_test_examples(self):
        pass

    @abstractmethod
    def evaluate(self, predictor, test_exs):
        pass

    @abstractmethod
    def stringify_prediction(self, pred):
        pass







class ClassificationTask(DataProcessor):
    def process_example(ex, predictor, prompt):
        pred = predictor.inference(ex, prompt)
        return ex, pred
    
    def process_example_batch(self,exs, predictor, prompt):
        preds = predictor.inference(exs, prompt)
        return exs, preds

    def run_evaluate(self, predictor, prompt, test_exs, n=100):
        labels = []
        preds = []
        texts = []

        # with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_threads) as executor:
        #     futures = [executor.submit(self.process_example_batch, ex, prompt) for ex in test_exs]
        #     for i, future in tqdm(enumerate(concurrent.futures.as_completed(futures)), total=len(futures), desc='running evaluate'):
        #         ex, pred = future.result()
        #         texts.append(ex['text'])
        #         labels.append(ex['label'])
        #         preds.append(pred)
        texts = [ex['text'] for ex in test_exs]
        texts, preds = self.process_example_batch(texts, predictor, prompt)
        labels = [ex['label'] for ex in test_exs]
        print("length of labels: ", len(labels))
        accuracy = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='micro')
        return accuracy, texts, labels, preds

    def evaluate(self, predictor, prompt, test_exs, n=100):
        while True:
            try:
                f1, texts, labels, preds = self.run_evaluate(predictor, prompt, test_exs, n=n)
                break
            except (concurrent.futures.process.BrokenProcessPool, requests.exceptions.SSLError):
                pass
        return f1, texts, labels, preds


class BinaryClassificationTask(ClassificationTask):
    categories = ['No', 'Yes']

    def stringify_prediction(self, pred):
        return BinaryClassificationTask.categories[pred]


class EthosBinaryTask(BinaryClassificationTask):
    categories = ['No', 'Yes']

    def get_train_examples(self):
        df = pd.read_csv(self.data_dir + '/ethos_ishate_binary_shuf.csv', sep=';', header=None)
        df = df[(df[1] <= 0) | (df[1] >= 0.7)]
        exs = df.reset_index().to_dict('records')
        exs = [{'id': x['index'], 'text': x[0], 'label': 1 if x[1] > 0.4 else 0} for x in exs[200:]]
        return exs
    
    def get_test_examples(self):
        df = pd.read_csv(self.data_dir + '/ethos_ishate_binary_shuf.csv', sep=';', header=None)
        df = df[(df[1] <= 0) | (df[1] >= 0.7)]
        exs = df.reset_index().to_dict('records')
        exs = [{'id': x['index'], 'text': x[0], 'label': 1 if x[1] > 0.4 else 0} for x in exs[:200]]
        return exs


class JailbreakBinaryTask(BinaryClassificationTask):
    categories = ['No', 'Yes']

    def get_train_examples(self):
        exs = []
        for i, l in enumerate(open(self.data_dir + '/train.tsv')):
            convo, label = l.strip().split('\t')
            label = int(label)
            text = ' '.join([x['text'].strip() for x in json.loads(convo) if x['role'] == 'user'])
            exs.append({'id': i, 'text': text, 'label': label})
        return exs
    
    def get_test_examples(self):
        exs = []
        for i, l in enumerate(open(self.data_dir + '/test.tsv')):
            convo, label = l.strip().split('\t')
            label = int(label)
            text = ' '.join([x['text'].strip() for x in json.loads(convo) if x['role'] == 'user'])
            exs.append({'id': i, 'text': text, 'label': label})
        return exs


class DefaultHFBinaryTask(BinaryClassificationTask):
    categories = ['No', 'Yes']

    def get_train_examples(self):
        exs = []
        for i, row in enumerate(open(self.data_dir + '/train.jsonl')):
            row = json.loads(row.strip())
            exs.append({'id': f'train-{i}', 'label': row['label'], 'text': row['text']})
        return exs
    
    def get_test_examples(self):
        exs = []
        for i, row in enumerate(open(self.data_dir + '/test.jsonl')):
            row = json.loads(row.strip())
            exs.append({'id': f'test-{i}', 'label': row['label'], 'text': row['text']})
        return exs

class SST2Task(BinaryClassificationTask):
    categories = ["negative", "positive"] 

    def __init__(self, dev_data, test_data, max_threads):
        super().__init__(dev_data,test_data)
        self.dev_data = dev_data
        self.test_data = test_data
        self.max_threads = max_threads

    def get_train_examples(self):
        test_data = read_lines(self.dev_data)
        exs = []
        for i, line in enumerate(test_data):
            try:
                cur_src, cur_tgt = line.split('\t')
                exs.append({'text': cur_src, 'label': int(cur_tgt)})
            except:
                raise ValueError
        return exs
    
    def get_test_examples(self):
        test_data = read_lines(self.test_data)
        exs = []
        for i, line in enumerate(test_data):
            try:
                cur_src, cur_tgt = line.split('\t')
                exs.append({'text': cur_src, 'label': int(cur_tgt)})
            except:
                raise ValueError
        return exs

    def stringify_prediction(self, pred):
        return SST2Task.categories[pred]


class SubjTask(BinaryClassificationTask):
    categories = ["subjective", "objective"] 

    def __init__(self, dev_data, test_data, max_threads):
        super().__init__(dev_data,test_data)
        self.dev_data = dev_data
        self.test_data = test_data
        self.max_threads = max_threads

    def get_train_examples(self):
        test_data = read_lines(self.dev_data)
        exs = []
        for i, line in enumerate(test_data):
            try:
                cur_src, cur_tgt = line.split('\t')
                exs.append({'text': cur_src, 'label': int(cur_tgt)})
            except:
                raise ValueError
        return exs
    
    def get_test_examples(self):
        test_data = read_lines(self.test_data)
        exs = []
        for i, line in enumerate(test_data):
            try:
                cur_src, cur_tgt = line.split('\t')
                exs.append({'text': cur_src, 'label': int(cur_tgt)})
            except:
                raise ValueError
        return exs

    def stringify_prediction(self, pred):
        return SubjTask.categories[pred]