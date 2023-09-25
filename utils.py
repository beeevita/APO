"""
https://oai.azure.com/portal/be5567c3dd4d49eb93f58914cccf3f02/deployment
clausa gpt4
"""

import time
import requests
import config
import string
import openai

def parse_sectioned_prompt(s):

    result = {}
    current_header = None

    for line in s.split('\n'):
        line = line.strip()

        if line.startswith('# '):
            # first word without punctuation
            current_header = line[2:].strip().lower().split()[0]
            current_header = current_header.translate(str.maketrans('', '', string.punctuation))
            result[current_header] = ''
        elif current_header is not None:
            result[current_header] += line + '\n'

    return result


def extract_seconds(text, retried=5):
    words = text.split()
    for i, word in enumerate(words):
        if "second" in word:
            return int(words[i - 1])
    return 60

def chatgpt(prompt, temperature=0.7, n=1, top_p=1, stop=None, max_tokens=1024, 
                  presence_penalty=0, frequency_penalty=0, logit_bias={}, timeout=10):
    messages = [{"role": "user", "content": prompt}]
    payload = {
        "prompt": prompt,
        # "engine":"gpt-35-turbo",
        "engine":"text-davinci-003",
        "model": "text-davinci-003",
        # "organization":"mt-gpt-benchmarking",
        # "model": "gpt-3.5-turbo",
        "temperature": temperature,
        "n": n,
        "top_p": top_p,
        "stop": stop,
        "max_tokens": max_tokens,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        # "logit_bias": logit_bias,
    }
    retries = 0
    retried = 0
    while True:
        # try:
        #     response = openai.ChatCompletion.create(**payload)
        #     result = response["choices"][0]["message"]["content"]
        #     break
        try:
            response = openai.Completion.create(**payload)
            response = response["choices"]
            response = [r["text"] for r in response]
            return response

        except Exception as e:
            error = str(e)
            print("retring...", error)
            second = extract_seconds(error, retried)
            retried = retried + 1
            time.sleep(second)
            continue
        # except Exception as e:
        #     # print(e)
        #     seconds = extract_seconds(str(e))
        #     if "filter" not in str(e):
        #         # print("retring...")
        #         time.sleep(seconds)
        #     continue
    return [result]


def instructGPT_logprobs(prompt, temperature=0.7):
    payload = {
        "prompt": prompt,
        "model": "text-davinci-003",
        "temperature": temperature,
        "max_tokens": 1,
        "logprobs": 1,
        "echo": True
    }
    while True:
        try:
            r = requests.post('https://api.openai.com/v1/completions',
                headers = {
                    "Authorization": f"Bearer {config.OPENAI_KEY}",
                    "Content-Type": "application/json"
                },
                json = payload,
                timeout=10
            )  
            if r.status_code != 200:
                time.sleep(2)
                retries += 1
            else:
                break
        except requests.exceptions.ReadTimeout:
            time.sleep(5)
    r = r.json()
    return r['choices']


def read_lines(file_, sample_indices=None):
    ret = []
    if sample_indices:
        sample_indices.sort()
        with open(file_, 'r') as f:
            for i, line in enumerate(f):
                if i in sample_indices:
                    ret.append(line.rstrip())
        return ret
    else:
        with open(file_, 'r') as f:
            lines = f.readlines()
        return [line.rstrip() for line in lines]


def batchify(data, batch_size=16):
    batched_data = []
    for i in range(0, len(data), batch_size):
        batched_data.append(data[i:i + batch_size])
    return batched_data