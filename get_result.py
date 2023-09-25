import numpy as np
import argparse

parser = argparse.ArgumentParser(description="src or tgt")
parser.add_argument('--path', '-p', required=True)

args = parser.parse_args()

def cal_mean_std(results):
    mean = np.mean(results)
    std = np.std(results)
    return round(mean, 2), round(std, 2)

def cal_test_result_3seed(path):
    scores = []
    for seed in [5, 10, 15]:
        with open(f'{path}/seed{seed}/test.txt') as f:
            line = f.readlines()[0].strip()
            score = line.split('\t')[-1]
            print(f'seed {seed}: {score}')
            score = float(score)
            if score < 1.0:
                score *= 100
            scores.append(score)
    print('mean, std:')
    print(cal_mean_std(scores))

def cal_dev_result_3seed(path):
    for seed in [5, 10, 15]:
        scores = []
        best_score = 0
        with open(f'{path}/seed{seed}/step10.txt') as f:
            texts = f.readlines()
            for line in texts:
                line = line.strip()
                score = line.split('\t')[-1]
                score = float(score)
                if score > best_score:
                    best_score = score
                scores.append(score)
            avg_score = sum(scores) / len(scores)
            # best_score = line[0].strip().split('best score: ')[-1]
            # avg_score = line[1].strip().split('average score: ')[-1]
            print(f'seed {seed}: best score: {best_score}, average score: {avg_score}')

if __name__ == '__main__':
    print('dev result:\n')
    cal_dev_result_3seed(args.path)
    print('test result: \n')
    cal_test_result_3seed(args.path)

