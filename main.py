import argparse
from datasets import preprocess_vote
parser = argparse.ArgumentParser()

### run--> python3 main.py --dataset vote
parser.add_argument("--dataset", type=str, default='vote')
con = parser.parse_args()

def configuration():
    config = {
                'dataset':con.dataset
             }
    return config

def main():
    config = configuration()

    if config['dataset'] == 'vote':
        df = preprocess_vote()
        print(df.head())


if __name__ == '__main__':
	main()