from myabc.utils import get_model_and_tokenizer, get_env_conf, gradient_color
from corpus import get_processor, LazyRandomSampleCorpus
import argparse
import torch


def build_dataset(env_conf, tokenizer):
    proc = get_processor(env_conf['eval']['config'], tokenizer)
    corp = LazyRandomSampleCorpus(env_conf['eval']['file_path'], proc, max_instance=env_conf['eval']['max_instance'])
    return corp


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_conf", type=str)
    args = parser.parse_args()

    env_conf = get_env_conf(args.env_conf)
    model, tokenizer = get_model_and_tokenizer(**env_conf['model'])
    
    corpus = build_dataset(env_conf, tokenizer)


    for data in corpus:
        input_ids = data['input_ids']
        labels = input_ids[1:] + [-100]

        input_ids = torch.tensor(input_ids, dtype=torch.int64, device='cuda').unsqueeze(0)
        labels = torch.tensor(labels, dtype=torch.int64, device='cuda').unsqueeze(0)

        outs = model(input_ids=input_ids, labels=labels)
        outs_bl = model(input_ids=input_ids, labels=labels, enable_prune=False)

        print(f"loss:{outs['loss'].item():.3f}, loss(baseline): {outs_bl['loss'].item():.3f}")
        for m, tok in zip(outs['mask'], data['input_ids'][1:]):
            print(gradient_color(tokenizer.decode(tok) + ' ' if tok != 13 else '\\n', int(m)), end='')
        print()
        print('=' * 40)
