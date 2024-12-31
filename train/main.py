from torch.utils.data import ConcatDataset, DataLoader
import torch.distributed as dist

import torch


from corpus import get_processor, RandomSampleCorpus
from myabc.utils import (
    get_model_and_tokenizer, 
    get_env_conf, 
    get_torch_dtype,
    get_optimizer_and_lr_adjuster, 
    PolicyGradient,
    History)

import argparse, random, numpy, os


def zero_grad(params):
    for param in params:
        param.grad = None


def build_dataset(env_conf, tokenizer):
    sum_partition = 0

    num_iters = env_conf['train']['train_iters']
    corpus = []
    for info in env_conf['train']['corpus']:
        sum_partition += info['partition']
        num_instance = int(info['partition'] * num_iters)

        proc = get_processor(info['conf'], tokenizer)
        corp = RandomSampleCorpus(info['data'], proc, max_instance=num_instance, use_cache=True)
        corpus.append(corp)

    assert sum_partition == 1
    return ConcatDataset(corpus)


def collate_fn(batch):
    input_ids = batch[0]['input_ids']
    labels = input_ids[1:] + [-100]

    input_ids = torch.tensor(input_ids, dtype=torch.int64, device='cuda')
    labels = torch.tensor(labels, dtype=torch.int64, device='cuda')

    input_ids = input_ids.unsqueeze(0)
    labels = labels.unsqueeze(0)

    return dict(
        input_ids=input_ids,
        labels=labels)


def seed_everything(seed):
    torch.random.manual_seed(seed)
    random.seed(seed)
    numpy.random.seed(seed)


def backend_setup():
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    dist.init_process_group("nccl", rank=local_rank, world_size=world_size)
    torch.cuda.set_device(local_rank)


def backend_cleanup():
    dist.destroy_process_group()


def collect_grad(params):
    grads = []
    for param in params:
        grads.append(param.grad.data.ravel())
    return torch.cat(grads)


def sample_mask(outputs, params, is_last_sample):
    zero_grad(params)

    _score = outputs['score'].detach()
    _score.requires_grad_(True)

    s = _score.ravel()

    mask = s.sigmoid() > torch.rand_like(s)
    index = mask.type(torch.int64).unsqueeze(-1)
    mask = mask.tolist()

    # compute log likelihood and backward
    pos_nll = torch.nn.functional.logsigmoid(s)
    neg_nll = s + pos_nll
    nll = torch.stack([neg_nll, pos_nll], dim=-1)
    torch.gather(nll, dim=-1, index=index).mean().backward()

    # collect gradients
    retain_graph = not is_last_sample
    outputs['score'].backward(gradient=_score.grad.data, retain_graph=retain_graph)
    grad = collect_grad(params)

    return mask, grad
    

if __name__ == '__main__':


    backend_setup()


    parser = argparse.ArgumentParser()
    parser.add_argument("--env_conf", type=str, required=True)
    parser.add_argument("--n_sample", type=int, default=128)
    parser.add_argument("--accum_grad", type=int, default=1)
    args = parser.parse_args()

    
    env_conf = get_env_conf(args.env_conf)
    env_conf['model']['device_map'] = {"": dist.get_rank()}
    dtype = get_torch_dtype(env_conf['model']['model_dtype'])


    # load model
    seed_everything(0)
    model, tokenizer = get_model_and_tokenizer(**env_conf['model'])
    seed_everything(dist.get_rank())


    model.train()


    # constraits
    assert args.n_sample % dist.get_world_size() == 0, f"argument `--n_sample` must be divisible by the number of GPUs"



    params = model.ft_params()
    optimizer, lr_adjuster = get_optimizer_and_lr_adjuster(**env_conf['train'], params=params)


    # build dataset
    """
    NOTE: The purpose of prioritizing rank-0's dataset loading is to let other processes
    use the cache from rank-0, ensuring dataset consistency.
    """
    if dist.get_rank() == 0:
        corpus = build_dataset(env_conf, tokenizer)
    dist.barrier()
    if dist.get_rank() != 0:
        corpus = build_dataset(env_conf, tokenizer)

    loader = DataLoader(
        corpus, 
        batch_size=1, 
        collate_fn=collate_fn)
    

    pg = PolicyGradient(params)
    history = History()


    for step, batch in enumerate(loader):


        lr_adjuster(step=step)

        outputs = model(**batch)


        n_sample_per_gpu = args.n_sample // dist.get_world_size()
        
        for i in range(n_sample_per_gpu):

            is_last_sample = i == n_sample_per_gpu - 1
            mask, grad = sample_mask(outputs, params, is_last_sample) 

            sample_outputs = model(**batch, mask=mask)

            pg.update(sample_outputs['reward'], grad)
            history.update(sample_outputs)

        pg.step()
        history.step(outputs)

        if (step + 1) % args.accum_grad == 0:
            pg.prepare()
            optimizer.step()


    history.summary()


    if dist.get_rank() == 0:
        model.save_checkpoint()

    backend_cleanup()
