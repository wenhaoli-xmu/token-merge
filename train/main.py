from torch.utils.data import ConcatDataset, DataLoader
import torch.distributed as dist

import torch
import numpy as np
import json


from corpus import get_processor, RandomSampleCorpus
from myabc.misc import get_model_and_tokenizer, get_env_conf, get_torch_dtype, get_optimizer_and_lr_adjuster

import argparse, random, numpy, os
from itertools import chain


def filter_valid(x_list):
    return list(filter(lambda x: x is not None, x_list))


def flatten_then_concat(x_list):
    x_list = [x.flatten() for x in x_list]
    return torch.cat(x_list)


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
    input_ids = [x.get('input_ids') for x in batch]
    input_ids = torch.tensor(input_ids, dtype=torch.int64)
    return {"input_ids": input_ids, "labels": input_ids.clone()}


def seed_everything(seed):
    torch.random.manual_seed(seed)
    random.seed(seed)
    numpy.random.seed(seed)


def backend_setup():
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    dist.init_process_group("nccl", rank=local_rank, world_size=world_size)


def backend_cleanup():
    dist.destroy_process_group()


def construct_cumulative_masks(masks):
    cumulative_mask = None

    for i in range(len(masks)):
        if masks[i] is not None:
            cumulative_mask = torch.masked_scatter(cumulative_mask, cumulative_mask == False, source=masks[i]) if cumulative_mask is not None else masks[i]
            masks[i] = cumulative_mask.clone()

    return masks


def first_forward_pass(args, model, batch, current_layer):
    with torch.no_grad():
        model.enable_grad(current_layer)
        loss_baseline, logits, _ = model(
            input_ids=batch['input_ids'],
            labels=batch['labels'],
            return_logits=[current_layer],
            sample_pruning=[],
            deterministic_pruning=list(range(current_layer)),
            trainable_tokens=args.trainable_tokens,
            merge_method=args.merge_method)

        logits = logits[current_layer]

        # NOTE: trainable tokens should not be merged
        if args.trainable_tokens is not None:
            logits[..., -args.trainable_tokens:] = torch.finfo(logits.dtype).min

        # NOTE: the probability calculated here is used for sampling, thus doesn't requires gradient
        pos_probs = logits.sigmoid()

        assert pos_probs.requires_grad is False, f"expect `prob_pos` not requiring grad with the `no_grad()` context."

        neg_probs = 1 - pos_probs
        probs = torch.stack([neg_probs, pos_probs], dim=-1)

    # NOTE: the negative log probability calculated here is used for gradient calculation
    assert logits.requires_grad, f"expect all the bottom `layer_idx + 1` layers requires grads."
    assert loss_baseline.requires_grad is False, f"expect final output loss not requiring grad."

    return loss_baseline, logits, probs


def compute_surprise(logits):
    # get the negative logorithm probability, used in the later backward propagation
    surprise = torch.nn.functional.softplus(-logits)
    assert torch.allclose(surprise, -torch.nn.functional.logsigmoid(logits)), f"numerical problems occured when computing log probability."
    surprise = torch.stack([logits + surprise, surprise], dim=-1)
    return surprise


def do_some_assertions(loss_baseline, probs):

    # Assertion 1, ensuring all sub-processes are tackling the same data
    global_loss_baseline = [torch.empty_like(loss_baseline) for _ in range(dist.get_world_size())]
    dist.all_gather(global_loss_baseline, loss_baseline)
    assert all([other_loss_baseline == loss_baseline for other_loss_baseline in global_loss_baseline]), f"expect `loss_baseline` from all processes close, but not."

    # Assertion 2, ensuring identical results of probabilities from all sub-process
    global_probs = [torch.empty_like(probs) for _ in range(dist.get_world_size())]
    dist.all_gather(global_probs, probs)
    assert all([torch.allclose(other_probs, probs) for other_probs in global_probs]), f"expect `prob` from all processes close, but not"


def sample_once(args, model, batch, current_layer):

    with torch.no_grad():
        model.disable_grad()
        loss, _, masks = model(
            input_ids=batch['input_ids'],
            labels=batch['labels'],
            return_logits=[],
            sample_pruning=[current_layer],
            deterministic_pruning=list(range(current_layer)),
            trainable_tokens=args.trainable_tokens,
            merge_method=args.merge_method)
        
        num_pruned = sum([m.int().sum().item() for m in filter(lambda x: x is not None , masks)])
        total = batch['input_ids'].shape[-1] - 1 - args.trainable_tokens if args.trainable_tokens is not None else 0
        ratio = num_pruned / total

    return loss, masks, ratio



def compute_gradient(surprises, masks, params, current_layer):

    index = masks[current_layer].to(torch.int64).unsqueeze(-1)

    detached_surprises = surprises.detach()
    detached_surprises.requires_grad_(True)
    torch.gather(detached_surprises, dim=-1, index=index).mean().backward()

    surprises.backward(gradient=detached_surprises.grad.data, retain_graph=True)

    # and then store this gradient for future use
    grad = []
    for param in params:
        grad.append(param.grad.data.clone().ravel())
        param.grad.data.zero_()
    
    return torch.cat(grad)


def gather_losses_and_ratios_and_grads(local_losses, local_ratios, local_grads):
    # gather losses & ratios & gradients
    local_losses = torch.tensor(local_losses, device=dist.get_rank())
    global_losses = [torch.empty_like(local_losses) for _ in range(dist.get_world_size())]
    dist.all_gather(global_losses, local_losses)
    losses = list(chain.from_iterable([loss.tolist() for loss in global_losses]))


    local_ratios = torch.tensor(local_ratios, device=dist.get_rank())
    global_ratios = [torch.empty_like(local_ratios) for _ in range(dist.get_world_size())]
    dist.all_gather(global_ratios, local_ratios)
    ratios = list(chain.from_iterable([ratio.tolist() for ratio in global_ratios]))


    local_grads = torch.stack(local_grads ,dim=0)
    global_grads = [torch.empty_like(local_grads) for _ in range(dist.get_world_size())]
    dist.all_gather(global_grads, local_grads)
    grads = torch.cat(global_grads, dim=0)

    return losses, ratios, grads


def compute_reward(args, loss_baseline, losses, ratios, grads):
    # calculate reward
    rewards = [-abs(loss - loss_baseline.item()) + args.gamma * ratio for loss, ratio in zip(losses, ratios)]
    rewards_mean = np.mean(rewards)
    rewards_std = np.std(rewards)
    rewards_norm = [(reward - rewards_mean) / (rewards_std + args.eps) for reward in rewards]
    rewards_norm = torch.tensor(rewards_norm, dtype=dtype, device=grads.device).unsqueeze(-1)

    return rewards, rewards_norm


def copy_gradients(params, grads):
    # copy gradients
    start, end = 0, 0
    for param in params:
        end = param.numel() + start
        param.grad.data.copy_(grads[start:end].reshape_as(param.grad.data))
        start = end
    

if __name__ == '__main__':


    backend_setup()


    parser = argparse.ArgumentParser()
    parser.add_argument("--env_conf", type=str, required=True)
    parser.add_argument("--n_samples_per_gpu", type=int, default=32)
    parser.add_argument("--trainable_tokens", type=int, default=None)
    parser.add_argument("--trainable_layers", type=str, default="[0,1,2,3]")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--merge_method", type=str, default='avg')
    args = parser.parse_args()

    
    args.trainable_layers = json.loads(args.trainable_layers)
    env_conf = get_env_conf(args.env_conf)
    env_conf['model']['device_map'] = {"": dist.get_rank()}
    dtype = get_torch_dtype(env_conf['model']['model_dtype'])


    # load model
    seed_everything(0)
    model, tokenizer = get_model_and_tokenizer(**env_conf['model'])
    seed_everything(dist.get_rank())


    # fetch parameters
    model.train()


    # constraits
    assert args.n_samples_per_gpu % dist.get_world_size() == 0, f"argument `--n_samples` must be divisible by the number of GPUs"


    # train loop
    for current_layer in args.trainable_layers:

        params = model.layer_ft_params(current_layer)
        optimizer, lr_adjuster = get_optimizer_and_lr_adjuster(**env_conf['train'], params=params)


        # build dataset
        if dist.get_rank() == 0:
            corpus = build_dataset(env_conf, tokenizer)
        dist.barrier()
        if dist.get_rank() != 0:
            corpus = build_dataset(env_conf, tokenizer)
        loader = DataLoader(
            corpus, 
            batch_size=1, 
            collate_fn=collate_fn)
        

        rewards_record = []
        losses_record = []
        ratios_record = []



        for step, batch in enumerate(loader):


            lr_adjuster(step=step)
            optimizer.zero_grad()


            loss_baseline, logits, probs = first_forward_pass(args, model, batch, current_layer)
            


            surprises = compute_surprise(logits)



            try:
                do_some_assertions(loss_baseline, probs)
            except:
                if dist.get_rank() == 0:
                    import IPython
                    IPython.embed()
            dist.barrier()



            local_grads = []
            local_losses = []
            local_ratios = []


            for _ in range(args.n_samples_per_gpu):


                loss, masks, ratio = sample_once(args, model, batch, current_layer)
                local_losses.append(loss.item())
                local_ratios.append(ratio)


                grad = compute_gradient(surprises, masks, params, current_layer)
                local_grads.append(grad)



            # Release all computational graph, deleting attached variables is enough to do so
            del logits, surprises


            losses, ratios, grads = gather_losses_and_ratios_and_grads(local_losses, local_ratios, local_grads)
            losses_record.append(-np.mean(losses))
            ratios_record.append(np.mean(ratios) * args.gamma)


            rewards, weights = compute_reward(args, loss_baseline, losses, ratios, grads)
            rewards_record.append(np.mean(rewards))



            if step % 100 == 0 and dist.get_rank() == 0:
                print(
                    f"step-{step:<5d} | "
                    f"loss_baseline: {loss_baseline.item():>.3f} | "
                    f"loss: {np.mean(losses):>.3f} | "
                    f"ratio: {np.mean(ratios):>.3f}", 
                    flush=True)


            weighted_grads = weights * grads
            grads = weighted_grads.sum(0)


            copy_gradients(params, grads)


            # optim step
            optimizer.step()


        if dist.get_rank() == 0:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(rewards_record)
            plt.plot(losses_record)
            plt.plot(ratios_record)
            plt.legend(['rewards', '-losses', 'gamma * ratios'])
            plt.savefig(f"train-{current_layer}.jpg", dpi=640)


    if dist.get_rank() == 0:
        model.save_checkpoint()


    backend_cleanup()
