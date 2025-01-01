import torch
import types
import torch.distributed
from transformers.models.llama.modeling_llama import repeat_kv
from ..modifier import Modifier
from .utils import ScoreHead, check_and_apply_qk_rope, do_projection, prune_labels, merge


def model_forward(self, input_ids, mask):
    hidden_states, score = self.model(input_ids, mask)
    logits = self.lm_head(hidden_states).float()
    return logits, score



def model_model_forward(self, input_ids, mask):

    inputs_embeds = self.embed_tokens(input_ids)
    hidden_states = inputs_embeds

    score = None

    for layer in self.layers:

        """
        1. if mask is not None, then do pruning
        2. if mask is None, then return score
        """
        if hasattr(layer, 'score_head'):
            if mask is not None:
                hidden_states = merge(hidden_states, mask, self.merge_method)
            else:
                with torch.enable_grad():
                    score = layer.score_head(hidden_states)


        hidden_states = layer(hidden_states)
        
    hidden_states = self.norm(hidden_states)

    return hidden_states, score



def layer_forward(self, hidden_states):    

    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states = self.self_attn(hidden_states)
    hidden_states = residual + hidden_states
    
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    return hidden_states


def self_attn_forward(self, hidden_states):

    num_heads, embed_dim = self.config.num_attention_heads, self.config.hidden_size
    num_kv_heads = self.config.num_key_value_heads
    num_kv_group = num_heads // num_kv_heads
    head_dim = embed_dim // num_heads


    # query & key & value projection
    ques = do_projection(self.q_proj, hidden_states, num_heads, head_dim)
    keys = do_projection(self.k_proj, hidden_states, num_kv_heads, head_dim)
    vals = do_projection(self.v_proj, hidden_states, num_kv_heads, head_dim)

    # position embedding
    len1 = self.config.max_position_embeddings if hasattr(self.config, "max_position_embeddings") else 0
    len2 = max(ques.shape[-2], keys.shape[-2])
    cos, sin = self.rotary_emb(keys, seq_len=max(len1, len2))
    ques, keys = check_and_apply_qk_rope(ques, keys, cos, sin)


    keys = repeat_kv(keys, num_kv_group)
    vals = repeat_kv(vals, num_kv_group)

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query=ques,
        key=keys,
        value=vals,
        is_causal=True)

    attn_output = attn_output.transpose(1,2).flatten(2)
    attn_output = self.o_proj(attn_output)

    return attn_output


class ModelForTraining(Modifier):
    def __init__(self, model, save_ckp: str, load_ckp: str, config: str):
        self.get_conf(config)
        model.forward = types.MethodType(model_forward, model)
        model.model.forward = types.MethodType(model_model_forward, model.model)
        model.model.merge_method = self.conf['merge_method']

        for i, layer in enumerate(model.model.layers):
            layer.forward = types.MethodType(layer_forward, layer)
            layer.self_attn.forward = types.MethodType(self_attn_forward, layer.self_attn)

            if i == self.conf['layer_cut']:
                layer.score_head = ScoreHead(hidden_size=model.config.hidden_size, kernel_size=2)

        super().__init__(model, save_ckp, load_ckp)


    def get_model(self):
        return self.model


    def ft_params(self):
        params = []
        for i, layer in enumerate(self.get_model().model.layers):
            if i == self.conf['layer_cut']:
                params += layer.score_head.parameters()
        return params
    

    @torch.no_grad()
    def forward(self, input_ids, labels, mask=None, outputs=None):

        """
        Return
        ------
        1. mask == None: return (loss, score, logits)
        2. mask != None: return (loss, ratio, reward)
        """

        ratio, reward = None, None
        last_n = self.conf['last_n']

        if mask is not None:
            mask[-last_n:] = [False] * last_n
            labels = prune_labels(labels, mask)
            ratio = sum(mask) / (len(mask) - last_n)

        # compute logits
        logits, score = self.model(input_ids=input_ids, mask=mask)
        
        # compute loss
        logits = logits.squeeze(0)[-last_n:]
        labels = labels.squeeze(0)[-last_n:]
        loss = torch.nn.functional.cross_entropy(logits, labels, reduce=False)

        # compute reward
        if mask is not None:
            if self.conf['loss_version'] == 'v1':
                reward = -loss.mean() + self.conf['ratio_weight'] * ratio
                
            elif self.conf['loss_version'] == 'v2':
                reward = -(loss.mean() - outputs['loss'].mean()).abs() + self.conf['ratio_weight'] * ratio

            elif self.conf['loss_version'] == 'v3':
                reward = -(loss.mean() - outputs['loss'].mean()).abs().exp() + self.conf['ratio_weight'] * ratio

            elif self.conf['loss_version'] == 'v4':
                reward = -(loss - outputs['loss']).abs().mean() + self.conf['ratio_weight'] * ratio

            elif self.conf['loss_version'] == 'v5':
                logits = torch.nn.functional.log_softmax(logits, dim=-1)
                target = torch.nn.functional.softmax(outputs['logits'] / 0.9, dim=-1)
                reward = -torch.nn.functional.kl_div(logits, target) + self.conf['ratio_weight'] * ratio
            else:
                raise NotImplementedError(self.conf['loss_version'])

            logits = None
            

        """
        Return
        ------

        :loss: langauge modeling loss without batch dim reduction

        :logits: output logits, maintaining only the last n tokens

        :score: mask predictor output

        :ratio: pruning ratio

        :reward: reward used in policy gradeint
        """

        return dict(
            loss=loss,
            logits=logits, 
            score=score, 
            ratio=ratio, 
            reward=reward)
