import os
import torch
import numpy as np

from transformers import CLIPTextModel, CLIPTokenizer

from lora_diffusion import (
    PivotalTuningDatasetCapation,
    extract_lora_ups_down,
    inject_trainable_lora,
    inject_trainable_lora_extended,
    inspect_lora,
    save_lora_weight,
    save_all,
    prepare_clip_model_sets,
    evaluate_pipe,
    UNET_EXTENDED_TARGET_REPLACE,
    parse_safeloras_embeds,
    apply_learned_embed_in_clip,
    load_safeloras_embeds
)

def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


def compute_pairwise_distances(x,y):
    # compute the L2 distance of each row in x to each row in y (both are torch tensors)
    # x is a torch tensor of shape (m, d)
    # y is a torch tensor of shape (n, d)
    # returns a torch tensor of shape (m, n)

    n = y.shape[0]
    m = x.shape[0]
    d = x.shape[1]

    x = x.unsqueeze(1).expand(m, n, d)
    y = y.unsqueeze(0).expand(m, n, d)

    return torch.pow(x - y, 2).sum(2)


def print_most_similar_tokens(tokenizer, optimized_token, text_encoder, n=10):
    with torch.no_grad():
        # get all the token embeddings:
        token_embeds = text_encoder.get_input_embeddings().weight.data

        # Compute the cosine-similarity between the optimized tokens and all the other tokens
        similarity = sim_matrix(optimized_token.unsqueeze(0), token_embeds).squeeze()
        similarity = similarity.detach().cpu().numpy()

        distances = compute_pairwise_distances(optimized_token.unsqueeze(0), token_embeds).squeeze()
        distances = distances.detach().cpu().numpy()
        
        # print similarity for the most similar tokens:
        most_similar_tokens = np.argsort(similarity)[::-1]

        print(f"{tokenizer.decode(most_similar_tokens[0])} --> mean: {optimized_token.mean().item():.3f}, std: {optimized_token.std().item():.3f}, norm: {optimized_token.norm():.4f}")
        for token_id in most_similar_tokens[1:n+1]:
            print(f"sim of {similarity[token_id]:.3f} & L2 of {distances[token_id]:.3f} with \"{tokenizer.decode(token_id)}\"")


lora_dir = "/home/xander/Projects/cog/lora/exps/compare_tokens"
pretrained_model_name_or_path = "dreamlike-art/dreamlike-photoreal-2.0"
patch_ti = True

print(f"Loading tokenizer and text_encoder from {pretrained_model_name_or_path}...")
tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=None,
    )
text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=None,
    )



for subdir in sorted(os.listdir(lora_dir)):
  subdirpath = os.path.join(lora_dir, subdir)
  lora_path = os.path.join(subdirpath, "final_lora.safetensors")
  try:
    tok_dict = load_safeloras_embeds(lora_path)
  except:
    continue

  tok_dict_new = {}
  for key in tok_dict.keys():
    tok_dict_new[key + '_' + subdir] = tok_dict[key]

  tok_dict = tok_dict_new

  print('\n-------------------', subdir)
  for key in tok_dict.keys():
    print_most_similar_tokens(tokenizer, tok_dict[key], text_encoder, n=4)

  if patch_ti:
      apply_learned_embed_in_clip(
          tok_dict,
          text_encoder,
          tokenizer,
          token=None,
          idempotent=0,
      )