
import os
import json
import torch
import numpy as np

root_dir = '/home/xander/Projects/cog/lora/exps/1_LORAS/dimi/dimi_fin'
save_as = 'tokens.pt'
save_path_safetenors = 'tokens.safetensors'
loras = 0

from lora_diffusion import parse_safeloras_embeds, save_all, save_safeloras_with_embeds
from safetensors.torch import safe_open

embeddings = []

def print_tensor_stats(t):
    print(f"norm: {t.norm()}, min: {t.min()} max: {t.max()} mean: {t.mean()} std: {t.std()}")

token_names = ['<object1>', '<person1>']
all_embeds = {}

counter = 0
# crawl this directory and grab all files called "final_lora.safetensors"
for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        if file == 'final_lora.safetensors':
            full_path = os.path.join(subdir, file)
            safeloras = safe_open(full_path, framework="pt", device="cpu")
            embeds = parse_safeloras_embeds(safeloras)
            print(embeds.keys())


            # save each one to .pt file:
            torch.save(embeds, f"dimi.pt")

            for key in embeds.keys():
                if key not in all_embeds:
                    all_embeds[key] = []
                embedding = embeds[key]
                #all_embeds[key] = embedding


            print('------------------------')
            print('Using token: ', token_names[counter])
            
            key = token_names[counter]
            all_embeds[key] = embedding

            print('------------------------')

            if len(embeds.keys()) == 1:
                embedding = list(embeds.values())[0]
                print_tensor_stats(embedding)
                embeddings.append(embedding)
                loras += 1

            counter += 1

print(f"Found {loras} loras")

# Average the embeddings:
embeddings = torch.stack(embeddings)

# Mean of the norms:
norms = torch.stack([e.norm() for e in embeddings])
mean_norm = torch.mean(norms)

avg_embedding = torch.mean(embeddings, dim=0)
# renormalize the average embedding:
avg_embedding = avg_embedding / avg_embedding.norm() * mean_norm

print_tensor_stats(avg_embedding)

# Save the average embedding:
torch.save(avg_embedding, save_as)


loras = {}
embeds = {}

for tok in all_embeds.keys():
    learned_embeds = all_embeds[tok]
    print(
        f"Current Learned Embeddings for {tok}:",
        learned_embeds[:4],
    )

    embeds[tok] = learned_embeds.detach().cpu()

save_safeloras_with_embeds(loras, embeds, save_path_safetenors)
print("Saved safeloras with embeds to %s" %save_path_safetenors)


