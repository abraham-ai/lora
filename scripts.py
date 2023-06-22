
export CUDA_VISIBLE_DEVICES=3
bash <(wget -qO- https://raw.githubusercontent.com/AUTOMATIC1111/stable-diffusion-webui/master/webui.sh) --xformers


ffmpeg -framerate 8 -i input_folder/*.jpg -vf "scale=640:-1" -loop 0 -r 24 -f gif - | gifsicle --optimize=3 --delay=3 > output.gif

ffmpeg -framerate 8 -i frames/*.jpg -vf "scale=640:-1" -loop 0 -r 8 -f gif - | gifsicle --optimize=3 --delay=3 > output.gif


export CUDA_VISIBLE_DEVICES=0
conda activate diffusers
cd /home/xander/Projects/cog/lora/lora_diffusion
python preprocess_files.py \
    --files="/home/xander/Pictures/Mars2023/people/saved_face_datasets/martians/karo/imgs" \
    --output_dir="/home/xander/Pictures/Mars2023/people/saved_face_datasets/martians/karo/train" \
    --target_prompts="face" \
    --crop_based_on_salience=True \
    --target_size=1024




source venv /bin/activate



STYLE:

export CUDA_VISIBLE_DEVICES=2
conda activate diffusers
cd /home/xander/Projects/cog/lora

export MODEL_NAME="dreamlike-art/dreamlike-photoreal-2.0"
export INSTANCE_DIR="/home/xander/Pictures/Mars2023/styles/Cooper/train_big"
export OUTPUT_DIR="./exps/cooper_abandoned_buildings"

python lora_diffusion/cli_lora_pti.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --train_text_encoder=True \
  --perform_inversion=False \
  --resolution=768 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=1 \
  --scale_lr \
  --learning_rate_ti=2e-4 \
  --continue_inversion \
  --continue_inversion_lr=2.0e-5 \
  --learning_rate_unet=1e-5 \
  --learning_rate_text=2.0e-5 \
  --color_jitter \
  --lr_scheduler="linear" \
  --lr_warmup_steps=0 \
  --use_mask_captioned_data=True \
  --save_steps=200 \
  --max_train_steps_ti=600 \
  --max_train_steps_tuning=5000 \
  --clip_ti_decay \
  --weight_decay_ti=0.0005 \
  --weight_decay_lora=0.001 \
  --lora_rank_unet=8 \
  --lora_rank_text_encoder=8 \
  --cached_latents=False \
  --use_extended_lora=True \
  --enable_xformers_memory_efficient_attention=True






BEST (FAST) PERSON SETTINGS SO FAR:


export CUDA_VISIBLE_DEVICES=2
conda activate diffusers
cd /home/xander/Projects/cog/lora

export MODEL_NAME="dreamlike-art/dreamlike-photoreal-2.0"
export INSTANCE_DIR="/home/xander/Pictures/Mars2023/people/train/dimi/train"
export OUTPUT_DIR="./exps/dimi_fin"

python lora_diffusion/cli_lora_pti.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --train_text_encoder=True \
  --perform_inversion=True \
  --resolution=512 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=1 \
  --scale_lr \
  --learning_rate_ti=5.0e-4 \
  --continue_inversion \
  --continue_inversion_lr=1.0e-5 \
  --learning_rate_unet=2.0e-5 \
  --learning_rate_text=3.0e-5 \
  --color_jitter \
  --lr_scheduler="linear" \
  --lr_warmup_steps=0 \
  --placeholder_tokens="<person1>" \
  --proxy_token="person" \
  --use_template="person"\
  --use_mask_captioned_data=False \
  --save_steps=100 \
  --max_train_steps_ti=400 \
  --max_train_steps_tuning=600 \
  --clip_ti_decay \
  --weight_decay_ti=0.0005 \
  --weight_decay_lora=0.001 \
  --lora_rank_unet=4 \
  --lora_rank_text_encoder=8 \
  --cached_latents=False \
  --use_extended_lora=False \
  --enable_xformers_memory_efficient_attention=True \
  --use_face_segmentation_condition=True



  param_grid = {
    'pretrained_model_name_or_path': ['dreamlike-art/dreamlike-photoreal-2.0'],
    'instance_data_dir':             "/home/xander/Pictures/Mars2023/people/ready/xander/train",

    'train_text_encoder':            True,
    'perform_inversion':             True,
    'learning_rate_ti':              [5e-4],
    'continue_inversion':            True,
    'continue_inversion_lr':         [1e-5],
    'learning_rate_unet':            [2e-5],
    'learning_rate_text':            [3e-5],
    'save_steps':                    100,
    'max_train_steps_ti':            [400], 
    'max_train_steps_tuning':        [600], 
    'weight_decay_ti':               [0.0005],
    'weight_decay_lora':             [0.0010],
    'lora_rank_unet':                [4],
    'lora_rank_text_encoder':        [8],
    'use_extended_lora':             [False],

    'use_face_segmentation_condition': True,
    'use_mask_captioned_data':       False,
    'placeholder_tokens':            ["\"<person1>\""],
    'proxy_token':                   "person",
    'use_template':                  "person",
    'initializer_tokens':            [None],
    'clip_ti_decay':                 True,
    'load_pretrained_inversion_embeddings_path': [None],
    'cached_latents':                False,
    'train_batch_size':              [4],
    'gradient_accumulation_steps':   1,
    'color_jitter':                  True,
    'scale_lr':                      True,
    'lr_scheduler':                  "linear",
    'lr_warmup_steps':               0,
    'resolution':                    512,
    'enable_xformers_memory_efficient_attention': True,

  }

















export CUDA_VISIBLE_DEVICES=2
conda activate diffusers
cd /home/xander/Projects/cog/lora

python lora_diffusion/cli_lora_add.py \
--path_1="dreamlike-art/dreamlike-photoreal-2.0" \
--path_2="/home/xander/Projects/cog/lora/exps/people_bs_6/vitalik_train_00_ff058b/final_lora.safetensors" \
--output_path="vincent_SDv1.5" \
--alpha_1=0.9 \
--mode=upl


export CUDA_VISIBLE_DEVICES=2
conda activate diffusers
cd /home/xander/Projects/cog/lora

python lora_diffusion/cli_lora_add.py \
--path_1="dreamlike-art/dreamlike-photoreal-2.0" \
--path_2="/home/xander/Projects/cog/lora/exps/1_LORAS/FIN/BASE/girls/active_prn/emma_big_train_00_65e1da/final_lora.safetensors" \
--output_path="/home/xander/stable-diffusion-webui/models/Stable-diffusion/emma.ckpt" \
--alpha_1=0.9 \
--mode=upl-ckpt-v2



export CUDA_VISIBLE_DEVICES=2
conda activate diffusers
cd /home/xander/Projects/cog/lora

python lora_diffusion/cli_lora_add.py \
--path_1="/home/xander/Projects/cog/lora/exps/1_LORAS/FIN/active_men/jmill_train_00_eebd4b/final_lora.safetensors" \
--path_2="/home/xander/Projects/cog/lora/exps/1_LORAS/FIN/active_woman/lucy_train_00_e0d08a/final_lora.safetensors" \
--output_path="/home/xander/Projects/cog/lora/exps/juicy.safetensors" \
--alpha_1=0.5 \
--alpha_2=0.5


instance_data_dir: str,
pretrained_model_name_or_path: str,

pretrained_vae_name_or_path: str = None,
class_data_dir: Optional[str] = None,

stochastic_attribute: Optional[str] = None,
perform_inversion: bool = True,
use_template: Literal[None, "object", "style"] = None,
train_inpainting: bool = False,


placeholder_tokens: str = "",
placeholder_token_at_data: Optional[str] = None,
initializer_tokens: Optional[str] = None,
class_prompt: Optional[str] = None,
with_prior_preservation: bool = False,
prior_loss_weight: float = 1.0,
num_class_images: int = 100,


seed: int = 42,
resolution: int = 512,
color_jitter: bool = True,
train_batch_size: int = 1,
sample_batch_size: int = 1,
max_train_steps_tuning: int = 1000,
max_train_steps_ti: int = 1000,
save_steps: int = 100,
gradient_accumulation_steps: int = 4,
gradient_checkpointing: bool = False,
mixed_precision="fp16",
lora_rank: int = 4,
lora_unet_target_modules={"CrossAttention", "Attention", "GEGLU"},
lora_clip_target_modules={"CLIPAttention"},
lora_dropout_p: float = 0.0,
lora_scale: float = 1.0,
use_extended_lora: bool = False,
clip_ti_decay: bool = True,
learning_rate_unet: float = 1e-4,
learning_rate_text: float = 1e-5,
learning_rate_ti: float = 5e-4,
continue_inversion: bool = False,
continue_inversion_lr: Optional[float] = None,
use_face_segmentation_condition: bool = False,
use_mask_captioned_data: bool = False,
mask_temperature: float = 1.0,
scale_lr: bool = False,
lr_scheduler: str = "linear",
lr_warmup_steps: int = 0,
lr_scheduler_lora: str = "linear",
lr_warmup_steps_lora: int = 0,
weight_decay_ti: float = 0.00,
weight_decay_lora: float = 0.001,
use_8bit_adam: bool = False,
device="cuda:0",
extra_args: Optional[dict] = None,
log_wandb: bool = False,
wandb_log_prompt_cnt: int = 10,
wandb_project_name: str = "new_pti_project",
wandb_entity: str = "new_pti_entity",
proxy_token: str = "person",
enable_xformers_memory_efficient_attention: bool = False,
out_name: str = "final_lora",








# Merge lora into checkpoint:


export CUDA_VISIBLE_DEVICES=0
conda activate diffusers
cd /home/xander/Projects/cog/lora

export MODEL_NAME="dreamlike-photoreal-2.0"
export LORA_FILE= "/home/xander/Projects/cog/lora/exps/1_LORAS/dimi/dimi_fin/final_lora.safetensors"
export OUTPUT_FILE="/home/xander/Projects/cog/lora/exps/merged_no_inversion"

python lora_diffusion/cli_lora_add.py --path_1=$MODEL_NAME --path_2=$LORA_FILE alpha_1=0.2 alpha_2=0.8 --output_path=$OUTPUT_FILE --mode='upl'



# convert to diffusers:
conda activate diffusers
cd /home/xander/Projects/cog/diffusers/scripts
python convert_original_stable_diffusion_to_diffusers.py --checkpoint_path="/home/xander/Projects/cog/eden-sd-pipelines/eden/models/dreamlike-photoreal-2.0.ckpt"  \
    --dump_path="/home/xander/Projects/cog/diffusers/eden/models/dreamlike-photoreal-new" \
    --from_safetensors

# convert to ckpt:

    parser.add_argument("--model_path", default=None, type=str, required=True, help="Path to the model to convert.")
    parser.add_argument("--checkpoint_path", default=None, type=str, required=True, help="Path to the output model.")
    parser.add_argument("--half", action="store_true", help="Save weights in half precision.")
    parser.add_argument(
        "--use_safetensors", action="store_true", help="Save weights use safetensors, default is ckpt."
    )

conda activate diffusers
cd /home/xander/Projects/cog/diffusers/scripts
python convert_diffusers_to_original_stable_diffusion.py \
    --model_path="/home/xander/Projects/cog/lora/exps/2_MODELS/emma_big_train_00_65e1da"  \
    --checkpoint_path="/home/xander/Projects/cog/lora/exps/2_MODELS/emma_big_train_00_65e1da.ckpt" \





-------------------------------------------------------------------------------------------------------------------------



# Merge lora into checkpoint:

export CUDA_VISIBLE_DEVICES=0
conda activate diffusers
cd /home/xander/Projects/cog/lora

export MODEL_NAME="dreamlike-art/dreamlike-photoreal-2.0"
export LORA_FILE= "/home/xander/Projects/cog/lora/exps/1_LORAS/dimi/dimi_fin/final_lora.safetensors"
export OUTPUT_FILE="/home/xander/Projects/cog/lora/exps/2_MODELS/dimi"

python lora_diffusion/cli_lora_add.py  \
        --path_1=$MODEL_NAME \
        --path_2=$LORA_FILE  \
        --alpha_1=0.5  \
        --alpha_2=0.5  \
        --output_path=$OUTPUT_FILE --mode='upl'


# convert to diffusers:

conda activate diffusers
cd /home/xander/Projects/cog/diffusers/scripts
python convert_original_stable_diffusion_to_diffusers.py --checkpoint_path="/home/xander/Projects/cog/eden-sd-pipelines/eden/models/rpg_V4.ckpt"  \
    --dump_path="/home/xander/Projects/cog/eden-sd-pipelines/eden/models/rpg_V4" \
    --extract_ema \
    --from_safetensors



# convert diffusers to WEBui ckpt
conda activate diffusers
cd /home/xander/Projects/cog/diffusers/scripts
python convert_diffusers_to_original_stable_diffusion.py \
    --model_path="/home/xander/Projects/cog/lora/exps/2_MODELS/dimi"  \
    --checkpoint_path="/home/xander/Projects/cog/lora/exps/2_MODELS/dimi.ckpt"







ln -s /home/xander/stable-diffusion-webui/models/Stable-diffusion /home/xander/Projects/cog/lora/exps/2_MODELS
ln -s /home/xander/Projects/cog/lora/exps/2_MODELS /home/xander/stable-diffusion-webui/models/Stable-diffusion


export CUDA_VISIBLE_DEVICES=0
conda activate diffusers
cd /home/xander/Projects/cog/diffusers
python scripts/convert_original_stable_diffusion_to_diffusers.py \
    --checkpoint_path="/home/xander/Projects/cog/eden-sd-pipelines/eden/models/offset_noise.ckpt.safetensors" \
    --image_size=512 \
    --prediction_type='epsilon'\
    --extract_ema \
    --from_safetensors \
    --dump_path="/home/xander/Projects/cog/eden-sd-pipelines/eden/models/offset_noise_v1.5"
