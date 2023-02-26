import os
import sys
import tempfile
import requests
from typing import Iterator, Optional
from dotenv import load_dotenv

load_dotenv()

os.environ["TORCH_HOME"] = "/src/.torch"
os.environ["TRANSFORMERS_CACHE"] = "/src/.huggingface/"
os.environ["DIFFUSERS_CACHE"] = "/src/.huggingface/"
os.environ["HF_HOME"] = "/src/.huggingface/"

sys.path.append("./lora_diffusion")

# from settings import (
#     LoraMaskingSettings, 
#     LoraTrainingSettings
# )
#from lora import train_lora

from preprocess_files import load_and_save_masks_and_captions
from cli_lora_pti import train

from cog import BasePredictor, BaseModel, File, Input, Path


checkpoint_options = [
    "runwayml/stable-diffusion-v1-5",
    "prompthero/openjourney-v2",
    "dreamlike-art/dreamlike-photoreal-2.0"
]


class CogOutput(BaseModel):
    file: Path
    name: Optional[str] = None
    thumbnail: Optional[Path] = None
    attributes: Optional[dict] = None
    progress: Optional[float] = None
    isFinal: bool = False


def download(url, folder, ext):
    filename = url.split('/')[-1]+ext
    filepath = folder / filename
    os.makedirs(folder, exist_ok=True)
    if filepath.exists():
        return filepath
    raw_file = requests.get(url, stream=True).raw
    with open(filepath, 'wb') as f:
        f.write(raw_file.read())
    return filepath


class Predictor(BasePredictor):

    def setup(self):
        print("cog:setup")

    def predict(
        self,
        
        checkpoint: str = Input(
            description="Which Stable Diffusion checkpoint to use",
            choices=checkpoint_options,
            default="dreamlike-art/dreamlike-photoreal-2.0"
        ),
        lora_training_urls: str = Input(
            description="Training images for new LORA concept", 
            default=None
        ),
        name: str = Input(
            description="Name of new LORA concept",
            default=None
        ),
        train_text_encoder: bool = Input(
            description="Train text encoder",
            default=True
        ),
        perform_inversion: bool = Input(
            description="Perform inversion",
            default=True
        ),
        resolution: int = Input(
            description="Resolution",
            default=512
        ),
        train_batch_size: int = Input(
            description="Batch size",
            default=4
        ),
        gradient_accumulation_steps: int = Input(
            description="Gradient accumulation steps",
            default=1
        ),
        scale_lr: bool = Input(
            description="Scale learning rate",
            default=True
        ),
        learning_rate_ti: float = Input(
            description="Learning rate for textual inversion",
            default=2.5e-4
        ),
        continue_inversion: bool = Input(
            description="Continue inversion",
            default=True
        ),
        continue_inversion_lr: float = Input(
            description="Continue inversion learning rate",
            default=2.5e-5
        ),
        learning_rate_unet: float = Input(
            description="Learning rate for U-Net",
            default=1.5e-5
        ),
        learning_rate_text: float = Input(
            description="Learning rate for text encoder",
            default=2.5e-5
        ),
        color_jitter: bool = Input(
            description="Color jitter",
            default=True
        ),
        lr_scheduler: str = Input(
            description="Learning rate scheduler",
            default="linear"
        ),
        lr_warmup_steps: int = Input(
            description="Learning rate warmup steps",
            default=0
        ),
        placeholder_tokens: str = Input(
            description="Placeholder token for concept",
            default="<person1>"
        ),
        use_template: str = Input(
            description="Use template",
            default="person",
            choices=["person", "object", "style"]
        ),
        use_mask_captioned_data: bool = Input(
            description="Use mask captioned data",
            default=False
        ),
        max_train_steps_ti: int = Input(
            description="Max train steps for textual inversion",
            default=300
        ),
        max_train_steps_tuning: int = Input(
            description="Max train steps for tuning (U-Net and text encoder)",
            default=500
        ),
        clip_ti_decay: bool = Input(
            description="CLIP textual inversion decay",
            default=True
        ),
        weight_decay_ti: float = Input(
            description="Weight decay for textual inversion",
            default=0.0005
        ),
        weight_decay_lora: float = Input(
            description="Weight decay for LORA",
            default=0.001
        ),
        lora_rank_unet: int = Input(
            description="LORA rank for U-Net",
            default=2
        ),
        lora_rank_text_encoder: int = Input(
            description="LORA rank for text encoder",
            default=8
        ),
        use_extended_lora: bool = Input(
            description="Use extended LORA",
            default=False
        ),        
        use_face_segmentation_condition: bool = Input(
            description="Use face segmentation condition",
            default=True
        ),

    ) -> Iterator[CogOutput]:

        print("cog:predict:")

        data_dir = Path(tempfile.mkdtemp())
        out_dir = Path(tempfile.mkdtemp())

        print("train lora", str(data_dir), str(out_dir))

        data_dir.mkdir(exist_ok=True)
        out_dir.mkdir(exist_ok=True)
        
        lora_training_urls = lora_training_urls.split('|')
        for lora_url in lora_training_urls:
            print("download", lora_url)
            download(lora_url, data_dir, '.jpg')

        load_and_save_masks_and_captions(
            files = str(data_dir),
            output_dir = str(data_dir) + "/train",
            caption_text = None,
            target_prompts = "face",
            target_size = 512,
            crop_based_on_salience = True,
            use_face_detection_instead = False,
            temp = 1.0,
            n_length = -1
        )

        train(
            instance_data_dir = str(data_dir) + "/train",
            output_dir = str(out_dir),
            pretrained_model_name_or_path = checkpoint,
            out_name = name,            
            train_text_encoder = train_text_encoder,
            perform_inversion = perform_inversion,
            resolution = resolution,
            train_batch_size = train_batch_size,
            gradient_accumulation_steps = gradient_accumulation_steps,
            scale_lr = scale_lr,
            learning_rate_ti = learning_rate_ti,
            continue_inversion = continue_inversion,
            continue_inversion_lr = continue_inversion_lr,
            learning_rate_unet = learning_rate_unet,
            learning_rate_text = learning_rate_text,
            color_jitter = color_jitter,
            lr_scheduler = lr_scheduler,
            lr_warmup_steps = lr_warmup_steps,
            placeholder_tokens = placeholder_tokens,
            proxy_token = "person",
            use_template = use_template,
            use_mask_captioned_data = use_mask_captioned_data,
            save_steps = 10000,
            max_train_steps_ti = max_train_steps_ti,
            max_train_steps_tuning = max_train_steps_tuning,
            clip_ti_decay = clip_ti_decay,
            weight_decay_ti = weight_decay_ti,
            weight_decay_lora = weight_decay_lora,
            lora_rank_unet = lora_rank_unet,
            lora_rank_text_encoder = lora_rank_text_encoder,
            cached_latents = False,
            use_extended_lora = use_extended_lora,
            enable_xformers_memory_efficient_attention = True,
            use_face_segmentation_condition = True
        )

        lora_location = os.path.join(str(out_dir), f'{name}.safetensors')
    
        yield CogOutput(file=Path(lora_location), name=name, thumbnail=None, attributes=None, isFinal=True, progress=1.0)
