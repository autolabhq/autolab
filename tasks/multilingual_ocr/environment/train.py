"""Fine-tune DeepSeek-OCR 3B for Persian OCR."""

import torch, os, sys, math, io, logging
os.environ["UNSLOTH_DISABLE_WEIGHT_CHECK"] = "1"
sys.path.insert(0, "/models")

_orig_warning = logging.Logger.warning
def _patched_warning(self, msg, *args, **kwargs):
    if "not initialized from" in str(msg) and "position_ids" in str(msg):
        return
    return _orig_warning(self, msg, *args, **kwargs)
logging.Logger.warning = _patched_warning

from PIL import Image, ImageOps
from dataclasses import dataclass
from typing import List, Dict, Any
from torch.nn.utils.rnn import pad_sequence
from unsloth import FastVisionModel
from transformers import AutoModel, Trainer, TrainingArguments
from datasets import load_from_disk
from deepseek_ocr.modeling_deepseekocr import BasicImageTransform, text_encode, dynamic_preprocess

# ── Model ──────────────────────────────────────────────────────
model, tokenizer = FastVisionModel.from_pretrained(
    model_name="/models/DeepSeek-OCR",
    max_seq_length=1024,
    load_in_4bit=False,
    auto_model=AutoModel,
    trust_remote_code=True,
    unsloth_force_compile=True,
)

model = FastVisionModel.get_peft_model(
    model,
    r=16, lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0, bias="none",
    use_gradient_checkpointing="unsloth", random_state=3407,
)

# ── Dataset ────────────────────────────────────────────────────
INSTRUCTION = "<image>\nFree OCR. "

def convert_to_conversation(sample):
    return {
        "messages": [
            {"role": "<|User|>", "content": INSTRUCTION,
             "images": [Image.open(sample["image_path"]).convert("RGB")]},
            {"role": "<|Assistant|>", "content": sample["text"]},
        ]
    }

raw_dataset = load_from_disk("/data/ocr_train")
dataset = [convert_to_conversation(s) for s in raw_dataset]

# ── DataCollator ───────────────────────────────────────────────
@dataclass
class DeepSeekOCRDataCollator:
    tokenizer: Any
    model: Any
    image_size: int = 640
    base_size: int = 1024
    crop_mode: bool = True
    train_on_responses_only: bool = True
    image_token_id: int = 128815

    def __init__(self, tokenizer, model, image_size=640, base_size=1024,
                 crop_mode=True, train_on_responses_only=True):
        self.tokenizer = tokenizer
        self.model = model
        self.image_size = image_size
        self.base_size = base_size
        self.crop_mode = crop_mode
        self.train_on_responses_only = train_on_responses_only
        self.image_token_id = 128815
        self.dtype = model.dtype
        self.image_transform = BasicImageTransform(
            mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), normalize=True
        )
        self.patch_size = 16
        self.downsample_ratio = 4
        self.bos_id = (tokenizer.bos_token_id
                       if hasattr(tokenizer, "bos_token_id") and tokenizer.bos_token_id is not None
                       else 0)

    def process_image(self, image):
        images_list, images_crop_list, images_spatial_crop = [], [], []

        if self.crop_mode:
            if image.size[0] <= 640 and image.size[1] <= 640:
                crop_ratio = (1, 1)
                images_crop_raw = []
            else:
                images_crop_raw, crop_ratio = dynamic_preprocess(
                    image, min_num=2, max_num=9,
                    image_size=self.image_size, use_thumbnail=False
                )
            global_view = ImageOps.pad(
                image, (self.base_size, self.base_size),
                color=tuple(int(x * 255) for x in self.image_transform.mean)
            )
            images_list.append(self.image_transform(global_view).to(self.dtype))
            width_crop_num, height_crop_num = crop_ratio
            images_spatial_crop.append([width_crop_num, height_crop_num])
            if width_crop_num > 1 or height_crop_num > 1:
                for crop_img in images_crop_raw:
                    images_crop_list.append(self.image_transform(crop_img).to(self.dtype))
            num_q = math.ceil((self.image_size // self.patch_size) / self.downsample_ratio)
            num_q_base = math.ceil((self.base_size // self.patch_size) / self.downsample_ratio)
            tok_img = ([self.image_token_id] * num_q_base + [self.image_token_id]) * num_q_base
            tok_img += [self.image_token_id]
            if width_crop_num > 1 or height_crop_num > 1:
                tok_img += ([self.image_token_id] * (num_q * width_crop_num) + [self.image_token_id]) * (num_q * height_crop_num)
        else:
            crop_ratio = (1, 1)
            images_spatial_crop.append([1, 1])
            if self.base_size <= 640:
                resized = image.resize((self.base_size, self.base_size), Image.LANCZOS)
                images_list.append(self.image_transform(resized).to(self.dtype))
            else:
                global_view = ImageOps.pad(
                    image, (self.base_size, self.base_size),
                    color=tuple(int(x * 255) for x in self.image_transform.mean)
                )
                images_list.append(self.image_transform(global_view).to(self.dtype))
            num_q = math.ceil((self.base_size // self.patch_size) / self.downsample_ratio)
            tok_img = ([self.image_token_id] * num_q + [self.image_token_id]) * num_q
            tok_img += [self.image_token_id]

        return images_list, images_crop_list, images_spatial_crop, tok_img, crop_ratio

    def process_single_sample(self, messages):
        images = []
        for msg in messages:
            for img in msg.get("images", []):
                if img is None:
                    continue
                if isinstance(img, Image.Image):
                    images.append(img.convert("RGB"))
                elif isinstance(img, dict) and "bytes" in img:
                    images.append(Image.open(io.BytesIO(img["bytes"])).convert("RGB"))

        tokenized_str = [self.bos_id]
        images_seq_mask = [False]
        images_list, images_crop_list, images_spatial_crop = [], [], []
        prompt_token_count = -1
        assistant_started = False
        image_idx = 0

        for msg in messages:
            role, content = msg["role"], msg["content"]
            if role == "<|Assistant|>":
                if not assistant_started:
                    prompt_token_count = len(tokenized_str)
                    assistant_started = True
                content = f"{content.strip()} {self.tokenizer.eos_token}"

            parts = content.split("<image>")
            for i, part in enumerate(parts):
                toks = text_encode(self.tokenizer, part, bos=False, eos=False)
                tokenized_str.extend(toks)
                images_seq_mask.extend([False] * len(toks))
                if i < len(parts) - 1:
                    img_list, crop_list, spatial, tok_img, _ = self.process_image(images[image_idx])
                    images_list.extend(img_list)
                    images_crop_list.extend(crop_list)
                    images_spatial_crop.extend(spatial)
                    tokenized_str.extend(tok_img)
                    images_seq_mask.extend([True] * len(tok_img))
                    image_idx += 1

        if not assistant_started:
            prompt_token_count = len(tokenized_str)

        images_ori = torch.stack(images_list, dim=0)
        images_spatial_crop_t = torch.tensor(images_spatial_crop, dtype=torch.long)
        images_crop = (torch.stack(images_crop_list, dim=0) if images_crop_list
                       else torch.zeros((1, 3, self.base_size, self.base_size), dtype=self.dtype))

        return {
            "input_ids": torch.tensor(tokenized_str, dtype=torch.long),
            "images_seq_mask": torch.tensor(images_seq_mask, dtype=torch.bool),
            "images_ori": images_ori,
            "images_crop": images_crop,
            "images_spatial_crop": images_spatial_crop_t,
            "prompt_token_count": prompt_token_count,
        }

    def __call__(self, features: List[Dict[str, Any]]) -> Dict:
        batch_data = []
        for feat in features:
            try:
                batch_data.append(self.process_single_sample(feat["messages"]))
            except Exception as e:
                print(f"Error processing sample: {e}")
        if not batch_data:
            raise ValueError("No valid samples in batch")

        input_ids = pad_sequence(
            [d["input_ids"] for d in batch_data],
            batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        images_seq_mask = pad_sequence(
            [d["images_seq_mask"] for d in batch_data],
            batch_first=True, padding_value=False
        )
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        labels[images_seq_mask] = -100

        if self.train_on_responses_only:
            for idx, d in enumerate(batch_data):
                if d["prompt_token_count"] > 0:
                    labels[idx, :d["prompt_token_count"]] = -100

        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        images_batch = [(d["images_crop"], d["images_ori"]) for d in batch_data]
        images_spatial_crop = torch.cat([d["images_spatial_crop"] for d in batch_data], dim=0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "images": images_batch,
            "images_seq_mask": images_seq_mask,
            "images_spatial_crop": images_spatial_crop,
        }

# ── Trainer ────────────────────────────────────────────────────
FastVisionModel.for_training(model)

data_collator = DeepSeekOCRDataCollator(
    tokenizer=tokenizer,
    model=model,
    image_size=640,
    base_size=1024,
    crop_mode=True,
    train_on_responses_only=True,
)

training_args = TrainingArguments(
    output_dir="/app/output",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=5,
    max_steps=60,
    learning_rate=2e-4,
    logging_steps=5,
    optim="adamw_8bit",
    weight_decay=0.001,
    lr_scheduler_type="linear",
    seed=3407,
    report_to="none",
    remove_unused_columns=False,
    bf16=True,
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    data_collator=data_collator,
    args=training_args,
)

trainer.train()
model.save_pretrained("/app/output")
tokenizer.save_pretrained("/app/output")
print("Training complete. Adapter saved to /app/output/")
