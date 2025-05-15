import logging
import open_clip
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from PIL import Image
from transformers import AutoConfig, AutoModel
from .criterions import VTC_VTM_Loss, Ranking_Loss, get_sim
from safetensors.torch import load_file

logger = logging.getLogger(__name__)


class H3DVideoScore(nn.Module):
    def __init__(self, config, tokenizer=None, is_pretrain=True):
        super().__init__()

        self.config = config
        self.tokenizer = tokenizer
        self.is_pretrain = is_pretrain

        # create modules.
        if tokenizer is None:
            self.tokenizer = open_clip.get_tokenizer(config.model.tokenizer)
        self.vision_encoder = self.build_vision_encoder()
        self.clip_encoder = self.build_clip_encoder()
        self.head = self.build_head()
        self.load_checkpoint()

        self.temp = nn.parameter.Parameter(torch.ones([]) * config.model.temp)
        self.temp_min = config.model.temp_min


        # freeze model
        if self.config.train_stage == 1 and not self.config.train_full:
            for name, p in self.head.named_parameters():
                logger.info(f"Freeze {name}")
                p.requires_grad = False        
        if self.config.train_stage == 2:
            for name, p in self.vision_encoder.named_parameters():
                if name.startswith('mlp') and self.config.train_full:
                    continue
                logger.info(f"Freeze {name}")
                p.requires_grad = False

        if self.config.model.freeze_clip:
            for name, p in self.clip_encoder.named_parameters():
                logger.info(f"Freeze {name}")
                p.requires_grad = False

        print('Total Params:', sum([p.numel() for _, p in self.head.named_parameters()])/1024/1024, 'MB')
        
        # criterions
        self.format_loss = VTC_VTM_Loss(False)
        self.prompt_loss = VTC_VTM_Loss(False)
        self.score_loss = nn.MSELoss() if config.criterion.loss_type=='mse' \
                    else nn.CrossEntropyLoss() if config.criterion.loss_type=='ce' \
                    else nn.SmoothL1Loss()
        self.rank_loss = Ranking_Loss()

    def no_weight_decay(self):
        ret = {"temp"}
        # ret.update(
        #     {"vision_encoder." + k for k, _ in self.vision_encoder.named_parameters()}
        # )
        # no weight decay for LLM if training
        # ret.update(
        #     {"text_encoder." + k for k, _ in self.clip_encoder.named_parameters()}
        # )

        return ret
    
    @torch.no_grad()
    def clip_contrastive_temperature(self):
        """Seems only used during pre-training"""
        self.temp.clamp_(min=self.temp_min)

    def forward(self, image, text, idx, prompt, is_text_prompt, score):
        """forward and calculate loss."""
        image.requires_grad = True

        B, device = image.shape[0], image.device

        if self.config.train_stage == 2:
            image = image.flatten(0, 1)
            
        self.clip_contrastive_temperature()
        vision_embeds = self.encode_vision(image)
        
        prompt_embeds = torch.zeros([B,vision_embeds.shape[-1]], dtype=vision_embeds.dtype).to(device)
        text_prompts = [p for p, v in zip(prompt, is_text_prompt) if v]
        image_prompts = [p for p, v in zip(prompt, is_text_prompt) if not v]
        is_image_prompt = torch.where(~is_text_prompt)
        is_text_prompt = torch.where(is_text_prompt)
        if text_prompts:
            prompt_embeds[is_text_prompt] = self.encode_text(self.tokenizer(text_prompts).to(device))
        if image_prompts:
            prompt_embeds[is_image_prompt] = self.encode_image(image_prompts).to(device)

        if self.config.train_stage == 1:
            text_embeds = self.encode_text(text)
            loss_format = self.format_loss.vtc_loss(
                vision_embeds, text_embeds, idx, self.temp, all_gather=True
            )
            loss_prompt = self.prompt_loss.vtc_loss(
                vision_embeds, prompt_embeds, idx, self.temp, all_gather=True
            )
            return dict(
                loss_format=loss_format,
                loss_prompt=loss_prompt,
                loss_score=None,
                loss_rank=None,
            )
        
        elif self.config.train_stage == 2:
            output = self.head(
                vision_embeds.reshape(B, -1, vision_embeds.shape[-1]),
                prompt_embeds.unsqueeze(dim=1),
            )
            
            if self.config.criterion.loss_type=='ce':
                score = (5*score.flatten()).to(torch.long)
            else:    
                loss_score = self.score_loss(output, score.to(device))
            loss_rank = self.rank_loss.rk_loss(output, score.to(device))
            return dict(
                loss_format=None,
                loss_prompt=None,
                loss_score=loss_score,
                loss_rank=loss_rank,
            )
        
    def generate(self, image, text, idx, prompt, is_text_prompt):

        B, device = image.shape[0], image.device
        image = image.flatten(0, 1)

        self.clip_contrastive_temperature()
        vision_embeds = self.encode_vision(image)
        
        prompt_embeds = torch.zeros([B,vision_embeds.shape[-1]], dtype=vision_embeds.dtype).to(device)
        text_prompts = [p for p, v in zip(prompt, is_text_prompt) if v]
        image_prompts = [p for p, v in zip(prompt, is_text_prompt) if not v]
        is_image_prompt = torch.where(~is_text_prompt)
        is_text_prompt = torch.where(is_text_prompt)
        if text_prompts:
            prompt_embeds[is_text_prompt] = self.encode_text(self.tokenizer(text_prompts).to(device))
        if image_prompts:
            prompt_embeds[is_image_prompt] = self.encode_image(image_prompts).to(device)

        output = self.head(
                vision_embeds.reshape(B, -1, vision_embeds.shape[-1]),
                # prompt_embeds = prompt_embeds.unsqueeze(dim=1),
            )
        return output

    def encode_vision(self, image):

        vision_embeds = self.vision_encoder(image)
        
        return vision_embeds
    
    def encode_image(self, image):

        image = [self.processor(Image.open(i)) for i in image]
        image = torch.stack(image).to(torch.bfloat16).to(self.config.device)
        image_embeds = self.clip_encoder.encode_image(image)
        return image_embeds

    def encode_text(self, text):

        text_embeds = self.clip_encoder.encode_text(text)
        return text_embeds

    def build_vision_encoder(self):

        vision_encoder_config = AutoConfig.from_pretrained(self.config.model.vision_encoder.config, trust_remote_code=True)
        vision_encoder = AutoModel.from_config(vision_encoder_config, trust_remote_code=True).to(torch.bfloat16)

        return vision_encoder

    def build_clip_encoder(self):

        clip_encoder, _, clip_processor = open_clip.create_model_and_transforms(self.config.model.clip_encoder.name, pretrained=False)
        clip_encoder.eval()
        self.processor = clip_processor
        return clip_encoder

    def build_head(self):

        head_config = AutoConfig.from_pretrained(self.config.model.head.config, trust_remote_code=True)
        head = AutoModel.from_config(head_config, trust_remote_code=True).to(torch.bfloat16)

        return head

    def load_checkpoint(self):
        new_ckpt = {}

        ### load vision_encoder
        if hasattr(self.config.model.vision_encoder, 'ckpt'):
            vision_ckpt_path = self.config.model.vision_encoder.ckpt
            logger.info(f"Load vision_encoder checkpoint from {vision_ckpt_path}")
            if vision_ckpt_path.endswith('.safetensors'):
                ckpt = load_file(vision_ckpt_path)
                for k, v in ckpt.items():
                    new_ckpt['vision_encoder.'+k] = v
            elif vision_ckpt_path.endswith('.pt'):
                ckpt = torch.load(vision_ckpt_path)
                ckpt = ckpt['module'] if 'module' in ckpt.keys() else ckpt
                for k,v in ckpt.items():
                    if k.startswith('vision_encoder.'):
                        new_ckpt[k] = v
        else:
            logger.info(f"Load vision_encoder checkpoint from scratch")
            for k,v in self.vision_encoder.named_parameters():
                new_ckpt['vision_encoder.'+k] = v

        ### load head
        if hasattr(self.config.model.head, 'ckpt'):
            head_ckpt_path = self.config.model.head.ckpt
            logger.info(f"Load head checkpoint from {head_ckpt_path}")
            if head_ckpt_path.endswith('.safetensors'):
                ckpt = load_file(head_ckpt_path)
                for k, v in ckpt.items():
                    new_ckpt['head.'+k] = v
            elif head_ckpt_path.endswith('.pt'):
                ckpt = torch.load(head_ckpt_path)
                ckpt = ckpt['module'] if 'module' in ckpt.keys() else ckpt
                for k,v in ckpt.items():
                    if k.startswith('qformer.'):
                        new_ckpt[k.replace('qformer','head')] = v
                    if k.startswith('vision_encoder.mlp'):
                        new_ckpt[k] = v
        else:
            logger.info(f"Load head checkpoint from scratch")
            for k,v in self.head.named_parameters():
                new_ckpt['head.'+k] = v

        ### load clip
        assert hasattr(self.config.model.clip_encoder, 'ckpt')
        clip_ckpt_path = self.config.model.clip_encoder.ckpt
        logger.info(f"Load clip checkpoint from {clip_ckpt_path}")
        if clip_ckpt_path.endswith('.safetensors'):
            ckpt = load_file(clip_ckpt_path)
            for k, v in ckpt.items():
                new_ckpt['clip_encoder.'+k] = v

        msg = self.load_state_dict(new_ckpt, strict=True)
        logger.info(msg)
