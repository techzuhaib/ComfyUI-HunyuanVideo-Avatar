import os
import numpy as np
from pathlib import Path
from loguru import logger
import torch
from einops import rearrange
import torch.distributed
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from hymm_sp.config import parse_args
from hymm_sp.sample_inference_audio import HunyuanVideoSampler
from hymm_sp.data_kits.audio_dataset import VideoAudioTextLoaderVal
from hymm_sp.data_kits.face_align import AlignImage

from transformers import WhisperModel
from transformers import AutoFeatureExtractor


MODEL_OUTPUT_PATH = "./weights"


def main():
    args = parse_args()
    models_root_path = Path(args.ckpt)

    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")

    # Create save folder to save the samples
    save_path = args.save_path if args.save_path_suffix=="" else f'{args.save_path}_{args.save_path_suffix}'
    if not os.path.exists(args.save_path):
        os.makedirs(save_path, exist_ok=True)

    # Load models
    rank = 0
    vae_dtype = torch.float16
    device = torch.device("cuda")
    
    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(args.ckpt, args=args, device=device)
    # Get the updated args
    args = hunyuan_video_sampler.args
    if args.cpu_offload:
        from diffusers.hooks import apply_group_offloading
        onload_device = torch.device("cuda")
        apply_group_offloading(hunyuan_video_sampler.pipeline.transformer, onload_device=onload_device, offload_type="block_level", num_blocks_per_group=1)

    wav2vec = WhisperModel.from_pretrained(f"{MODEL_OUTPUT_PATH}/ckpts/whisper-tiny/").to(device=device, dtype=torch.float32)
    wav2vec.requires_grad_(False)
    
    BASE_DIR = f'{MODEL_OUTPUT_PATH}/ckpts/det_align/'
    det_path = os.path.join(BASE_DIR, 'detface.pt')    
    align_instance = AlignImage("cuda", det_path=det_path)
    
    feature_extractor = AutoFeatureExtractor.from_pretrained(f"{MODEL_OUTPUT_PATH}/ckpts/whisper-tiny/")

    kwargs = {
            "text_encoder": hunyuan_video_sampler.text_encoder, 
            "text_encoder_2": hunyuan_video_sampler.text_encoder_2, 
            "feature_extractor": feature_extractor, 
        }
    video_dataset = VideoAudioTextLoaderVal(
            image_size=args.image_size,
            meta_file=args.input, 
            **kwargs,
        )

    sampler = DistributedSampler(video_dataset, num_replicas=1, rank=0, shuffle=False, drop_last=False)
    json_loader = DataLoader(video_dataset, batch_size=1, shuffle=False, sampler=sampler, drop_last=False)

    for batch_index, batch in enumerate(json_loader, start=1):

        fps = batch["fps"]
        videoid = batch['videoid'][0]
        audio_path = str(batch["audio_path"][0])
        save_path = args.save_path 
        output_path = f"{save_path}/{videoid}.mp4"
        output_audio_path = f"{save_path}/{videoid}_audio.mp4"

        if args.infer_min:
            batch["audio_len"][0] = 129
            
        samples = hunyuan_video_sampler.predict(args, batch, wav2vec, feature_extractor, align_instance)
        
        sample = samples['samples'][0].unsqueeze(0)                    # denoised latent, (bs, 16, t//4, h//8, w//8)
        sample = sample[:, :, :batch["audio_len"][0]]
        
        video = rearrange(sample[0], "c f h w -> f h w c")
        video = (video * 255.).data.cpu().numpy().astype(np.uint8)  # （f h w c)
        
        torch.cuda.empty_cache()

        final_frames = []
        for frame in video:
            final_frames.append(frame)
        final_frames = np.stack(final_frames, axis=0)
        
        if rank == 0:
            from hymm_sp.data_kits.ffmpeg_utils import save_video
            save_video(final_frames, output_path, n_rows=len(final_frames), fps=fps.item())
            os.system(f"ffmpeg -i '{output_path}' -i '{audio_path}' -shortest '{output_audio_path}' -y -loglevel quiet; rm '{output_path}'")


class LoadAvatarModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_path": ("STRING", {"default": "./weights/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states_fp8.pt"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "input_model"
    CATEGORY = "HunyuanVideo-Avatar"

    def input_model(self, model_path):
        model = model_path
        return (model,)


class LoadAvatarInput:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_path": ("STRING", {"default": "assets/test.csv"}),
            }
        }

    RETURN_TYPES = ("INPUT",)
    RETURN_NAMES = ("input",)
    FUNCTION = "input_avatar"
    CATEGORY = "HunyuanVideo-Avatar"

    def input_avatar(self, input_path):
        input = input_path
        return (input,)


class HunyuanVideoAvatar:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "input": ("INPUT",),
                "sample-n-frames": ("CONFIG",),
                "seed": ("INT", {"default": 128}),
                "image-size": ("INT", {"default": 704}),
                "cfg-scale": ("FLOAT", {"default": 7.5}),
                "infer-steps": ("INT", {"default": 50}),
                "use-deepcache": ("INT", {"default": 1}),
                "flow-shift-eval-video": ("FLOAT", {"default": 5.0}),
                "save-path": ("STRING", {"default": "./results-single"}),
            }
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("final_frames",)
    FUNCTION = "generate"
    CATEGORY = "HunyuanVideo-Avatar"

    def generate(self, model, input, sample-n-frames, seed, image-size, cfg-scale, infer-steps, use-deepcache, flow-shift-eval-video, save-path):

        
    
        main(model, input, sample-n-frames, seed, image-size, cfg-scale, infer-steps, use-deepcache, flow-shift-eval-video, save-path)
        
        return ( ) 
    
    
    
    
