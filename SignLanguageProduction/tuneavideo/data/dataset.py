import decord
decord.bridge.set_bridge('torch')

from torch.utils.data import Dataset
from einops import rearrange
import torch


class TuneAVideoDataset(Dataset):
    def __init__(
            self,
            video_path: str,
            prompt: str,
            width: int = 512,
            height: int = 512,
            n_sample_frames: int = 8,
            sample_start_idx: int = 0,
            sample_frame_rate: int = 1,
    ):
        self.video_path = video_path
        self.prompt = prompt
        self.prompt_ids = []

        self.width = width
        self.height = height
        self.n_sample_frames = n_sample_frames
        self.sample_start_idx = sample_start_idx
        self.sample_frame_rate = sample_frame_rate

    def __len__(self):
        return (len(self.video_path))

    def __getitem__(self, index):
        # load and sample video frames
        vr = decord.VideoReader(self.video_path[index], width=self.width, height=self.height)
        #print(len(vr))
        sample_index = list(range(self.sample_start_idx, len(vr), self.sample_frame_rate))[:self.n_sample_frames]
        video = vr.get_batch(sample_index)
        #if video.shape[0]<self.n_sample_frames:
            # repeat the last frame
            #last_frame = video[-1].unsqueeze(0)  # Extract the last frame and add a new dimension
            #repeated_frames = last_frame.repeat(self.n_sample_frames - video.shape[0], 1, 1, 1)  # Repeat the last frame to fill up to necessary number frames
            #video = torch.cat([video, repeated_frames], dim=0)  # Concatenate the original video and the repeated frames
        video = rearrange(video, "f h w c -> f c h w")

        example = {
            "pixel_values": (video / 127.5 - 1.0),
            "prompt_ids": self.prompt_ids[index],
            "prompt": self.prompt[index]
        }

        return example
