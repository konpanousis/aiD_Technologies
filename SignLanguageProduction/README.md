# aiD project: The Sign Language Production Module.

Sign Language Proudction (SLP) is incredibly important as it
allows deaf and hard-of-hearing individuals to access information and communication that they 
might not otherwise have access to. Just like spoken and written language, SL is a complete and 
complex language that enables expression and communication. However, not everyone can understand 
or use SL, which can lead to social isolation and exclusion for the deaf and hard-of-hearing community. 
By providing accurate and effective translation from text to SL, individuals who rely on SL can access 
information on the internet, 
in educational settings, and in daily interactions with others, ultimately promoting inclusivity and equality for all.

Our work, based on the success of [1] on video generation, implements a novel approach for end-to-end text to SL video translation using an autoregressive method which utilizes the power of a video diffusion architecture.
To the best of our knowledge there are no existing models that can directly perform this task, as previous models.

## Inference:
While we adhere to the pipeline of the tune-a-video model during the training of the SLP model, the inference process diverges. This is primarily because the SLP model is generative, while the tune-a-video model is primarily utilized for editing purposes. During inference, instead of inputting a video to the DDIM inversion model, we use a single repetitive frame to impart structure to the resulting SL video. However, having a stagnant reference video as the input, results in a stagnant video as the output. It was discovered that introducing noise to the input video prevented the resulting video from becoming stagnant (4 DDPM forward steps were found enough to get the desired output). 

Therefore, the inference pipeline unfolds as follows: replicate a single frame 12 times to generate a stagnated input video, subject it to 4 forward steps of noise injection, pass it through the DDIM backward algorithm that feeds the latents of each step to the U-net alongside the text prompt to generate the final video.

The inference time is about 1.5 minute to generate a SL video of 12 frames with one V100 15GB GPU.

## Acknowledgements
This  research  was supported  by  the  the European Union’s Horizon2020 research and innovation program,
under grant agreement  No  872139,  project  aiD.

## References 
[1] Jay Zhangjie Wu et al. Tune-A-Video: One-Shot Tuning of Image Diffusion Models for Text-to-Video Generation. 2023. arXiv: 2212.11565 [cs.CV].

