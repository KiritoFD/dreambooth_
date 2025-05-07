from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", revision="main")
pipe.save_pretrained("./stable-diffusion-v1-5")