from diffusers import StableDiffusionPipeline
import torch
from matplotlib import pyplot as plt


 
model_id1 = "dreamlike-art/dreamlike-diffusion-1.0"
model_id2 = "stabilityai/stable-diffusion-xl-base-1.0"

#load the models
pipe1 = StableDiffusionPipeline.from_pretrained(model_id1, torch_dtype=torch.float16)
pipe2 = StableDiffusionPipeline.from_pretrained(model_id2, torch_dtype=torch.float16)


#set the prompts
prompt1 = "A beautiful sunset over a calm ocean"
prompt2 = "A beautiful sunset over a calm ocean"

#generate the images
image1 = pipe1(prompt1).images[0]
image2 = pipe2(prompt2).images[0]

#save the images
image1.save("image1.png")
image2.save("image2.png")

#display the images
# plt.imshow(image1)
# plt.show()