from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel, LMSDiscreteScheduler
from PIL import Image
import torchvision.transforms as transforms
import torch
import loguru
import numpy as np
from tqdm import tqdm
from torch import autocast


SD_PATH = '/ssd/sdf/Ckpts/stable-diffusion-v1-5'
CLIP_PATH='/ssd/sdf/Ckpts/clip-vit-large-patch14'

device = "cuda" if torch.cuda.is_available() else "cpu"
vae = AutoencoderKL.from_pretrained(
    SD_PATH,
    subfolder='vae',
    use_auth_token=True
)
vae = vae.to(device)

tokenizer = CLIPTokenizer.from_pretrained(
    CLIP_PATH,
    use_auth_token=True
)
encoder = CLIPTextModel.from_pretrained(
    CLIP_PATH,
    use_auth_token=True
)
encoder = encoder.to(device)

unet = UNet2DConditionModel.from_pretrained(
    SD_PATH,
    subfolder='unet',
    use_auth_token=True
)
unet = unet.to(device)

scheduler = LMSDiscreteScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    num_train_timesteps=1000
)

loguru.logger.info("Models loaded successfully")

def get_text_embeds(prompts):
    text_embeddings = []
    for prompt in prompts:
        loguru.logger.info(f"Prompt: {len(prompt)}")

        # Tokenize text and get embeddings
        text_input = tokenizer(
            prompt, padding='max_length', max_length=tokenizer.model_max_length,
            truncation=True, return_tensors='pt')
        uncond_input = tokenizer(
            [''*len(prompt)] , padding='max_length',
            max_length=tokenizer.model_max_length, return_tensors='pt')

        with torch.no_grad():
            text_embedding = encoder(text_input.input_ids.to(device))[0]
            uncond_embedding = encoder(uncond_input.input_ids.to(device))[0]

        # Cat for final embeddings

        text_embedding = torch.cat([uncond_embedding, text_embedding])
        text_embeddings.append(text_embedding)
    #text_embeddings = torch.stack([np.array(text_embedding) for text_embedding in text_embeddings] ,axis=0)
    text_embeddings = np.stack(
        [text_embedding.cpu().numpy() if isinstance(text_embedding, torch.Tensor) else np.array(text_embedding) for text_embedding in text_embeddings],
        axis=0
    )
    loguru.logger.info(f"Text embeddings shape {text_embeddings.shape}")
    return text_embeddings


def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    image_tensor = transform(image).unsqueeze(0).to("cuda")
    return image_tensor


def encode_img_latents(imgs):
  if not isinstance(imgs, list):
    imgs = [imgs]
  print(type(imgs[0]))
  img_arr = np.stack([np.array(img) for img in imgs], axis=0)
  img_arr = img_arr / 255.0
  img_arr = torch.from_numpy(img_arr).float().permute(0, 3, 1, 2)
  img_arr = 2 * (img_arr - 0.5)

  latent_samples = vae.encode(img_arr.to(device)).latent_dist.sample()
  latent_samples *= 0.18215

  return latent_samples


def produce_latents(
        text_embeddings,
        image_lantent,
        empty_text_embedding,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        latents=None
):
    if latents is None:
        #random get the noise
        latents = torch.randn((text_embeddings.shape[0] // 2, unet.in_channels, \
                               height // 8, width // 8))
    latents, image_lantent = latents.to(device), image_lantent.to(device)

    scheduler.set_timesteps(num_inference_steps)
    latents = latents * scheduler.sigmas[0]

    with autocast('cuda'):
        for i, t in tqdm(enumerate(scheduler.timesteps)):
            noise_pred_list,sim_list = [],[]

            for s_i,text_embedding in enumerate(text_embeddings):
                text_embedding = torch.from_numpy(text_embedding).float().to(device)
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_model_input = torch.cat([latents] * 2)
                image_lantents_input = torch.cat([image_lantent] * 2)
                sigma = scheduler.sigmas[i]
                latent_model_input = latent_model_input / ((sigma ** 2 + 1) ** 0.5)
                image_lantents_input = image_lantents_input / ((sigma ** 2 + 1) ** 0.5)

                # predict the noise residual
                with torch.no_grad():
                    noise_pred_from_empty_text = unet(image_lantents_input,t,encoder_hidden_states=empty_text_embedding)['sample']
                    noise_pred_s_i = unet(latent_model_input, t, encoder_hidden_states=text_embedding)['sample']
                    noise_pred_list.append(noise_pred_s_i)
                    #compute the cosine similarity between the two noise_pred
                    cosine_similarity = abs(torch.nn.functional.cosine_similarity(noise_pred_s_i.view(1,-1),noise_pred_from_empty_text.view(1,-1)))
                    loguru.logger.info(f"cosine_similarity: {cosine_similarity}")
                    sim_list.append(cosine_similarity)
            loguru.logger.info(f"sim_list: {sim_list}")
            # find the max index in cosine similarity
            max_index = sim_list.index(min(sim_list))
            loguru.logger.info(f"max_index: {max_index}")
            # perform guidance
            noise_pred = noise_pred_list[max_index]
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, t, image_lantent).prev_sample
            #latents_from_image = scheduler.step(noise_pred_from_image, i, image_lantent).prev_sample

    return latents


# test_latents = produce_latents(test_embeds)
# print(test_latents)
# print(test_latents.shape)


def decode_img_latents(latents):
  latents = 1 / 0.18215 * latents

  with torch.no_grad():
    imgs = vae.decode(latents)

  imgs = (imgs.sample / 2 + 0.5).clamp(0, 1)
  imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
  imgs = (imgs * 255).round().astype('uint8')
  pil_images = [Image.fromarray(image) for image in imgs]
  return pil_images

# imgs = decode_img_latents(test_latents)
# imgs[0]


def prompt_to_img(prompts, height=512, width=512, num_inference_steps=50,
                  guidance_scale=7.5, latents=None,image_latents=None):
    if isinstance(prompts, str):
        prompts = [prompts]

    # Prompts -> text embeds
    text_embeds = get_text_embeds(prompts)

    # Text embeds -> img latents
    for text_embed in text_embeds:
        latents,image_latents = produce_latents(
            text_embed, height=height, width=width, latents=latents,
            num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)

    # Img latents -> imgs
    imgs = decode_img_latents(latents)

    return imgs

#split_prompt=['Add a hat to the cat', 'Add a bow tie to the cat','Add a colored background']
split_prompt=['change the dress color to red','add some dot pattern to the dress']
text_embeddings = get_text_embeds(split_prompt)

empty_text_embedding = get_text_embeds([''])[0]
empty_text_embedding = torch.from_numpy(empty_text_embedding).float().to(device)

img_path ='/ssd/sdf/wangwenji/project/InstructPersonalizedImageGeneration/test.png'

# image_tensor = preprocess_image(img_path).to(device)
# loguru.logger.info(f"Image tensor shape {image_tensor.shape}")
image = Image.open(img_path).convert("RGB")

image_lantents = encode_img_latents([image])
# image_lantents = torch.from_numpy(image_lantents).float()
loguru.logger.info(f"Image latents shape {image_lantents.shape}")

test_latents = produce_latents(text_embeddings,image_lantents,empty_text_embedding)
loguru.logger.info(f"Latents shape {test_latents.shape}")

imgs = decode_img_latents(test_latents)
imgs[0].save('result.png')
