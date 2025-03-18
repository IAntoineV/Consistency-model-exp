import torch

import torchvision
import numpy as np
from diffusers import ConsistencyModelPipeline

# Karras diffusion Hyperparameters
sigma_data: float = 0.5
sigma_max=80.0
sigma_min=0.002
rho=7.0
weight_schedule="karras"

device = "cuda" if torch.cuda.is_available() else "cpu"

# TODO change name of variable model/pipe

@torch.no_grad()
def iterative_inpainting(
    pipe,
    images,
    x,
    ts,
    t_min=0.002,
    t_max=80.0,
    rho=7.0,
    steps=40,
    generator=None,
    use_square_mask = False,
    is_imagenet=False,
    class_labels=None,
):
    assert not is_imagenet or class_labels, "class_labels need to be provided for ImageNet models" 
    image_size = x.shape[-1]

    if use_square_mask:
        mask=create_square_mask(image_size)

    else:
        mask = create_s_mask(image_size)
    
    def replacement(x0, x1):
        x_mix = x0 * mask + x1 * (1 - mask)
        return x_mix

    t_max_rho = t_max ** (1 / rho)
    t_min_rho = t_min ** (1 / rho)
    s_in = x.new_ones([x.shape[0]])
    print(f"{s_in=}")
    images = replacement(images, -torch.ones_like(images))

    sigmas = get_sigmas_karras(steps, sigma_min, sigma_max, rho, device=device)
 
    if class_labels:
        class_labels = pipe.prepare_class_labels(batch_size=1, device=device, class_labels=class_labels)
    
    for i in range(len(ts) - 1):
        t = (t_max_rho + ts[i] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        print(f"Timestep T={ts[i]}")
        x0 = denoise(pipe, x, sigmas[i], class_labels=class_labels, is_imagenet=is_imagenet)[1]

        x0 = torch.clamp(x0, -1.0, 1.0)
        x0 = replacement(images, x0)
        display_as_pilimg(x0, title=f"Timestep T={ts[i]}")
        next_t = (t_max_rho + ts[i + 1] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        next_t = np.clip(next_t, t_min, t_max)
        x = x0 + torch.randn_like(x) * np.sqrt(next_t**2 - t_min**2)

    return x, images


def denoise(pipe, x_t, sigmas, class_labels=None, is_imagenet=False):
    import torch.distributed as dist
    c_skip, c_out, c_in = [
        append_dims(x, x_t.ndim)
        for x in get_scalings_for_boundary_condition(sigmas)
    ]
    rescaled_t = 1000 * 0.25 * torch.log(sigmas + 1e-44)
    input_unet = c_in * x_t
    # if is_imagenet:
    #     input_unet.to(torch.float16)
    #     class_labels.to(torch.float16)
    #     rescaled_t.to(torch.float16)
    model_output = pipe.unet(input_unet, rescaled_t, class_labels=class_labels).sample
    denoised = c_out * model_output + c_skip * x_t
    return model_output, denoised


def get_scalings_for_boundary_condition(sigma):
    c_skip = sigma_data**2 / (
        (sigma - sigma_min) ** 2 + sigma_data**2
    )
    c_out = (
        (sigma - sigma_min)
        * sigma_data
        / (sigma**2 + sigma_data**2) ** 0.5
    )
    c_in = 1 / (sigma**2 + sigma_data**2) ** 0.5
    return c_skip, c_out, c_in

def get_sigmas_karras(n, sigma_min, sigma_max, rho=7.0, device="cpu"):
    """Constructs the noise schedule of Karras et al. (2022)."""
    ramp = torch.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas).to(device)



def create_square_mask(img_size):
    h,w = img_size, img_size
    hcrop, wcrop = h // 2, w // 2
    corner_top, corner_left = h // 4, int(0.45 * w)

    # Create mask: Initialize with ones (all visible), then set cropped area to 0 (masked area)
    mask = torch.ones((3, h, w), device=device)  # Assuming 3 channels (RGB)
    mask[:, corner_top:corner_top + hcrop, corner_left:corner_left + wcrop] = 0
    display_as_pilimg(mask)
    return mask

def create_s_mask(img_size):
    from PIL import Image, ImageDraw, ImageFont

    # create a blank image with a white background
    img = Image.new("RGB", (img_size, img_size), color="white")

    # get a drawing context for the image
    draw = ImageDraw.Draw(img)

    # load a font
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    font = ImageFont.truetype(font_path, 250)
    # draw the letter "C" in black
    draw.text((50, 0), "S", font=font, fill=(0, 0, 0))

    # convert the image to a numpy array
    img_np = np.array(img)
    plt.imshow(img_np)
    img_np = img_np.transpose(2, 0, 1)
    img_th = torch.from_numpy(img_np).to(device)
    display_as_pilimg(img_th)

    img_gray = img.convert("L")

    # Convert the grayscale image to a numpy array
    img_np = np.array(img_gray)

    # Thresholding to create the mask
    # Set pixel values inside the "S" (dark area) to 1, others to 0
    mask = (img_np > 128).astype(np.float32)  # 128 is the threshold

    # Convert the mask to a PyTorch tensor
    mask = torch.from_numpy(mask).to(device)
    print(f"{mask.shape=}")
    return mask


## utils 

def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]


def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])

def display_as_pilimg_bigger(t, figsize=(3, 3), title=None):  # Set the desired figure size
    t = 0.5 + 0.5 * t.to('cpu')
    t = t.squeeze()
    t = t.clamp(0., 1.)
    pil_img = torchvision.transforms.ToPILImage()(t)

    # Display using matplotlib
    plt.figure(figsize=figsize)
    if title:
        plt.title(title)
    plt.imshow(pil_img)
    plt.axis('off')  # Hide axes
    plt.show()
    
    return pil_img

def pilimg_to_tensor(pil_img):
  t = torchvision.transforms.ToTensor()(pil_img)
  t = 2*t-1 # [0,1]->[-1,1]
  t = t.unsqueeze(0)
  t = t.to(device)
  return(t)

import torchvision
import matplotlib.pyplot as plt

def display_as_pilimg(t, title=None):
    t = 0.5 + 0.5 * t.to('cpu')
    t = t.squeeze()
    t = t.clamp(0., 1.)
    pil_img = torchvision.transforms.ToPILImage()(t)

    # Display using Matplotlib with a title
    plt.imshow(pil_img)
    plt.axis("off")  # Hide axis
    if title:
        plt.title(title)
    plt.show()

    return pil_img



if __name__=="__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load pretrained BedRoom model
    pipe_bedroom = ConsistencyModelPipeline.from_pretrained("openai/diffusers-cd_bedroom256_lpips")

    # Load Bedroom sample
    x_true_pil = Image.open(f'LSUN_bedroom_sample.jpg').resize((256, 256))
    x_true = pilimg_to_tensor(x_true_pil).to(device) 
    res = display_as_pilimg(x_true)

    x, images = iterative_inpainting(
    pipe_bedroom,
    images=x_true,
    x=x_true.clone(),
    ts=[i for i in range(40,0,-1)])