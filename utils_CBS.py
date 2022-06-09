import glob
import re
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from model import transformer_net


# image and label from train_DA_V2 come in a shape like (D, H, W)
def class_base_styling(image, label, class_id=[7], style_id=0, loss='crossentropy', j=0):

    if loss == 'dice':
        label = np.argmax(image, axis=0)

    fg_image = get_masked_image(label, image.transpose(1, 2, 0), category=class_id, bg=0)
    bg_image = get_masked_image(label, image.transpose(1, 2, 0), category=class_id, bg=1)
    fg_image = fg_image.transpose(2, 0, 1)
    bg_image = bg_image.transpose(2, 0, 1)

    # save_image("fg_image.png", fg_image)
    # save_image("bg_image.png", bg_image)

    image_style1 = stylize(image, style_id=style_id)

    # save_image("./images/"+str(j)+"imagestyle.png", image_style1)

    # Apply local style to fg
    fg_styled = image_style1 * (fg_image != 0)

    output = fg_styled + bg_image

    # save_image("./images/"+str(j)+"final_image.png", output)

    return output

def get_style_list_size(path='./model/styles/*.pth'):
    return len(glob.glob(path))


@torch.no_grad()
# image and label must be in the shape as follows: (H, W, D)
def get_masked_image(label, image, category, bg=0):
    output = label[:, :]

    # bin_mask = (output == category).astype('uint8')
    bin_mask = np.isin(output, category).astype('uint8')

    if bg:
        bin_mask = 1 - bin_mask

    masked = (bin_mask[:, :, None] * image) * 255

    return masked


def save_image(fname, data):
    img = data.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(fname)

#apply style to the image
def stylize(content_image, style_id=0):
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.transpose(1, 0).transpose(1, 2)

    if torch.cuda.is_available():
        content_image = content_image.unsqueeze(0).to(0)
    else:
        content_image = content_image.unsqueeze(0)

    with torch.no_grad():
        model_path = glob.glob('./model/styles/*.pth')[style_id]
        style_model = transformer_net.TransformerNet()
        state_dict = torch.load(model_path)

        # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
        for k in list(state_dict.keys()):
            if re.search(r'in\d+\.running_(mean|var)$', k):
                del state_dict[k]
        style_model.load_state_dict(state_dict)
        if torch.cuda.is_available():
            style_model.to(0)

        output = style_model(content_image).cpu()

    return output[0]
