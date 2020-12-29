import torch.nn as nn
import torchvision.transforms as transforms

from PIL import Image


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


def get_eval_images(args):
    loader = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),  # scale imported image
        transforms.ToTensor()])  # transform it into a torch tensor

    content_img = loader(Image.open(args.content_image)).unsqueeze(0)
    style_img = loader(Image.open(args.style_image)).unsqueeze(0)

    assert style_img.size() == content_img.size(), \
        "we need to import style and content images of the same size"

    return content_img, style_img
