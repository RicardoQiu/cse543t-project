##################################################################
# This is the main function to run our model
##################################################################
import sys
import argparse
import torch
import torchvision.models as models
import time
import matplotlib.pyplot as plt

from skimage.io import imsave

from loss import LapStyleLoss, TVLoss
from dataloader import get_eval_images
from optimizer import get_optimizer
from styleNet import get_style_model_and_losses
from utils import check_paths, transfer_color

torch.manual_seed(42)


def stylize(args):
    """Run the style transfer."""
    device = torch.device("cuda" if args.cuda else "cpu")
    content_img, style_img = get_eval_images(args)
    content_img = content_img.to(device, torch.float)
    style_img = style_img.to(device, torch.float)
    cnn = models.vgg19(pretrained=True).features.to(device).eval()
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn, device, cnn_normalization_mean,
                                                                     cnn_normalization_std,
                                                                     style_img,
                                                                     content_img)
    print(model)
    start = time.time()
    # input_img = content_img.clone()
    # input_img = style_img.clone()
    input_img = torch.randn_like(content_img).to(device)
    optimizer = get_optimizer(input_img, args)
    print('Optimizing..')
    run = [0]
    losses = []
    while run[0] <= args.num_steps:

        def closure():
            # correct the values of updated input image
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= args.style_weight / 5
            content_score *= args.content_weight

            if args.reg:
                if args.reg.lower() == "l2":
                    regularizer = args.reg_weight * 0.5 * torch.sum(torch.norm(input_img))
                elif args.reg.lower() == "tv":
                    tv = TVLoss()
                    regularizer = args.reg_weight * tv(input_img)
            else:
                regularizer = 0.0

            if args.lap_weight:
                lap_loss = args.lap_weight * LapStyleLoss(device, content_img)(input_img)
            else:
                lap_loss = 0.0

            loss = style_score + content_score + regularizer + lap_loss
            losses.append(loss.item())
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return loss

        optimizer.step(closure)

    # a last correction...
    input_img.data.clamp_(0, 1)

    print(f"running time {time.time() - start:.2f}s")

    image = input_img.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = image.detach().numpy().transpose(1, 2, 0)
    image = (image * 255).astype('uint8')

    if args.color_prev:
        content_img = content_img.cpu().clone().squeeze(0).detach().numpy().transpose(1, 2, 0)
        content_img = (content_img * 255).astype('uint8')

        image = transfer_color(content_img, image)

    imsave(f"{args.output_image}/{args.content_image.split('/')[-1].replace('.jpg', '')}_"
           f"{args.style_image.split('/')[-1].replace('.jpg', '')}_{args.optimizer}"
           f"{'_' + args.reg if args.reg else ''}"
           f"{'_lap' if args.lap_weight else ''}{'_color_prev' if args.color_prev else ''}_stylize.jpg", image)

    plt.plot(losses)
    plt.title("Losses along iterations")
    plt.savefig(f"{args.output_image}/{args.content_image.split('/')[-1].replace('.jpg', '')}_"
                f"{args.style_image.split('/')[-1].replace('.jpg', '')}_{args.optimizer}"
                f"{'_' + args.reg if args.reg else ''}"
                f"{'_lap' if args.lap_weight else ''}{'_color_prev' if args.color_prev else ''}_losses.jpg")


def main():
    # add more configurations as we go, this is just a sample
    main_arg_parser = argparse.ArgumentParser(description="parser for fast-neural-style")
    subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")

    eval_arg_parser = subparsers.add_parser("eval", help="parser for evaluation/stylizing arguments")
    eval_arg_parser.add_argument("--num-steps", type=int, default=500,
                                 help="num-steps")
    eval_arg_parser.add_argument("--lr", type=float, default=1,
                                 help="choose learning rate")
    eval_arg_parser.add_argument("--content-image", type=str, required=True,
                                 help="path to content image you want to stylize")
    eval_arg_parser.add_argument("--style-image", type=str, required=True,
                                 help="path to style image you want to stylize")
    eval_arg_parser.add_argument("--image-size", type=int, default=512,
                                 help="image size")

    eval_arg_parser.add_argument("--style-weight", type=float, default=10000,
                                 help="style_weight")
    eval_arg_parser.add_argument("--content-weight", type=float, default=0.01,
                                 help="style_weight")

    eval_arg_parser.add_argument("--output-image", type=str, required=True,
                                 help="path for saving the output image")
    eval_arg_parser.add_argument('--cuda', action='store_true', help='enables cuda')

    eval_arg_parser.add_argument("--optimizer", type=str, default="L-BFGS",
                                 help="choose optimizer")
    # regularization term
    eval_arg_parser.add_argument("--reg", type=str, default=None,
                                 help="choose regularizer")
    eval_arg_parser.add_argument("--reg-weight", type=float, default=0.0,
                                 help="choose regularization strength")

    # laplacian term
    eval_arg_parser.add_argument("--lap-weight", type=float, default=None,
                                 help="choose laplacian weights")

    # color preserving
    eval_arg_parser.add_argument('--color-prev', action='store_true', help='enables color preserving')

    args = main_arg_parser.parse_args()

    print(args)

    if args.subcommand is None:
        print("ERROR: specify either train or eval")
        sys.exit(1)
    if args.cuda and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)

    if args.cuda:
        args.image_size = 512

    if args.subcommand == "train":
        pass
        # train(args)
        # train the model
    else:
        check_paths(args)
        stylize(args)
        # stylize


if __name__ == '__main__':
    main()
