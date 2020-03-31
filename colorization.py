import os
import argparse
import numpy as np
import torch
from PIL import Image
import math, time
import cv2

from utils.file_utils import image_files, load_as_tensor, Tensor2PIL, split_to_batches
from utils.image_precossing import _sigmoid_to_tanh, _tanh_to_sigmoid, _add_batch_one
from derivable_models.derivable_generator import get_derivable_generator
from inversion.inversion_methods import get_inversion
from utils.manipulate import Color_loss, colorization_images
from inversion.losses import get_loss
from models.model_settings import MODEL_POOL


def main(args):
    os.makedirs(args.outputs, exist_ok=True)
    generator = get_derivable_generator(args.gan_model, args.inversion_type, args)
    loss = get_loss(args.loss_type, args)
    cor_loss = Color_loss(loss)
    generator.cuda()
    loss.cuda()
    inversion = get_inversion(args.optimization, args)
    image_list = image_files(args.target_images)
    frameSize = MODEL_POOL[args.gan_model]['resolution']

    for i, images in enumerate(split_to_batches(image_list, 1)):
        print('%d: Processing %d images :' % (i + 1, 1), end='')
        pt_image_str = '%s\n'
        print(pt_image_str % tuple(images))

        image_name_list = []
        image_tensor_list = []
        for image in images:
            image_name_list.append(os.path.split(image)[1])
            image_tensor_list.append(_add_batch_one(load_as_tensor(image)))
        y_gt = _sigmoid_to_tanh(torch.cat(image_tensor_list, dim=0)).cuda()
        # Invert
        latent_estimates, history = inversion.invert(generator, y_gt, cor_loss, batch_size=1, video=args.video)
        # Get Images
        y_estimate_list = torch.split(torch.clamp(_tanh_to_sigmoid(generator(latent_estimates)), min=0., max=1.).cpu(), 1, dim=0)
        # Save
        for img_id, image in enumerate(images):
            up_gray = colorization_images(image_tensor_list[img_id])
            y_gray_pil = Tensor2PIL(up_gray, mode='L')
            y_gray_pil.save(os.path.join(args.outputs, '%s-%s.png' % (image_name_list[img_id], 'gray')))

            Y_gt = Tensor2PIL(image_tensor_list[img_id], mode='RGB').convert('YCbCr')
            y_estimate_pil = Tensor2PIL(y_estimate_list[img_id], mode='RGB').convert('YCbCr')

            _, Cb, Cr = y_estimate_pil.split()
            Y, _, _ = Y_gt.split()
            y_colorization = Image.merge('YCbCr', (Y, Cb, Cr))
            y_colorization.convert('RGB').save(os.path.join(args.outputs, '%s-%d.png' % (image_name_list[img_id], math.floor(time.time()))))
            # Create video
            if args.video:
                print('Create GAN-Inversion video.')
                video = cv2.VideoWriter(
                    filename=os.path.join(args.outputs, '%s_inversion.avi' % image_name_list[img_id]),
                    fourcc=cv2.VideoWriter_fourcc(*'MJPG'),
                    fps=args.fps,
                    frameSize=(frameSize, frameSize))
                print('Save frames.')
                for i, sample in enumerate(history):
                    image = torch.clamp(_tanh_to_sigmoid(generator(sample)), min=0., max=1.).cpu()
                    image_pil = Tensor2PIL(image, mode='RGB').convert('YCbCr')
                    _, Cb, Cr = image_pil.split()
                    y_colorization = Image.merge('YCbCr', (Y, Cb, Cr)).convert('RGB')
                    image_cv2 = cv2.cvtColor(np.asarray(y_colorization), cv2.COLOR_RGB2BGR)
                    # image_cv2 = cv2.cvtColor(np.asarray(image_pil.convert('RGB')), cv2.COLOR_RGB2BGR)
                    video.write(image_cv2)
                video.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Colorization')
    # Image Path and Saving Path
    parser.add_argument('-i', '--target_images',
                        default='./examples/colorization/church-o',
                        help='Directory with images for colorizing')
    parser.add_argument('-o', '--outputs',
                        default='./color_output-420',
                        help='Directory for storing generated images')
    # Parameters for Multi-Code GAN Inversion
    parser.add_argument('--inversion_type', default='PGGAN-Multi-Z',
                        help='Inversion type, PGGAN-Multi-Z for Multi-Code-GAN prior.')
    parser.add_argument('--composing_layer', default=4,
                        help='Composing layer in multi-code gan inversion methods.', type=int)
    parser.add_argument('--z_number', default=20,
                        help='Number of the latent codes.', type=int)
    # Loss Parameters
    parser.add_argument('--image_size', default=256,
                        help='Size of images for perceptual model', type=int)
    parser.add_argument('--loss_type', default='Combine',
                        help="['VGG', 'L1', 'L2', 'Combine']. 'Combine' means using L2 and Perceptual Loss.")
    parser.add_argument('--vgg_loss_type', default='L1',
                        help="['L1', 'L2']. The loss used in perceptual loss.")
    parser.add_argument('--vgg_layer', default=16,
                        help='The layer used in perceptual loss.', type=int)
    parser.add_argument('--l1_lambda', default=0.,
                        help="Used when 'loss_type' is 'Combine'. Trade-off parameter for L1 loss.", type=float)
    parser.add_argument('--l2_lambda', default=1.,
                        help="Used when 'loss_type' is 'Combine'. Trade-off parameter for L2 loss.", type=float)
    parser.add_argument('--vgg_lambda', default=1.,
                        help="Used when 'loss_type' is 'Combine'. Trade-off parameter for Perceptual loss.", type=float)
    # Optimization Parameters
    parser.add_argument('--optimization', default='GD',
                        help="['GD', 'Adam']. Optimization method used.")  # inversion_type
    parser.add_argument('--init_type', default='Normal',
                        help="['Zero', 'Normal']. Initialization method. Using zero init or Gaussian random vector.")
    parser.add_argument('--lr', default=1.,
                        help='Learning rate.', type=float)
    parser.add_argument('--iterations', default=1500,
                        help='Number of optimization steps.', type=int)
    # Generator Setting
    parser.add_argument('--gan_model', default='pggan_churchoutdoor',
                        help='The name of model used.', type=str)

    # Video Settings
    parser.add_argument('--video', type=bool, default=True, help='Save video. False for no video.')
    parser.add_argument('--fps', type=int, default=24, help='Frame rate of the created video.')
    args, other_args = parser.parse_known_args()

    ### RUN
    main(args)
