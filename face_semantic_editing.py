import os
import argparse
import torch
import cv2

from utils.manipulate import convert_array_to_images, get_interpolated_wp, get_boundary, BOUNDARY_DIR
from utils.file_utils import image_files, load_as_tensor
from utils.image_precossing import _sigmoid_to_tanh, _add_batch_one
from derivable_models.derivable_generator import get_derivable_generator
from inversion.inversion_methods import get_inversion
from inversion.losses import get_loss


def main(args):
    generator = get_derivable_generator(args.gan_model, args.inversion_type, args)
    loss = get_loss(args.loss_type, args)
    loss.cuda()
    generator.cuda()
    inversion = get_inversion(args.optimization, args)

    os.makedirs(args.outputs, exist_ok=True)
    save_manipulate_dir = os.path.join(args.outputs, args.attribute_name)
    images_list = image_files(args.target_images)

    for i, images in enumerate(images_list):
        print('%d: Processing images ' % (i + 1), end='')
        image_name = os.path.split(images)[1]
        print(image_name)

        image_name_list = []
        image_tensor_list = []
        image_name_list.append(os.path.split(images)[1])
        image_tensor_list.append(_add_batch_one(load_as_tensor(images)))
        y_gt = _sigmoid_to_tanh(torch.cat(image_tensor_list, dim=0)).cuda()
        # Invert
        latent_estimates, history = inversion.invert(generator, y_gt, loss, batch_size=1)

        image_manipulate_dir = os.path.join(save_manipulate_dir, image_name[:-4])
        os.makedirs(image_manipulate_dir, exist_ok=True)

        wp = latent_estimates[0].cpu().detach().numpy()
        mask = latent_estimates[1].cpu().detach().numpy()

        # Visualize results with given w+ latent vector.
        if args.original:
            print('Save inversion.')
            image = generator(latent_estimates)
            image_cv2 = convert_array_to_images(image.detach().cpu().numpy())
            cv2.imwrite(os.path.join(image_manipulate_dir, 'original_inversion.png'), image_cv2[0][:, :, ::-1])


        boundary, bias = get_boundary(os.path.join(BOUNDARY_DIR, 'pggan_celebahq_%s_boundary.npy' % args.attribute_name))
        wp_list = get_interpolated_wp(wp, boundary, max_step=args.max_step, num_frames=args.fps * args.duration)

        # Create video for attribute manipulation with given w+ latent_vector.
        if args.video:
            print('Create attribute manipulation video.')
            video = cv2.VideoWriter(filename=os.path.join(image_manipulate_dir, '%s_manipulate.avi' % args.attribute_name),
                                    fourcc=cv2.VideoWriter_fourcc(*'MJPG'),
                                    fps=args.fps,
                                    frameSize=(1024, 1024))
            print('Save frames.')
            for i, sample in enumerate(wp_list):
                image = generator(
                    [torch.from_numpy(sample).view((1,) + sample.shape).cuda(), torch.from_numpy(mask).cuda()])
                image_cv2 = convert_array_to_images(image.detach().cpu().numpy())[0][:, :, ::-1]
                cv2.imwrite(os.path.join(image_manipulate_dir, '%d.png' % i), image_cv2)
                video.write(image_cv2)
            video.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Find latent representation of reference images using perceptual loss')
    # Image Path and Saving Path
    parser.add_argument('-i', '--target_images',
                        default='./examples/manipulation',
                        help='Target image to invert.')
    parser.add_argument('-o', '--outputs',
                        default='./manipulate_output',
                        help='Path to save results.')
    parser.add_argument('--attribute_name',
                        default='age',
                        help='Attribute to edit.')
    """
    Attribute List:
    expression
    pose
    age
    gender
    """
    # Parameters for Multi-Code GAN Inversion
    parser.add_argument('--inversion_type', default='PGGAN-Multi-Z',
                        help='Inversion type, PGGAN-Multi-Z for Multi-Code-GAN prior.')  # generator_type
    parser.add_argument('--composing_layer', default=6,
                        help='Composing layer in multi-code gan inversion methods.', type=int)  # blending_layer
    parser.add_argument('--z_number', default=30,
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
    parser.add_argument('--lr', default=1., help='Learning rate.', type=float)
    parser.add_argument('--iterations', default=2000,
                        help='Number of optimization steps.', type=int)
    # Generator Setting
    parser.add_argument('--gan_model', default='pggan_celebahq',
                        help='The name of model used.', type=str)
    # Video Settings
    parser.add_argument('--video', type=bool, default=True,
                        help='The length of the video (seconds). 0 indicates no videos.')
    parser.add_argument('--max_step', type=float, default=4.2,
                        help='Maximum step for attribute manipulation.')
    parser.add_argument('--fps', type=int, default=24,
                        help='Frame rate of the created video.')
    parser.add_argument('--duration', type=int, default=10,
                        help='Duration of created video, taking second as unit.')
    parser.add_argument('--original', action='store_true', default=True,
                        help='Whether to evaluate original inversion.')

    args, other_args = parser.parse_known_args()

    ### RUN
    main(args)
