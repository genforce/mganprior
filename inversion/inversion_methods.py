from tqdm import tqdm
import copy

import torch
import torch.nn as nn
import torch.optim as optim


def get_inversion(inversion_type, args):
    if inversion_type == 'GD':
        return GradientDescent(args.iterations, args.lr, optimizer=optim.SGD, args=args)
    elif inversion_type == 'Adam':
        return GradientDescent(args.iterations, args.lr, optimizer=optim.Adam, args=args)


class GradientDescent(object):
    def __init__(self, iterations, lr, optimizer, args):
        self.iterations = iterations
        self.lr = lr
        self.optimizer = optimizer
        self.init_type = args.init_type  # ['Zero', 'Normal']

    def invert(self, generator, gt_image, loss_function, batch_size=1, video=True, *init):
        input_size_list = generator.input_size()
        if len(init) == 0:
            if generator.init is False:
                latent_estimate = []
                for input_size in input_size_list:
                    if self.init_type == 'Zero':
                        latent_estimate.append(torch.zeros((batch_size,) + input_size).cuda())
                    elif self.init_type == 'Normal':
                        latent_estimate.append(torch.randn((batch_size,) + input_size).cuda())
            else:
                latent_estimate = list(generator.init_value(batch_size))
        else:
            assert len(init) == len(input_size_list), 'Please check the number of init value'
            latent_estimate = init

        for latent in latent_estimate:
            latent.requires_grad = True
        optimizer = self.optimizer(latent_estimate, lr=self.lr)

        history = []
        # Opt
        for i in tqdm(range(self.iterations)):
            y_estimate = generator(latent_estimate)
            optimizer.zero_grad()
            loss = loss_function(y_estimate, gt_image)
            loss.backward()
            optimizer.step()
            if video:
                history.append(copy.deepcopy(latent_estimate))
        return latent_estimate, history

