import argparse
import os
from util import util
import torch


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--dataroot', required=False, default='inkwash_dataset/horse', help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        self.parser.add_argument('--image_encoder_path', required=False, default='pretrained_models/vgg.pth',
                                 help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        self.parser.add_argument('--batchSize', type=int, required=False, default=4, help='input batch size')
        self.parser.add_argument('--loadSize', type=int, required=False, default=286, help='scale images to this size')
        self.parser.add_argument('--fineSize', type=int, required=False, default=256, help='then crop to this size')
        self.parser.add_argument('--input_nc', type=int, required=False, default=3, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, required=False, default=3, help='# of output image channels')
        self.parser.add_argument('--ngf', type=int, required=False, default=64, help='# of gen filters in first conv layer')
        self.parser.add_argument('--ndf', type=int, required=False, default=64, help='# of discrim filters in first conv layer')
        self.parser.add_argument('--which_model_netD', type=str, required=False, default='basic', help='selects model to use for netD')
        self.parser.add_argument('--which_model_netG', type=str, required=False, default='resnet_9blocks', help='selects model to use for netG,e.g.self_attention,resnet_9blocks')
        self.parser.add_argument('--n_layers_D', type=int, required=False, default=3, help='only used if which_model_netD==n_layers')
        self.parser.add_argument('--gpu_ids', type=str, required=False, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--name', type=str, required=False, default='SWAN', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--dataset_mode', type=str, required=False, default='unaligned', help='chooses how datasets are loaded. [unaligned | aligned | single]')
        self.parser.add_argument('--model', type=str, required=False, default='cycle_gan',
                                 help='chooses which model to use. cycle_gan, pix2pix, test')
        self.parser.add_argument('--which_direction', type=str, required=False, default='AtoB', help='AtoB or BtoA')
        self.parser.add_argument('--nThreads', default=2, type=int, required=False, help='# threads for loading data')
        self.parser.add_argument('--checkpoints_dir', type=str, required=False, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--norm', type=str, required=False, default='instance', help='instance normalization or batch normalization')
        self.parser.add_argument('--serial_batches', action='store_true', required=False, help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.parser.add_argument('--display_winsize', type=int, required=False, default=256,  help='display window size')
        self.parser.add_argument('--display_id', type=int, required=False, default=1, help='window id of the web display')
        self.parser.add_argument('--display_port', type=int, required=False, default=8097, help='visdom port of the web display')
        self.parser.add_argument('--no_dropout', action='store_true', required=False, help='no dropout for the generator')
        self.parser.add_argument('--max_dataset_size', type=int, required=False, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        self.parser.add_argument('--resize_or_crop', type=str, required=False, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        self.parser.add_argument('--no_flip', action='store_true', required=False, help='if specified, do not flip the images for data augmentation')
        self.parser.add_argument('--init_type', type=str, required=False, default='normal', help='network initialization [normal|xavier|kaiming|orthogonal]')
        self.parser.add_argument('--stage', type=str, required=False, default='first', help='which stage to train? first or second')
        self.parser.add_argument('--skip_connection_3', default='true',
                            help='if specified, add skip connection on ReLU-3')
        self.parser.add_argument('--shallow_layer', default='true',
                            help='if specified, also use features of shallow layers')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        util.mkdirs(expr_dir)
        if self.opt.stage == 'first':
            file_name = os.path.join(expr_dir, 'opt.txt')
        elif self.opt.stage == 'second':
            file_name = os.path.join(expr_dir, 'opt_few_shot_1.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt
