from torch.autograd import Variable
from collections import OrderedDict
import util.util as util
from .base_model import BaseModel
from . import multi_in_networks as networks
import numpy as np


class TestModel(BaseModel):
    def name(self):
        return 'TestModel'

    def initialize(self, opt):
        assert(not opt.isTrain)
        BaseModel.initialize(self, opt)
        self.input_0 = self.Tensor(opt.batchSize, opt.input_nc,
                                   opt.fineSize, opt.fineSize)
        self.input_1 = self.Tensor(opt.batchSize, opt.input_nc,
                                   opt.fineSize, opt.fineSize)
        self.input_2 = self.Tensor(opt.batchSize, opt.input_nc,
                                   opt.fineSize, opt.fineSize)

        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt.norm, not opt.no_dropout, self.gpu_ids, False,
                                      opt.learn_residual)
        which_epoch = opt.which_epoch
        self.netG.cuda()
        self.load_network(self.netG, 'G', which_epoch)
        for param in self.netG.parameters():
                param.requires_grad=False
        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        print('-----------------------------------------------')

    def set_input(self, input):
        # we need to use single_dataset mode
        input_0 = input['image0']
        input_1 = input['image1']
        input_2 = input['image2']
        self.input_0.resize_(input_0.size()).copy_(input_0)
        self.input_1.resize_(input_1.size()).copy_(input_1)
        self.input_2.resize_(input_2.size()).copy_(input_2)
        self.inputB = input['label']

    def test(self):
        self.real_A = Variable(self.input_1)
        self.real_A_pre = Variable(self.input_0)
        self.real_A_after = Variable(self.input_2)
        self.fake_B = self.netG.forward(self.real_A_pre, self.real_A, self.real_A_after)


    def get_current_visuals(self):
        real_B = self.inputB[0,:,:,:].cpu().numpy()
        real_B = (np.transpose(real_B, (1, 2, 0)) + 1) / 2.0 * 255.0
        fake_B = util.tensor2im(self.fake_B.data)
        return OrderedDict([('real_B', real_B.astype(np.uint8)), ('fake_B', fake_B)])
