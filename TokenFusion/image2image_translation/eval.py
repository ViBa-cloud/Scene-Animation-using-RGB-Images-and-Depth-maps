import os, argparse
import numpy as np
import torch
from tqdm import tqdm
from torchvision import transforms
from models.model_cfg import gen_b2, dis_b2
import cfg
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=1,
                    help='train batch size')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--input-size', type=int, default=256,
                    help='input size')
parser.add_argument('--resize-scale', type=int, default=286,
                    help='resize scale (0 is false)')
parser.add_argument('--crop-size', type=int, default=256,
                    help='crop size (0 is false)')
parser.add_argument('--fliplr', type=bool, default=True,
                    help='random fliplr True of False')
parser.add_argument('--num-epochs', type=int, default=300,
                    help='number of train epochs')
parser.add_argument('--val-every', type=int, default=5,
                    help='how often to validate current architecture')
parser.add_argument('--lrG', type=float, default=0.0002,
                    help='learning rate for generator, default=0.0002')
parser.add_argument('--lrD', type=float, default=0.0002,
                    help='learning rate for discriminator, default=0.0002')
parser.add_argument('--gama', type=float, default=100,
                    help='gama for L1 loss')
parser.add_argument('--beta1', type=float, default=0.5,
                    help='beta1 for Adam optimizer')
parser.add_argument('--beta2', type=float, default=0.999,
                    help='beta2 for Adam optimizer')
parser.add_argument('--print-loss', action='store_true', default=False,
                    help='whether print losses during training')
parser.add_argument('--gpu', type=int, nargs='+', default=[0],
                        help='select gpu.')
parser.add_argument('-c', '--ckpt', default='model', type=str, metavar='PATH',
                    help='path to save checkpoint (default: model)')
parser.add_argument('-i', '--img-types', default=[0, 6, 4], type=int, nargs='+', 
                    help='image types, last image is target, others are inputs')
parser.add_argument('--exchange', type=int, default=1,
                    help='whether use feature exchange')
parser.add_argument('-l', '--lamda', type=float, default=1e-3,
                    help='lamda for L1 norm on BN scales.')
parser.add_argument('-t', '--insnorm-threshold', type=float, default=1e-2,
                    help='threshold for slimming BNs')
parser.add_argument('--enc', default=[0], type=int, nargs='+')
parser.add_argument('--dec', default=[0], type=int, nargs='+')
params = parser.parse_args()

# Directories for loading data and saving results
#data_dir = '/home/suryasin/TokenFusion/image2image_translation/data/scenimefy_data'  # 'Modify data path'
data_dir = '/home/suryasin/TokenFusion/image2image_translation/data/taskonomy-sample-model-1'
# data_dir = '/home1/wyk/data/taskonomy-sample-model-1'
model_dir = os.path.join('ckpt', params.ckpt)
save_dir = os.path.join(model_dir, 'results')
save_dir_best = os.path.join(save_dir, 'best')
os.makedirs(save_dir_best, exist_ok=True)
os.makedirs(os.path.join(model_dir, 'insnorm_params'), exist_ok=True)
os.system('cp -r *py models utils data %s' % model_dir)
cfg.logger = open(os.path.join(model_dir, 'log.txt'), 'w+')
print_log(params)

#train_file = './data/scenimefy_data/train_domain.txt'
#val_file = './data/scenimefy_data/val_domain.txt'
train_file = './data/train_domain.txt'
val_file = './data/val_domain.txt'
domain_dicts = {0: 'rgb', 1: 'normal', 2: 'reshading', 3: 'depth_euclidean', 4: 'depth_zbuffer', 
                5: 'principal_curvature', 6: 'edge_occlusion', 7: 'edge_texture',
                8: 'segment_unsup2d', 9: 'segment_unsup25d'}
params.img_types = [domain_dicts[img_type] for img_type in params.img_types]
print_log('\n' + ', '.join(params.img_types[:-1]) + ' -> ' + params.img_types[-1])
num_parallel = len(params.img_types) - 1

cfg.num_parallel = num_parallel
cfg.use_exchange = params.exchange == 1
cfg.insnorm_threshold = params.insnorm_threshold
cfg.enc, cfg.dec = params.enc, params.dec

# Data pre-processing
transform = transforms.Compose([transforms.Resize(params.input_size),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

# Train data
train_data = DatasetFromFolder(data_dir, train_file, params.img_types, transform=transform,
                               resize_scale=params.resize_scale, crop_size=params.crop_size,
                               fliplr=params.fliplr)
train_data_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=params.batch_size,
                                                shuffle=True, drop_last=False)

# Test data
test_data = DatasetFromFolder(data_dir, val_file, params.img_types, transform=transform)
test_data_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=params.batch_size,
                                               shuffle=False, drop_last=False)
# test_input, test_target = test_data_loader.__iter__().__next__()

# Models
torch.cuda.set_device(params.gpu[0])
# G = Generator(3, params.ngf, 3)
G = gen_b2()
# D = Discriminator(6, params.ndf, 1)
D = dis_b2(img_size=256, patch_size=4)
G.cuda()
G = torch.nn.DataParallel(G, params.gpu)
D.cuda()
D = torch.nn.DataParallel(D, params.gpu)
state_dict = torch.load('./checkpoint-gen-399.pkl')
G.load_state_dict(state_dict, strict=True)

BCE_loss = torch.nn.BCELoss().cuda()
L2_loss = torch.nn.MSELoss().cuda()
L1_loss = torch.nn.L1Loss().cuda()


def evaluate(G, epoch, training):
    num_parallel_ = 1 if num_parallel == 1 else num_parallel + 1
    l1_losses = init_lists(num_parallel_)
    l2_losses = init_lists(num_parallel_)
    fids = init_lists(num_parallel_)
    kids = init_lists(num_parallel_)
    for i, (test_inputs, test_target) in tqdm(enumerate(test_data_loader), miniters=25, total=len(test_data_loader)):
    # for i, (test_inputs, test_target) in enumerate(test_data_loader):
        # Show result for test image
        test_inputs_cuda = [test_input.cuda() for test_input in test_inputs]
        #print(test_input_cuda)
        gen_images, alpha_soft, _ = G(test_inputs_cuda)
        test_target_cuda = test_target.cuda()
        for l, gen_image in enumerate(gen_images):
            if l < num_parallel or num_parallel > 1:
                l1_losses[l].append(L1_loss(gen_image, test_target_cuda).item())
                l2_losses[l].append(L2_loss(gen_image, test_target_cuda).item())
                gen_image = gen_image.cpu().data
                save_dir_ = os.path.join(save_dir, 'fake%d' % l)
                plot_test_result_single(gen_image, i, save_dir=save_dir_)
                if l < num_parallel:
                    save_dir_ = os.path.join(save_dir, 'input%d' % l)
                    if not os.path.exists(os.path.join(save_dir_, '%03d.png' % i)):
                        plot_test_result_single(test_inputs[l], i, save_dir=save_dir_)
        save_dir_ = os.path.join(save_dir, 'real')
        if not os.path.exists(os.path.join(save_dir_, '%03d.png' % i)):
            plot_test_result_single(test_target, i, save_dir=save_dir_)
        # break
        
    for l in range(num_parallel_):
        paths = [os.path.join(save_dir, 'fake%d' % l), os.path.join(save_dir, 'real')]
        fid, kid = calculate_given_paths(paths, batch_size=50, cuda=True, dims=2048)
        fids[l], kids[l] = fid, kid

    l1_avg_losses = [torch.mean(torch.FloatTensor(l1_losses_)) for l1_losses_ in l1_losses]
    l2_avg_losses = [torch.mean(torch.FloatTensor(l2_losses_)) for l2_losses_ in l2_losses]
    return l1_avg_losses, l2_avg_losses, fids, kids


l1_avg_losses, l2_avg_losses, fids, kids = evaluate(G, epoch=0, training=False)
for l in range(len(l1_avg_losses)):
    l1_avg_loss, rl2_avg_loss = l1_avg_losses[l], l2_avg_losses[l]** 0.5
    fid, kid = fids[l], kids[l]
    if l < num_parallel:
        img_type_str = '(%s)' % params.img_types[l][:10]
    else:
        img_type_str = '(ens)'
    print_log('Epoch %3d %-15s   l1_avg_loss: %.5f   rl2_avg_loss: %.5f   fid: %.3f   kid: %.3f' % \
        (0, img_type_str, l1_avg_loss, rl2_avg_loss, fid, kid))

cfg.logger.close()
