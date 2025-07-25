import torch
import torch.nn.functional as F
from torch.autograd import Variable

from .u2net import U2NET


def load_mask_net(model_dir):
    net = U2NET(3, 1)
    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_dir, weights_only=True))
        net.cuda()
    net.eval()
    return net


def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d - mi) / (ma - mi)
    return dn


def u2net_masks(images, net):
    images_transformed = F.interpolate(
        images, size=(320, 320), mode="bilinear", align_corners=False
    )
    inputs_test = images_transformed.type(torch.FloatTensor)
    if torch.cuda.is_available():
        inputs_test = Variable(inputs_test.cuda())
    else:
        inputs_test = Variable(inputs_test)

    d1, d2, d3, d4, d5, d6, d7 = net(inputs_test)

    # normalization
    pred = d1[:, 0, :, :]
    pred = normPRED(pred)

    del d1, d2, d3, d4, d5, d6, d7

    ds_pred = pred.detach().cpu().unsqueeze(1)
    us_pred = F.interpolate(
        ds_pred, size=(512, 512), mode="bilinear", align_corners=False
    )
    return us_pred
