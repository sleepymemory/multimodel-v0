import torch.nn as nn
import torch
from torch.nn import functional as F
from torch.distributions import Normal


def product_of_experts(m_vect, v_vect):
    T_vect = 1.0 / v_vect

    mu = (m_vect * T_vect).sum(2) * (1 / T_vect.sum(2))
    var = 1 / T_vect.sum(2)

    return mu, var


def gaussian_parameters(h, dim=-1):
    m, h = torch.split(h, h.size(dim) // 2, dim=dim)
    v = F.softplus(h) + 1e-8
    return m, v


def sample_gaussian(m, v, device):
    epsilon = Normal(0, 1).sample(m.size())
    z = m + torch.sqrt(v) * epsilon.to(device)

    return z


def conv2d(in_channels, out_channels, kernel_size=3, stride=1, dilation=1, bias=True):
    """`same` convolution with LeakyReLU, i.e. output shape equals input shape.
  Args:
    in_planes (int): The number of input feature maps.
    out_planes (int): The number of output feature maps.
    kernel_size (int): The filter size.
    dilation (int): The filter dilation factor.
    stride (int): The filter stride.
  """
    # compute new filter size after dilation
    # and necessary padding for `same` output size
    dilated_kernel_size = (kernel_size - 1) * (dilation - 1) + kernel_size
    same_padding = (dilated_kernel_size - 1) // 2

    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=same_padding,
            dilation=dilation,
            bias=bias,
        ),
        nn.LeakyReLU(0.1, inplace=True),
    )


def duplicate(x, rep):
    return x.expand(rep, *x.shape).reshape(-1, *x.shape[1:])


# In[65]:


# 非图片decoder
class Non_Img_Decoder(nn.Module):
    def __init__(self, z_dim, out_dim, initailize_weights=True):
        """
        Decodes the EE Delta
        """
        super().__init__()

        self.non_img_decoder = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(32, out_dim),
        )

        if initailize_weights:
            init_weights(self.modules())

    def forward(self, x):
        return self.non_img_decoder(x)


class Flatten(nn.Module):
    """Flattens convolutional feature maps for fc layers.
  """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.reshape(x.size(0), -1)


class CausalConv1D(nn.Conv1d):
    """A causal 1D convolution.
  """

    def __init__(
            self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True
    ):
        self.__padding = (kernel_size - 1) * dilation

        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.__padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        res = super().forward(x)
        if self.__padding != 0:
            return res[:, :, : -self.__padding]
        return res


def init_weights(modules):
    """
    Weight initialization from original SensorFusion Code
    """
    for m in modules:
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


class ImageEncoder(nn.Module):
    def __init__(self, z_dim, initailize_weights=True):
        """
        Image encoder taken from Making Sense of Vision and Touch
        """
        super().__init__()
        self.z_dim = z_dim

        self.img_conv1 = conv2d(3, 16, kernel_size=3, stride=2)
        self.img_conv2 = conv2d(16, 32, kernel_size=3, stride=2)
        self.img_conv3 = conv2d(32, 64, kernel_size=3, stride=2)
        self.img_conv4 = conv2d(64, 64, kernel_size=3, stride=2)
        self.img_conv5 = conv2d(64, 128, kernel_size=3, stride=2)
        self.img_conv6 = conv2d(128, 128, kernel_size=3, stride=2)
        self.img_conv7 = conv2d(128, self.z_dim, stride=2)
        self.img_encoder = nn.Linear(4 * self.z_dim, 2 * self.z_dim)
        self.flatten = Flatten()

        if initailize_weights:
            init_weights(self.modules())

    def forward(self, image):
        # image encoding layers
        out_img_conv1 = self.img_conv1(image)
        out_img_conv2 = self.img_conv2(out_img_conv1)
        out_img_conv3 = self.img_conv3(out_img_conv2)
        out_img_conv4 = self.img_conv4(out_img_conv3)
        out_img_conv5 = self.img_conv5(out_img_conv4)
        out_img_conv6 = self.img_conv6(out_img_conv5)
        out_img_conv7 = self.img_conv7(out_img_conv6)

        img_out_convs = (
            out_img_conv1,
            out_img_conv2,
            out_img_conv3,
            out_img_conv4,
            out_img_conv5,
            out_img_conv6,
            out_img_conv7,
        )

        # image embedding parameters
        flattened = self.flatten(out_img_conv7)
        img_out = self.img_encoder(flattened).unsqueeze(2)

        return img_out, img_out_convs


class CloudEncoder(nn.Module):
    def __init__(self, z_dim, initailize_weights=True):
        """
        Image encoder taken from Making Sense of Vision and Touch
        """
        super().__init__()
        self.z_dim = z_dim

        self.cloud_conv1 = conv2d(3, 16, kernel_size=3, stride=2)
        self.cloud_conv2 = conv2d(16, 32, kernel_size=3, stride=2)
        self.cloud_conv3 = conv2d(32, 64, kernel_size=3, stride=2)
        self.cloud_conv4 = conv2d(64, 64, kernel_size=3, stride=2)
        self.cloud_conv5 = conv2d(64, 128, kernel_size=3, stride=2)
        self.cloud_conv6 = conv2d(128, 128, kernel_size=3, stride=2)
        self.cloud_conv7 = conv2d(128, 128, kernel_size=3, stride=2)
        self.cloud_conv8 = conv2d(128, 128, kernel_size=3, stride=2)
        self.cloud_conv9 = conv2d(128, self.z_dim, stride=2)
        self.cloud_encoder = nn.Linear(4 * self.z_dim, 2 * self.z_dim)
        self.flatten = Flatten()

        if initailize_weights:
            init_weights(self.modules())

    def forward(self, cloud):
        # image encoding layers
        out_cloud_conv1 = self.cloud_conv1(cloud)
        out_cloud_conv2 = self.cloud_conv2(out_cloud_conv1)
        out_cloud_conv3 = self.cloud_conv3(out_cloud_conv2)
        out_cloud_conv4 = self.cloud_conv4(out_cloud_conv3)
        out_cloud_conv5 = self.cloud_conv5(out_cloud_conv4)
        out_cloud_conv6 = self.cloud_conv6(out_cloud_conv5)
        out_cloud_conv7 = self.cloud_conv7(out_cloud_conv6)
        out_cloud_conv8 = self.cloud_conv8(out_cloud_conv7)
        out_cloud_conv9 = self.cloud_conv9(out_cloud_conv8)

        cloud_out_convs = (
            out_cloud_conv1,
            out_cloud_conv2,
            out_cloud_conv3,
            out_cloud_conv4,
            out_cloud_conv5,
            out_cloud_conv6,
            out_cloud_conv7,
            out_cloud_conv8,
            out_cloud_conv9,
        )

        # image embedding parameters
        flattened = self.flatten(out_cloud_conv9)
        cloud_out = self.cloud_encoder(flattened).unsqueeze(2)

        return cloud_out, cloud_out_convs


class F_Encoder(nn.Module):
    def __init__(self, z_dim, initailize_weights=True):
        """
        Image encoder taken from selfsupervised code
        """
        super().__init__()
        self.z_dim = z_dim

        self.force_encoder = nn.Sequential(
            nn.Linear(3, 32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(32, 64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, 2 * self.z_dim),
            nn.LeakyReLU(0.1, inplace=True),
        )

        if initailize_weights:
            init_weights(self.modules())

    def forward(self, force):
        return self.force_encoder(force).unsqueeze(2)


class P_Encoder(nn.Module):
    def __init__(self, z_dim, initailize_weights=True):
        """
        Image encoder taken from selfsupervised code
        """
        super().__init__()
        self.z_dim = z_dim

        self.pose_encoder = nn.Sequential(
            nn.Linear(3, 32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(32, 64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, 2 * self.z_dim),
            nn.LeakyReLU(0.1, inplace=True),
        )

        if initailize_weights:
            init_weights(self.modules())

    def forward(self, pose):
        return self.pose_encoder(pose).unsqueeze(2)


class D_Encoder(nn.Module):
    def __init__(self, z_dim, initailize_weights=True):
        """
        Image encoder taken from selfsupervised code
        """
        super().__init__()
        self.z_dim = z_dim

        self.trans_encoder = nn.Sequential(
            nn.Linear(4, 32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(32, 64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, 2 * self.z_dim),
            nn.LeakyReLU(0.1, inplace=True),
        )

        if initailize_weights:
            init_weights(self.modules())

    def forward(self, trans):
        return self.trans_encoder(trans).unsqueeze(2)


# In[66]:


class SensorFusion(nn.Module):
    def __init__(self, device=None, z_dim=128, action_dim=4):
        super().__init__()
        self.z_dim = z_dim
        self.device = device
        # zero centered, 1 std normal distribution
        self.z_prior_m = torch.nn.Parameter(torch.zeros(1, self.z_dim), requires_grad=False)
        self.z_prior_v = torch.nn.Parameter(torch.ones(1, self.z_dim), requires_grad=False)
        self.z_prior = (self.z_prior_m, self.z_prior_v)
        # encoder
        self.img_encoder = ImageEncoder(self.z_dim)
        self.us_img_encoder = ImageEncoder(self.z_dim)
        # self.cloud_encoder = CloudEncoder(self.z_dim)
        self.F_encoder = F_Encoder(self.z_dim)
        self.P_encoder = P_Encoder(self.z_dim)
        self.D_encoder = D_Encoder(self.z_dim)

        self.pair_fc = nn.Sequential(nn.Linear(self.z_dim, 1))

        self.st_fusion_fc1 = nn.Sequential(nn.Linear(self.z_dim, 128), nn.LeakyReLU(0.1, inplace=True))
        self.st_fusion_fc2 = nn.Sequential(nn.Linear(128, self.z_dim), nn.LeakyReLU(0.1, inplace=True))

        # decoder
        self.F_decoder = Non_Img_Decoder(self.z_dim, 3)
        self.P_decoder = Non_Img_Decoder(self.z_dim, 3)
        self.D_decoder = Non_Img_Decoder(self.z_dim, 4)

        # decoder_2
        self.F_20_decoder = Non_Img_Decoder(self.z_dim, 3)
        self.P_20_decoder = Non_Img_Decoder(self.z_dim, 3)
        self.D_20_decoder = Non_Img_Decoder(self.z_dim, 4)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, image_in, us_image_in, F_in, P_in, D_in):
        # batch size
        batch_dim = image_in.size()[0]
        # out
        img_out, img_out_convs = self.img_encoder(image_in)
        us_image_out, us_image_out_convs = self.us_img_encoder(us_image_in)
        # cloud_out, cloud_out_convs = self.cloud_encoder(cloud_in)
        F_out = self.F_encoder(F_in)
        P_out = self.P_encoder(P_in)
        D_out = self.D_encoder(D_in)

        # Encoder priors
        mu_prior, var_prior = self.z_prior
        # Duplicate prior parameters for each data point in the batch
        mu_prior_resized = duplicate(mu_prior, batch_dim).unsqueeze(2)
        var_prior_resized = duplicate(var_prior, batch_dim).unsqueeze(2)

        # Modality Mean and Variances
        mu_z_img, var_z_img = gaussian_parameters(img_out, dim=1)
        mu_z_us_img, var_z_us_img = gaussian_parameters(us_image_out, dim=1)
        mu_z_F, var_z_F = gaussian_parameters(F_out, dim=1)
        mu_z_P, var_z_P = gaussian_parameters(P_out, dim=1)
        mu_z_D, var_z_D = gaussian_parameters(D_out, dim=1)

        # Tile distribution parameters using concatonation
        m_vect = torch.cat([mu_z_img, mu_z_us_img, mu_z_F, mu_z_P, mu_z_D, mu_prior_resized], dim=2)
        var_vect = torch.cat([var_z_img, var_z_us_img, var_z_F, var_z_P, var_z_D, var_prior_resized], dim=2)
        # Fuse modalities mean / variances using product of experts
        mu_z, var_z = product_of_experts(m_vect, var_vect)
        # Sample Gaussian to get latent
        z = sample_gaussian(mu_z, var_z, self.device)
        pair_out = self.pair_fc(z)
        finalout1 = self.st_fusion_fc1(z)
        finalout = self.st_fusion_fc2(finalout1)

        F_final = self.F_decoder(finalout)
        P_final = self.P_decoder(finalout)
        D_final = self.D_decoder(finalout)

        F_20_final = self.F_20_decoder(finalout)
        P_20_final = self.P_20_decoder(finalout)
        D_20_final = self.D_20_decoder(finalout)

        return F_final, P_final, D_final, F_20_final, P_20_final, D_20_final
