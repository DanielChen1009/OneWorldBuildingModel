import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torchvision.transforms as T


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class AttentionBlock(nn.Module):
    def __init__(self, f_g, f_l, f_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(f_g, f_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(f_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(f_l, f_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(f_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(f_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class AttentionUNET(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(AttentionUNET, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in self.features:
            self.downs.append(ConvBlock(in_channels, feature))
            in_channels = feature

        for feature in reversed(self.features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(AttentionBlock(f_g=feature, f_l=feature, f_int=feature//2))
            self.ups.append(ConvBlock(feature*2, feature))

        self.bottom = ConvBlock(self.features[-1], self.features[-1]*2)
        self.last_conv = nn.Conv2d(self.features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_conn = []

        for down in self.downs:
            x = down(x)
            skip_conn.append(x)
            x = self.pool(x)

        skip_conn = skip_conn[::-1]
        x = self.bottom(x)

        for idx in range(0,  len(self.ups), 3):
            x = self.ups[idx](x)
            atten_block = self.ups[idx + 1](x, skip_conn[idx // 3])

            if x.shape != atten_block.shape:
                x = TF.resize(x, size=atten_block.shape[2:])

            concat_skip = torch.cat((atten_block, x), dim=1)
            x = self.ups[idx + 2](concat_skip)

        return self.last_conv(x)


def test():
    x = torch.randn((1, 3, 160, 160))
    model = AttentionUNET(in_channels=3, out_channels=1)
    preds = model.forward(x)
    # attn = model.attenBlock(x, torch.randn((3, 32, 32, 32)), 32)
    print(preds.shape)
    print(x.shape)
    # print(attn.shape)


if __name__ == "__main__":
    test()