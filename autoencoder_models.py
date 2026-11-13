"""The notebook's three models and their image helpers, used by the Streamlit app.

The model classes match autoencoder.ipynb exactly, so the notebook's saved autoencoders.pt loads into them.
Images here are float tensors of shape N x 3 x 32 x 32 with values from 0 to 1.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from skimage.color import lab2rgb, rgb2lab
from skimage.metrics import structural_similarity

SIZE = 32


class ConvAutoencoder(nn.Module):
    # Feature extraction: a convolutional, undercomplete autoencoder with a label signal.
    def __init__(self, latent):
        super().__init__()
        def down(cin, cout):   # halves the image size, then adds a second layer at the new size
            return nn.Sequential(nn.Conv2d(cin, cout, 3, stride=2, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
                                 nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU())

        def up(cin, cout):     # doubles the image size
            return nn.Sequential(nn.ConvTranspose2d(cin, cout, 4, stride=2, padding=1), nn.BatchNorm2d(cout), nn.ReLU())

        self.encoder = nn.Sequential(down(3, 64), down(64, 128), down(128, 256),   # 32 -> 16 -> 8 -> 4
                                     nn.Flatten(), nn.Linear(256 * 4 * 4, latent))
        self.decoder = nn.Sequential(nn.Linear(latent, 256 * 4 * 4), nn.Unflatten(1, (256, 4, 4)),
                                     up(256, 128), up(128, 64),                     # 4 -> 8 -> 16
                                     nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1), nn.Sigmoid())   # -> 32
        self.classifier = nn.Linear(latent, 10)   # predicts the class from the 128 numbers

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        return self.decoder(self.encoder(x))


class UNet(nn.Module):
    # Colorization and enhancement: a convolutional encoder-decoder with skip connections.
    def __init__(self, cin, cout, ch=48):
        super().__init__()
        def block(a, b):
            return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(),
                                 nn.Conv2d(b, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU())

        self.d1, self.d2, self.d3 = block(cin, ch), block(ch, ch * 2), block(ch * 2, ch * 4)   # 32, 16, 8
        self.mid = block(ch * 4, ch * 8)                                                     # 4
        self.up3, self.u3 = nn.ConvTranspose2d(ch * 8, ch * 4, 2, stride=2), block(ch * 8, ch * 4)
        self.up2, self.u2 = nn.ConvTranspose2d(ch * 4, ch * 2, 2, stride=2), block(ch * 4, ch * 2)
        self.up1, self.u1 = nn.ConvTranspose2d(ch * 2, ch, 2, stride=2), block(ch * 2, ch)
        self.out = nn.Conv2d(ch, cout, 1)

    def forward(self, x):
        h1 = self.d1(x)
        h2 = self.d2(F.max_pool2d(h1, 2))
        h3 = self.d3(F.max_pool2d(h2, 2))
        h = self.mid(F.max_pool2d(h3, 2))
        h = self.u3(torch.cat([self.up3(h), h3], dim=1))   # the shortcuts: join each size with its match
        h = self.u2(torch.cat([self.up2(h), h2], dim=1))
        h = self.u1(torch.cat([self.up1(h), h1], dim=1))
        return self.out(h)


class Autoencoders:
    """Loads the three trained models from the notebook's autoencoders.pt and runs them."""

    def __init__(self, path, device="cpu"):
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        self.device = device
        self.class_names = list(checkpoint["class_names"])
        self.noise_level = float(checkpoint["noise_level"])
        self.autoencoder = ConvAutoencoder(checkpoint["latent_size"])
        self.colorizer, self.enhancer = UNet(1, 2), UNet(3, 3)
        for name in ("autoencoder", "colorizer", "enhancer"):
            model = getattr(self, name)
            model.load_state_dict(checkpoint[name])
            model.to(device).eval()

    @torch.no_grad()
    def raw_features(self, x):
        # The encoder's 128 numbers for each image, as the classifier receives them.
        return self.autoencoder.encode(x.to(self.device)).cpu()

    @torch.no_grad()
    def features(self, x):
        # The 128 numbers for each image, scaled to length 1 so that similarity is the angle between them.
        return F.normalize(self.autoencoder.encode(x.to(self.device)), dim=1).cpu()

    @torch.no_grad()
    def class_probabilities(self, x):
        # Averages the classifier over 10 views of each image: the image and its mirror image, and four shifts of
        # 2 pixels of each. This test-time averaging makes the answer less dependent on exact framing.
        x = x.to(self.device)
        padded = F.pad(x, (2, 2, 2, 2), mode="reflect")
        views = [x, x.flip(3)]
        for dy, dx in ((0, 0), (0, 4), (4, 0), (4, 4)):
            shifted = padded[:, :, dy:dy + SIZE, dx:dx + SIZE]
            views += [shifted, shifted.flip(3)]
        probabilities = [F.softmax(self.autoencoder.classifier(self.autoencoder.encode(v)), dim=1) for v in views]
        return torch.stack(probabilities).mean(0).cpu()

    @torch.no_grad()
    def colorize(self, x):
        # Keeps the brightness (L) of x and predicts the color (a, b), as in the notebook.
        L = to_lab(x)[:, :1]
        ab = torch.tanh(self.colorizer(L.to(self.device))).cpu()
        return to_rgb(L, ab)

    @torch.no_grad()
    def enhance(self, x):
        x = x.to(self.device)
        return (x + self.enhancer(x)).clamp(0, 1).cpu()


def load_image(file):
    """Opens an image file, crops it to a centered square and resizes it to 32 x 32. Returns a 1 x 3 x 32 x 32 tensor."""
    image = Image.open(file).convert("RGB")
    side = min(image.size)
    left, top = (image.width - side) // 2, (image.height - side) // 2
    image = image.crop((left, top, left + side, top + side))
    if image.size != (SIZE, SIZE):
        image = image.resize((SIZE, SIZE), Image.LANCZOS)
    return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1)[None].float() / 255


def to_lab(x):
    # Brightness L becomes 0 to 1, and the color channels a and b roughly -1 to 1, as in the notebook.
    lab = torch.from_numpy(rgb2lab(x.permute(0, 2, 3, 1).numpy())).permute(0, 3, 1, 2).float()
    return torch.cat([lab[:, :1] / 100, lab[:, 1:] / 110], dim=1)


def to_rgb(L, ab):
    lab = torch.cat([L * 100, ab * 110], dim=1).permute(0, 2, 3, 1).double().numpy()
    return torch.from_numpy(lab2rgb(lab)).permute(0, 3, 1, 2).float()


def grayscale(x):
    # The brightness alone, shown as a gray image.
    return to_rgb(to_lab(x)[:, :1], torch.zeros(len(x), 2, SIZE, SIZE))


def is_grayscale(x, tolerance=2 / 255):
    return bool((x.max(dim=1).values - x.min(dim=1).values).max() <= tolerance)


def color_strength(x):
    # The average distance of a and b from gray, in Lab units, as in the notebook.
    return (to_lab(x)[:, 1:] * 110).pow(2).sum(1).sqrt().mean().item()


def damage(x, blur, noise, seed=0):
    """Blurs x by `blur` pixels (Gaussian), then adds noise of standard deviation `noise`, using the notebook's method."""
    n = len(x)
    offsets = torch.arange(-2, 3).float()
    kernel = torch.exp(-offsets ** 2 / (2 * blur ** 2))
    kernel = (kernel / kernel.sum()).repeat(n * 3, 1)
    y = x.reshape(1, n * 3, SIZE, SIZE)
    y = F.conv2d(F.pad(y, (2, 2, 0, 0), mode="reflect"), kernel.view(n * 3, 1, 1, 5), groups=n * 3)   # across
    y = F.conv2d(F.pad(y, (0, 0, 2, 2), mode="reflect"), kernel.view(n * 3, 1, 5, 1), groups=n * 3)   # down
    generator = torch.Generator().manual_seed(seed)
    return (y.view(n, 3, SIZE, SIZE) + torch.randn(n, 3, SIZE, SIZE, generator=generator) * noise).clamp(0, 1)


def psnr(a, b):
    mse = ((a - b) ** 2).mean().clamp_min(1e-10)
    return (10 * torch.log10(1 / mse)).item()


def ssim(a, b):
    return float(structural_similarity(a[0].permute(1, 2, 0).numpy(), b[0].permute(1, 2, 0).numpy(),
                                       channel_axis=2, data_range=1.0))


def to_pil(x, size=160):
    """Turns a 1 x 3 x 32 x 32 tensor into an enlarged PIL image for display, each pixel drawn as a sharp square."""
    array = (x[0].clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(array).resize((size, size), Image.NEAREST)
