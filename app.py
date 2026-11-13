"""Streamlit app for the three autoencoders: feature extraction, grayscale to color, and image enhancement."""
import os
from pathlib import Path

import streamlit as st
import torch

import autoencoder_models as am

ROOT = Path(__file__).parent
CHECKPOINT = Path(os.environ.get("CHECKPOINT", ROOT / "weights" / "autoencoders.pt"))
SAMPLES = Path(os.environ.get("SAMPLES", ROOT / "test" / "images"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SHOWN = 160   # display size in pixels for the 32 x 32 images

FEATURES, COLOR, ENHANCE = "🔎 Feature extraction", "🎨 Grayscale to color", "✨ Image enhancement"
SAMPLE_SOURCE, UPLOAD_SOURCE = "AI-generated samples", "Upload images"
DESCRIPTIONS = {
    FEATURES: "The encoder turns each image into 128 numbers that describe its content, a feature vector. "
              "They are used to recognize the object and to find the samples with the most similar content.",
    COLOR: "The model receives only the brightness of each image, which is the grayscale image, and predicts its "
           "colors. The brightness of the output always matches the input.",
    ENHANCE: "The model repairs a noisy, blurred copy of each image. It predicts a correction that is added to the "
             "damaged copy.",
}


@st.cache_resource(show_spinner="Loading the models...")
def load_models(path):
    return am.Autoencoders(path, DEVICE)


def sample_names():
    return sorted(p.name for p in SAMPLES.glob("*.png"))


@st.cache_resource(show_spinner=False)
def sample_features(_models, names):
    # The features of every sample image, computed once, for the similar-images search.
    images = torch.cat([am.load_image(SAMPLES / name) for name in names])
    return images, _models.features(images)


def label(name):
    return Path(name).stem.replace("_", " ")


def sample_class(name):
    # The class is the first part of a sample's file name, such as "cat" in cat_1.png.
    return Path(name).stem.rsplit("_", 1)[0]


def show_row(items):
    # items: (image, caption) pairs, shown side by side.
    for column, (image, caption) in zip(st.columns(len(items)), items):
        column.image(am.to_pil(image, SHOWN), caption=caption, width=SHOWN)


def feature_result(image, name):
    show_row([(image, "input image")])
    probabilities = models.class_probabilities(image)[0]
    st.markdown("**Recognized class**")
    for p, c in zip(*probabilities.topk(3)):
        st.progress(float(p), text=f"{models.class_names[c]}: {p:.0%}")
    st.caption("The model always picks one of its 10 classes, even for an image that shows none of them.")
    with st.expander("The 128 numbers extracted from this image"):
        st.code(" ".join(f"{value:6.2f}" for value in models.raw_features(image)[0].tolist()), language=None)
        st.caption("Each number summarizes part of the image's content. They are learned, so no single number has a "
                   "fixed meaning; together they decide the class and the closest samples.")

    names = sample_names()
    recognized = models.class_names[int(probabilities.argmax())]
    candidates = [i for i, n in enumerate(names) if n != name and sample_class(n) == recognized]
    searched = f"the {recognized} samples"
    if not candidates:
        candidates = [i for i, n in enumerate(names) if n != name]
        searched = "all samples, since none is labelled with the recognized class"
    if candidates:
        st.markdown("**Closest samples**")
        sample_images, sample_feats = sample_features(models, tuple(names))
        similarity = sample_feats[candidates] @ models.features(image)[0]
        order = similarity.argsort(descending=True)[:5].tolist()
        show_row([(sample_images[[candidates[k]]], f"{label(names[candidates[k]])} ({similarity[k]:.2f})")
                  for k in order])
        st.caption(f"Searched {searched}, ranked by how similar their 128 numbers are (cosine similarity, "
                   "1 = the same direction).")


def color_result(image):
    gray, colored = am.grayscale(image), models.colorize(image)
    if am.is_grayscale(image):
        show_row([(gray, "grayscale input"), (colored, "colorized")])
        st.caption(f"Color strength of the colorized image: {am.color_strength(colored):.1f}")
    else:
        show_row([(gray, "grayscale input"), (colored, "colorized"), (image, "true colors")])
        st.caption(f"PSNR against the true colors: grayscale {am.psnr(gray, image):.1f} dB, colorized "
                   f"{am.psnr(colored, image):.1f} dB. Color strength: true {am.color_strength(image):.1f}, "
                   f"colorized {am.color_strength(colored):.1f} (lower means more muted).")


def enhance_result(image, settings, seed):
    if settings is None:
        show_row([(image, "input"), (models.enhance(image), "repaired")])
        return
    blur, noise = settings
    damaged = am.damage(image, blur, noise, seed)
    repaired = models.enhance(damaged)
    show_row([(damaged, "damaged copy"), (repaired, "repaired"), (image, "original")])
    st.caption(f"PSNR: damaged {am.psnr(damaged, image):.1f} dB, repaired {am.psnr(repaired, image):.1f} dB. "
               f"SSIM: damaged {am.ssim(damaged, image):.3f}, repaired {am.ssim(repaired, image):.3f} "
               "(higher is closer to the original).")


st.set_page_config(page_title="Autoencoders", page_icon="🧩", layout="wide")
st.title("🧩 Autoencoders")
st.caption("Three encoder-decoder networks trained from random weights on CIFAR-10 (32×32 color images). "
           "Every image is resized to 32×32 pixels, the size the models were trained on, and shown enlarged, "
           "with each pixel drawn as a small square. The input images themselves are never changed.")

has_model = CHECKPOINT.exists()
models = None
if has_model:
    models = load_models(str(CHECKPOINT))

with st.sidebar:
    st.header("ℹ️ About")
    st.caption(f"Device: {torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'CPU'}")
    if models:
        st.caption("Models: loaded")
        st.caption("The models know 10 classes: " + ", ".join(models.class_names))
    else:
        st.caption("Models: not found")
    st.caption("The samples in `test/images/` are AI-generated with Stable Diffusion 1.5, not photographs.")

if not has_model:
    rel = CHECKPOINT.relative_to(ROOT) if CHECKPOINT.is_relative_to(ROOT) else CHECKPOINT
    st.warning(f"Trained models not found at `{rel}`. See the README for how to add them.")
    st.stop()

capability = st.radio("Choose a capability", [FEATURES, COLOR, ENHANCE], horizontal=True)
st.write(DESCRIPTIONS[capability])

enhance_settings, seed = None, 0
if capability == ENHANCE:
    already_damaged = st.toggle("My images are already noisy or blurred",
                                help="Off: the app damages a copy of each image, then repairs it. "
                                     "On: the app repairs each image as it is.")
    if not already_damaged:
        left, middle, right = st.columns(3)
        blur = left.slider("Blur (pixels)", 0.5, 1.5, 1.0, step=0.1, help="The training range was 0.5 to 1.5.")
        noise = middle.slider("Noise", 0.0, models.noise_level, models.noise_level, step=0.01,
                              help=f"The standard deviation of the noise. The training range was 0 to {models.noise_level}.")
        seed = int(right.number_input("Seed", min_value=0, value=0, step=1, help="The same seed gives the same noise."))
        enhance_settings = (blur, noise)

st.divider()
source = st.radio("Images", [SAMPLE_SOURCE, UPLOAD_SOURCE], horizontal=True)
chosen = []   # (image, sample file name or None, title)
if source == SAMPLE_SOURCE:
    names = sample_names()
    if not names:
        st.info("No sample images found in `test/images/`. Choose *Upload images* instead.")
    defaults = [label(n) for n in names if Path(n).stem in ("automobile_1", "cat_1", "dog_1")]
    picked = st.multiselect("AI-generated samples", [label(n) for n in names], default=defaults)
    for n in names:
        if label(n) in picked:
            chosen.append((am.load_image(SAMPLES / n), n, f"{label(n)} (AI-generated sample)"))
else:
    files = st.file_uploader("Your images (PNG, JPG or WebP). Each is cropped to a square and resized to 32×32 pixels.",
                             type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True)
    for file in files or []:
        chosen.append((am.load_image(file), None, file.name))

if not chosen:
    st.info("Choose at least one image.")
for image, name, title in chosen:
    with st.container(border=True):
        st.subheader(title)
        if capability == FEATURES:
            feature_result(image, name)
        elif capability == COLOR:
            color_result(image)
        else:
            enhance_result(image, enhance_settings, seed)
