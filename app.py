# import streamlit as st
# import torch
# import torch.nn as nn
# import torchvision.models as models
# import torchvision.transforms as transforms
# import cv2, os, tempfile
# import numpy as np
# from collections import deque
# from PIL import Image

# # ==== Model định nghĩa ====
# class SEBlock(nn.Module):
#     def __init__(self, channels, reduction=16):
#         super().__init__()
#         self.pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(channels, channels // reduction, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(channels // reduction, channels, bias=False),
#             nn.Sigmoid()
#         )
#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y = self.pool(x).view(b, c)
#         y = self.fc(y).view(b, c, 1, 1)
#         return x * y

# class AttentionPooling(nn.Module):
#     def __init__(self, hidden_dim):
#         super().__init__()
#         self.attn = nn.Linear(hidden_dim, 1)
#     def forward(self, x):
#         w = torch.softmax(self.attn(x), dim=1)
#         return (x * w).sum(dim=1)

# class CNNTransformer(nn.Module):
#     def __init__(self, num_classes, seq_len=30, hidden_dim=256, num_heads=4, num_layers=1, pretrained=True, freeze_until=10):
#         super().__init__()
#         mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None)
#         self.cnn = mobilenet.features
#         for i, child in enumerate(self.cnn.children()):
#             if i < freeze_until:
#                 for p in child.parameters():
#                     p.requires_grad = False
#         self.head = nn.Sequential(
#             nn.Conv2d(1280, 512, 3, 1, 1, bias=False),
#             nn.BatchNorm2d(512),
#             nn.ReLU(inplace=True),
#             SEBlock(512),
#             nn.Dropout(0.3)
#         )
#         self.gap = nn.AdaptiveAvgPool2d(1)
#         self.fc_in = nn.Linear(512, hidden_dim)
#         enc_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads,
#                                                dim_feedforward=hidden_dim*2, dropout=0.3, activation="gelu")
#         self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
#         self.attn_pool = AttentionPooling(hidden_dim)
#         self.fc_out = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(hidden_dim, num_classes)
#         )
#     def forward(self, x):
#         B, T, C, H, W = x.shape
#         x = x.view(B*T, C, H, W)
#         feats = self.cnn(x)
#         feats = self.head(feats)
#         feats = self.gap(feats).view(B, T, -1)
#         feats = self.fc_in(feats)
#         feats = self.transformer(feats)
#         feats = self.attn_pool(feats)
#         return self.fc_out(feats)

# # ==== Load model ====
# @st.cache_resource
# def load_model(path, num_classes, device):
#     model = CNNTransformer(num_classes=num_classes).to(device)
#     model.load_state_dict(torch.load(path, map_location=device))
#     model.eval()
#     return model

# # ==== Predict video và annotate ====
# from moviepy.editor import VideoFileClip
# import imageio

# def predict_video_voted(
#     model, video_path, label_names, device,
#     seq_len=30, step=30, img_size=256,
#     vote="soft", k=5
# ):
#     tfm = transforms.Compose([
#         transforms.ToPILImage(),
#         transforms.Resize((img_size, img_size)),
#         transforms.ToTensor()
#     ])
#     cap = cv2.VideoCapture(video_path)
#     fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
#     width, height = int(cap.get(3)), int(cap.get(4))

#     tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
#     out_path = tmp_out.name
#     writer = imageio.get_writer(out_path, fps=fps)

#     window_buf, preds = [], []
#     prob_deque, label_deque = deque(maxlen=k), deque(maxlen=k)

#     while True:
#         ok, fr = cap.read()
#         if not ok:
#             break
#         window_buf.append(fr)

#         if len(window_buf) == seq_len:
#             rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in window_buf]
#             tens = [tfm(x).unsqueeze(0) for x in rgb]
#             batch = torch.stack(tens, dim=1).to(device)
#             with torch.no_grad():
#                 logits = model(batch)
#                 prob_vec = torch.softmax(logits, dim=1)[0].cpu().numpy()

#             label_deque.append(np.argmax(prob_vec))
#             prob_deque.append(prob_vec)

#             if vote == "hard":
#                 vals, counts = np.unique(label_deque, return_counts=True)
#                 voted_idx = int(vals[np.argmax(counts)])
#             else:
#                 avg_prob = np.mean(np.stack(prob_deque), axis=0)
#                 voted_idx = int(np.argmax(avg_prob))

#             text = f"{label_names[voted_idx]} ({avg_prob[voted_idx]:.2f})"
#             for f in window_buf[:step]:
#                 cv2.putText(f, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
#                             1.2, (0,255,0), 3, cv2.LINE_AA)
#                 writer.append_data(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
#             window_buf = window_buf[step:]

#     cap.release()
#     writer.close()

#     final_out = out_path.replace(".mp4", "_final.mp4")
#     clip = VideoFileClip(out_path)
#     clip.write_videofile(final_out, codec="libx264", audio=False, verbose=False, logger=None)
#     clip.close()
#     os.remove(out_path)

#     return final_out

# # ==== Streamlit App ====
# st.title("Driver Fatigue & Drunk & Cognitive Distraction Detection 🚗")

# uploaded_file = st.file_uploader("Upload a video", type=["mp4","avi","mov","mkv"])
# if uploaded_file is not None:
#     tfile = tempfile.NamedTemporaryFile(delete=False)
#     tfile.write(uploaded_file.read())
#     video_path = tfile.name

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     label_names = ["safe_drive","fatigue","drunk","drinking","hair_and_makeup","phonecall","talking_to_passenger"]
#     model = load_model("best_model_epoch18.pth", num_classes=len(label_names), device=device)

#     with st.spinner("Loading video..."):
#         result_path = predict_video_voted(model, video_path, label_names, device)

#     st.video(result_path)

import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import cv2, os, tempfile, math
import numpy as np
from collections import deque
from PIL import Image
from typing import List
from moviepy.editor import VideoFileClip
import imageio

# =========================
# Model & Blocks
# =========================
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        w = torch.softmax(self.attn(x), dim=1)
        return (x * w).sum(dim=1)

class CNNTransformer(nn.Module):
    def __init__(self, num_classes, seq_len=30, hidden_dim=256, num_heads=4, num_layers=1, pretrained=True, freeze_until=10):
        super().__init__()
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None)
        self.cnn = mobilenet.features
        for i, child in enumerate(self.cnn.children()):
            if i < freeze_until:
                for p in child.parameters():
                    p.requires_grad = False
        self.head = nn.Sequential(
            nn.Conv2d(1280, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            SEBlock(512),
            nn.Dropout(0.3)
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc_in = nn.Linear(512, hidden_dim)
        enc_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads,
                                               dim_feedforward=hidden_dim*2, dropout=0.3, activation="gelu")
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.attn_pool = AttentionPooling(hidden_dim)
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
    def forward(self, x):
        # x: [B, T, 3, H, W]
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        feats = self.cnn(x)
        feats = self.head(feats)
        feats = self.gap(feats).view(B, T, -1)
        feats = self.fc_in(feats)
        feats = self.transformer(feats)
        feats = self.attn_pool(feats)
        return self.fc_out(feats)

# =========================
# Config
# =========================
MODEL_PATH = "best_model_epoch18.pth"
LABEL_NAMES = ["safe_drive","fatigue","drunk","drinking","hair_and_makeup","phonecall","talking_to_passenger"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
VID_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".m4v")

# Fixed defaults
IMG_SIZE = 256
SEQ_LEN = 30
STEP = 30
VOTE = "hard"
BATCH_SIZE = 16

# =========================
# Cached resources
# =========================
@st.cache_resource(show_spinner=False)
def load_model(path: str, num_classes: int, device: torch.device):
    model = CNNTransformer(num_classes=num_classes).to(device)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model

@st.cache_resource(show_spinner=False)
def get_transform(img_size: int):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

def warmup_model(model, device, seq_len: int, img_size: int):
    if "warmed_up" in st.session_state:
        return
    with torch.inference_mode():
        dummy = torch.zeros(1, seq_len, 3, img_size, img_size, device=device)
        _ = model(dummy)
    st.session_state["warmed_up"] = True

# =========================
# Video prediction
# =========================
def predict_video_voted(
    model, video_path, label_names, device,
    seq_len=SEQ_LEN, step=STEP, img_size=IMG_SIZE,
    vote=VOTE, k=5
):
    tfm = get_transform(img_size)
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25

    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out_path = tmp_out.name
    writer = imageio.get_writer(out_path, fps=fps)

    window_buf = []
    prob_deque, label_deque = deque(maxlen=k), deque(maxlen=k)
    last_prob_vec = None

    while True:
        ok, fr = cap.read()
        if not ok:
            break
        window_buf.append(fr)

        if len(window_buf) == seq_len:
            rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in window_buf]
            tens = [tfm(Image.fromarray(x)).unsqueeze(0) for x in rgb]
            batch = torch.stack(tens, dim=1).to(device)     # [1, T, 3, H, W]
            with torch.no_grad():
                logits = model(batch)
                prob_vec = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
            last_prob_vec = prob_vec

            label_deque.append(int(np.argmax(prob_vec)))
            prob_deque.append(prob_vec)

            # always hard vote
            vals, counts = np.unique(label_deque, return_counts=True)
            voted_idx = int(vals[np.argmax(counts)])
            conf = float(last_prob_vec[voted_idx]) if last_prob_vec is not None else 0.0

            text = f"{label_names[voted_idx]} ({conf:.2f})"
            for f in window_buf[:step]:
                cv2.putText(f, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1.2, (0,255,0), 3, cv2.LINE_AA)
                writer.append_data(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            window_buf = window_buf[step:]

    cap.release()
    writer.close()

    final_out = out_path.replace(".mp4", "_final.mp4")
    clip = VideoFileClip(out_path)
    clip.write_videofile(final_out, codec="libx264", audio=False, verbose=False, logger=None)
    clip.close()
    os.remove(out_path)
    return final_out

# =========================
# Image prediction (batch)
# =========================
def predict_images_batch(
    model, pil_images: List[Image.Image], label_names, device,
    img_size=IMG_SIZE, seq_len=SEQ_LEN, batch_size=BATCH_SIZE
):
    tfm = get_transform(img_size)
    tensors = [tfm(img).unsqueeze(0) for img in pil_images]
    tensors = torch.cat(tensors, dim=0)  # [N,3,H,W]
    tensors_seq = tensors.unsqueeze(1).repeat(1, seq_len, 1, 1, 1)

    preds, probs = [], []
    model.eval()
    with torch.no_grad():
        for i in range(0, tensors_seq.size(0), batch_size):
            batch = tensors_seq[i:i+batch_size].to(device)
            logits = model(batch)
            prob = torch.softmax(logits, dim=1)
            pred = torch.argmax(prob, dim=1)
            preds.extend(pred.detach().cpu().tolist())
            probs.extend(prob.max(dim=1).values.detach().cpu().tolist())
    return preds, probs

# =========================
# App
# =========================
st.title("Driver Fatigue/Drunk/Distraction Detection 🚗")

with st.spinner("Loading model..."):
    model = load_model(MODEL_PATH, num_classes=len(LABEL_NAMES), device=DEVICE)
    warmup_model(model, DEVICE, seq_len=SEQ_LEN, img_size=IMG_SIZE)

uploaded_files = st.file_uploader(
    "Upload images and/or videos",
    type=list(set([*IMG_EXTS, *VID_EXTS])),
    accept_multiple_files=True
)

if uploaded_files:
    image_files, video_files = [], []
    for f in uploaded_files:
        name = f.name.lower()
        if name.endswith(IMG_EXTS):
            image_files.append(f)
        elif name.endswith(VID_EXTS):
            video_files.append(f)

    # Images
    if image_files:
        st.subheader(f"Uploaded Images ({len(image_files)})")
        pil_images, titles = [], []
        for f in image_files:
            img = Image.open(f).convert("RGB")
            pil_images.append(img)

        with st.spinner("Predicting images..."):
            preds, probs = predict_images_batch(model, pil_images, LABEL_NAMES, DEVICE)

        cols_per_row = 4
        rows = math.ceil(len(pil_images) / cols_per_row)
        idx = 0
        for _ in range(rows):
            cols = st.columns(cols_per_row)
            for c in cols:
                if idx >= len(pil_images): break
                c.image(
                    pil_images[idx],
                    caption=f"Prediction: {LABEL_NAMES[preds[idx]]} ({probs[idx]:.2f})",
                    use_container_width=True
                )
                idx += 1

    # Videos
    if video_files:
        st.subheader(f"Uploaded Videos ({len(video_files)})")
        for i, vf in enumerate(video_files, 1):
            st.write(f"**{i}. {vf.name}**")
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(vf.name)[1])
            tfile.write(vf.read())
            video_path = tfile.name

            with st.spinner("Processing video..."):
                result_path = predict_video_voted(model, video_path, LABEL_NAMES, DEVICE)
            st.video(result_path)

else:
    st.info("Please upload one or more **images** and/or **videos**.")
