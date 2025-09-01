import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

import torch
import torch.nn as nn
import torchvision.models as models

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
        mobilenet = models.mobilenet_v2(pretrained=pretrained)
        self.cnn = mobilenet.features
        child_counter = 0
        for child in self.cnn.children():
            if child_counter < freeze_until:
                for param in child.parameters():
                    param.requires_grad = False
            child_counter += 1
        self.head = nn.Sequential(
            nn.Conv2d(1280, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            SEBlock(512),
            nn.Dropout(0.3)
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc_in = nn.Linear(512, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim*2, dropout=0.3, activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.attn_pool = AttentionPooling(hidden_dim)
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        feats = self.cnn(x)
        feats = self.head(feats)
        feats = self.gap(feats).view(B, T, -1)
        feats = self.fc_in(feats)
        feats = self.transformer(feats)
        feats = self.attn_pool(feats)
        out = self.fc_out(feats)
        return out

@st.cache_resource
def load_model(model_path, num_classes, device):
    model = CNNTransformer(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

import os, cv2, torch, numpy as np
from collections import deque
from torchvision import transforms
from PIL import Image

def predict_video_voted(
    model, video_path, label_names, device,
    seq_len=30, step=30, img_size=256,
    vote="soft", k=5
):
    tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out_path = tmpfile.name

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    tfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

    window_buf = []
    prob_deque, label_deque = deque(maxlen=k), deque(maxlen=k)

    while True:
        ok, fr_bgr = cap.read()
        if not ok:
            break
        window_buf.append(fr_bgr)

        if len(window_buf) == seq_len:
            rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in window_buf]
            tens = [tfm(x).unsqueeze(0) for x in rgb]
            batch = torch.stack(tens, dim=1).to(device)

            with torch.no_grad():
                logits = model(batch)
                prob_vec = torch.softmax(logits, dim=1)[0].cpu().numpy()

            label_deque.append(np.argmax(prob_vec))
            prob_deque.append(prob_vec)

            if vote == "hard":
                vals, counts = np.unique(label_deque, return_counts=True)
                voted_idx = int(vals[np.argmax(counts)])
                voted_conf = float(np.mean([p[voted_idx] for p in prob_deque]))
            else:
                avg_prob = np.mean(np.stack(prob_deque, axis=0), axis=0)
                voted_idx = int(np.argmax(avg_prob))
                voted_conf = float(avg_prob[voted_idx])

            text = f"{label_names[voted_idx]} ({voted_conf:.2f})"
            for f in window_buf:
                cv2.putText(f, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1.2, (0,255,0), 3, cv2.LINE_AA)
                out.write(f)

            window_buf = window_buf[step:]

    cap.release()
    out.release()
    return out_path

# ==== Streamlit App ====
st.title("Driver Monitoring Demo 🚗")

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    st.video(video_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_names = ["safe_drive", "fatigue", "drunk", "drinking", "hair_and_makeup", "phonecall", "talking_to_passenger"]

    model = load_model("best_model_epoch18.pth", num_classes=len(label_names), device=device)

    st.write("Predicting...")
    result_path = predict_video_voted(model, video_path, label_names, device)

    st.success("Done! Video with predictions:")
    with open(result_path, "rb") as f:
        st.video(f.read())

