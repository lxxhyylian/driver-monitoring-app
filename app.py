import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import cv2, os, tempfile
import numpy as np
from collections import deque
from PIL import Image

# ==== Model định nghĩa ====
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
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        feats = self.cnn(x)
        feats = self.head(feats)
        feats = self.gap(feats).view(B, T, -1)
        feats = self.fc_in(feats)
        feats = self.transformer(feats)
        feats = self.attn_pool(feats)
        return self.fc_out(feats)

# ==== Load model ====
@st.cache_resource
def load_model(path, num_classes, device):
    model = CNNTransformer(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

# ==== Predict video và annotate ====
from moviepy.editor import VideoFileClip

def predict_video_voted(
    model, video_path, label_names, device,
    seq_len=30, step=30, img_size=256,
    vote="soft", k=5
):
    tfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    width, height = int(cap.get(3)), int(cap.get(4))

    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out_path = tmp_out.name
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"avc1"), fps, (width, height))

    window_buf, preds = [], []
    prob_deque, label_deque = deque(maxlen=k), deque(maxlen=k)

    while True:
        ok, fr = cap.read()
        if not ok: break
        window_buf.append(fr)

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
            else:
                avg_prob = np.mean(np.stack(prob_deque), axis=0)
                voted_idx = int(np.argmax(avg_prob))

            text = f"{label_names[voted_idx]}"
            for f in window_buf[:step]:
                cv2.putText(f, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1.2, (0,255,0), 3, cv2.LINE_AA)
                out.write(f)
            window_buf = window_buf[step:]

    cap.release()
    out.release()

    # 🔄 Dùng moviepy re-encode để Streamlit đọc chắc chắn được
    final_out = out_path.replace(".mp4", "_final.mp4")
    clip = VideoFileClip(out_path)
    clip.write_videofile(final_out, codec="libx264", audio=False, verbose=False, logger=None)
    clip.close()
    os.remove(out_path)  # xoá file gốc, chỉ giữ file chuẩn

    return final_out

# ==== Streamlit App ====
st.title("Driver Monitoring Demo 🚗")

uploaded_file = st.file_uploader("Upload a video", type=["mp4","avi","mov","mkv"])
if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_names = ["safe_drive","fatigue","drunk","drinking","hair_and_makeup","phonecall","talking_to_passenger"]
    model = load_model("best_model_epoch18.pth", num_classes=len(label_names), device=device)

    with st.spinner("Loading video..."):
        result_path = predict_video_voted(model, video_path, label_names, device)

    st.video(result_path)
