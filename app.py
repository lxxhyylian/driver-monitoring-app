import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import cv2, os, tempfile, math, hashlib
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
# Session state (giữ kết quả cũ)
# =========================
if "processed_images" not in st.session_state:
    # key -> {"name": str, "bytes": bytes, "pred": int, "prob": float}
    st.session_state.processed_images = {}
if "image_order" not in st.session_state:
    st.session_state.image_order = []  # list of keys để giữ thứ tự hiển thị

if "processed_videos" not in st.session_state:
    # key -> {"name": str, "result_path": str}
    st.session_state.processed_videos = {}
if "video_order" not in st.session_state:
    st.session_state.video_order = []

# =========================
# Helpers
# =========================
def file_key(uploaded_file) -> str:
    """Tạo khóa duy nhất cho file dựa trên bytes (MD5) + tên."""
    data = uploaded_file.getvalue()
    md5 = hashlib.md5(data).hexdigest()
    return f"{uploaded_file.name}|{md5}"

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

# ====== Chỉ predict file MỚI; giữ kết quả cũ ======
new_image_entries = []  # [(key, name, bytes)]
new_video_entries = []  # [(key, name, bytes)]

if uploaded_files:
    for f in uploaded_files:
        key = file_key(f)
        if f.name.lower().endswith(IMG_EXTS):
            if key not in st.session_state.processed_images:
                new_image_entries.append((key, f.name, f.getvalue()))
        elif f.name.lower().endswith(VID_EXTS):
            if key not in st.session_state.processed_videos:
                new_video_entries.append((key, f.name, f.getvalue()))

# ---- Predict new IMAGES (batch) ----
if new_image_entries:
    pil_images_new = [Image.open(tempfile.NamedTemporaryFile(delete=False)).convert("RGB") for _ in new_image_entries]  # placeholder
    # Re-open from bytes properly
    pil_images_new = []
    for key, name, b in new_image_entries:
        pil_images_new.append(Image.open(Image.BytesIO(b)).convert("RGB") if hasattr(Image, "BytesIO") else Image.open(tempfile.NamedTemporaryFile(delete=False)))

    # Fallback if PIL.Image.BytesIO not available in your env
    pil_images_new = []
    for key, name, b in new_image_entries:
        from io import BytesIO
        pil_images_new.append(Image.open(BytesIO(b)).convert("RGB"))

    preds, probs = predict_images_batch(model, pil_images_new, LABEL_NAMES, DEVICE)
    # Lưu kết quả & thứ tự
    for (key, name, b), pred, prob in zip(new_image_entries, preds, probs):
        st.session_state.processed_images[key] = {
            "name": name, "bytes": b, "pred": int(pred), "prob": float(prob)
        }
        st.session_state.image_order.append(key)

# ---- Predict new VIDEOS (one by one) ----
if new_video_entries:
    for key, name, b in new_video_entries:
        # ghi tạm để xử lý
        suffix = os.path.splitext(name)[1]
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tfile.write(b)
        video_path = tfile.name
        result_path = predict_video_voted(model, video_path, LABEL_NAMES, DEVICE)
        st.session_state.processed_videos[key] = {"name": name, "result_path": result_path}
        st.session_state.video_order.append(key)

# =========================
# DISPLAY (luôn hiển thị tất cả kết quả đã có, không re-predict)
# =========================

# Images display (dynamic grid: <=4 per row; nếu còn 1-3 ảnh ở hàng cuối thì tạo đúng số cột)
if st.session_state.image_order:
    st.subheader(f"Images ({len(st.session_state.image_order)})")
    # Chuẩn bị list PIL Images từ bytes
    display_items = []
    for key in st.session_state.image_order:
        item = st.session_state.processed_images[key]
        from io import BytesIO
        img = Image.open(BytesIO(item["bytes"])).convert("RGB")
        caption = f"Prediction: {LABEL_NAMES[item['pred']]} ({item['prob']:.2f})"
        display_items.append((img, caption))

    n = len(display_items)
    i = 0
    while i < n:
        cols_this_row = min(4, n - i)  # số ảnh còn lại (<=4)
        cols = st.columns(cols_this_row)
        for c in cols:
            img, cap = display_items[i]
            c.image(img, caption=cap, use_container_width=True)
            i += 1
            if i >= n:
                break

# Videos display (giữ nguyên đã xử lý)
if st.session_state.video_order:
    st.subheader(f"Videos ({len(st.session_state.video_order)})")
    for idx, key in enumerate(st.session_state.video_order, 1):
        item = st.session_state.processed_videos[key]
        st.write(f"**{idx}. {item['name']}**")
        st.video(item["result_path"])

# Hint
if not uploaded_files and not st.session_state.image_order and not st.session_state.video_order:
    st.info("Please upload one or more **images** and/or **videos**.")

