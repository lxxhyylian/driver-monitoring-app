import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import cv2, os, tempfile, math, hashlib, gc
import numpy as np
from collections import deque
from PIL import Image
from typing import List, Tuple
from moviepy.editor import VideoFileClip
import imageio
from io import BytesIO

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
        enc_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim*2, dropout=0.3, activation="gelu")
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

MODEL_PATH = "best_model_epoch18.pth"
LABEL_NAMES = ["safe_drive","fatigue","drunk","drinking","hair_and_makeup","phonecall","talking_to_passenger"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
VID_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".m4v")
IMG_SIZE = 256
SEQ_LEN = 30
STEP = 30
VOTE = "hard"
BATCH_SIZE = 16
PAGE_SIZE = 4
NEW_BATCH_DRAW_COLS = 4

if "processed_images" not in st.session_state:
    st.session_state.processed_images = {}
if "image_order" not in st.session_state:
    st.session_state.image_order = []
if "processed_videos" not in st.session_state:
    st.session_state.processed_videos = {}
if "video_order" not in st.session_state:
    st.session_state.video_order = []
if "images_page" not in st.session_state:
    st.session_state.images_page = 1

def file_key(uploaded_file) -> str:
    data = uploaded_file.getvalue()
    md5 = hashlib.md5(data).hexdigest()
    return f"{uploaded_file.name}|{md5}"

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

def predict_video_voted(model, video_path, label_names, device, seq_len=SEQ_LEN, step=STEP, img_size=IMG_SIZE, vote=VOTE, k=5):
    tfm = get_transform(img_size)
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out_path = tmp_out.name
    writer = imageio.get_writer(out_path, fps=fps, macro_block_size=None)
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
            batch = torch.stack(tens, dim=1).to(device)
            with torch.no_grad():
                logits = model(batch)
                prob_vec = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
            last_prob_vec = prob_vec
            label_deque.append(int(np.argmax(prob_vec)))
            prob_deque.append(prob_vec)
            vals, counts = np.unique(label_deque, return_counts=True)
            voted_idx = int(vals[np.argmax(counts)])
            conf = float(last_prob_vec[voted_idx]) if last_prob_vec is not None else 0.0
            text = f"{label_names[voted_idx]} ({conf:.2f})"
            for f in window_buf[:step]:
                cv2.putText(f, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3, cv2.LINE_AA)
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

def predict_bytes_batch(model, entries: List[Tuple[str,str,bytes]], label_names, device, img_size=IMG_SIZE, seq_len=SEQ_LEN, batch_size=BATCH_SIZE):
    tfm = get_transform(img_size)
    preds, probs, valid = [], [], []
    model.eval()
    i = 0
    while i < len(entries):
        sub = entries[i:i+batch_size]
        xs = []
        keep_idx = []
        for j, (_, name, b) in enumerate(sub):
            try:
                img = Image.open(BytesIO(b))
                img.verify()
                img = Image.open(BytesIO(b)).convert("RGB")
                xs.append(tfm(img).unsqueeze(0))
                keep_idx.append(j)
            except Exception:
                st.warning(f"Skipped invalid image: {name}")
        if xs:
            x = torch.cat(xs, dim=0).to(device)
            with torch.no_grad():
                xb = x.unsqueeze(1).expand(-1, seq_len, -1, -1, -1).contiguous()
                logits = model(xb)
                prob = torch.softmax(logits, dim=1)
                pred = torch.argmax(prob, dim=1)
                preds.extend(pred.detach().cpu().tolist())
                probs.extend(prob.max(dim=1).values.detach().cpu().tolist())
            del x, xb, logits, prob, pred
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            valid.extend([sub[k] for k in keep_idx])
        i += batch_size
        yield valid, preds, probs

def images_paginator(total, page_size):
    total_pages = max(1, math.ceil(total / page_size))
    col1, col2, col3 = st.columns([1,1,6])
    with col1:
        prev = st.button("◀ Prev", disabled=st.session_state.images_page <= 1)
    with col2:
        nxt = st.button("Next ▶", disabled=st.session_state.images_page >= total_pages)
    st.caption(f"Page {st.session_state.images_page}/{total_pages}")
    if prev:
        st.session_state.images_page = max(1, st.session_state.images_page - 1)
        st.rerun()
    if nxt:
        st.session_state.images_page = min(total_pages, st.session_state.images_page + 1)
        st.rerun()
    start = (st.session_state.images_page - 1) * page_size
    end = min(total, start + page_size)
    return start, end

st.title("Driver Fatigue/Drunk/Distraction Detection 🚗")

with st.spinner("Loading model..."):
    model = load_model(MODEL_PATH, num_classes=len(LABEL_NAMES), device=DEVICE)
    warmup_model(model, DEVICE, seq_len=SEQ_LEN, img_size=IMG_SIZE)

uploaded_files = st.file_uploader(
    "Upload images and/or videos",
    type=list(set([*IMG_EXTS, *VID_EXTS])),
    accept_multiple_files=True
)

new_image_entries = []
new_video_entries = []

if uploaded_files:
    for f in uploaded_files:
        key = file_key(f)
        lower = f.name.lower()
        if lower.endswith(IMG_EXTS):
            if key not in st.session_state.processed_images:
                new_image_entries.append((key, f.name, f.getvalue()))
        elif lower.endswith(VID_EXTS):
            if key not in st.session_state.processed_videos:
                new_video_entries.append((key, f.name, f.getvalue()))

new_results_container = st.container()

if new_image_entries:
    st.session_state.images_page = 1
    prog = st.progress(0.0)
    done = 0
    total = len(new_image_entries)
    grid_cont = new_results_container.container()
    batch_iter = predict_bytes_batch(model, new_image_entries, LABEL_NAMES, DEVICE)
    for valid, preds, probs in batch_iter:
        delta = len(valid) - len(st.session_state.image_order[:0])
        if delta > 0:
            pass
        new_cards = []
        while len(new_cards) < len(valid) - done:
            idx = done + len(new_cards)
            (key, name, b) = valid[idx]
            p = preds[idx]
            pr = probs[idx]
            st.session_state.processed_images[key] = {"name": name, "bytes": b, "pred": int(p), "prob": float(pr)}
            st.session_state.image_order.insert(0, key)
            new_cards.append(key)
        done = len(valid)
        if new_cards:
            cols = grid_cont.columns(min(NEW_BATCH_DRAW_COLS, len(new_cards)))
            r = 0
            for k in new_cards:
                item = st.session_state.processed_images[k]
                img = Image.open(BytesIO(item["bytes"])).convert("RGB")
                cap = f"Prediction: {LABEL_NAMES[item['pred']]} ({item['prob']:.2f})"
                cols[r % len(cols)].image(img, caption=cap, width="stretch")
                r += 1
        prog.progress(min(1.0, done / total))
    prog.empty()

if new_video_entries:
    st.session_state.images_page = 1
    for key, name, b in new_video_entries:
        suffix = os.path.splitext(name)[1]
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tfile.write(b)
        video_path = tfile.name
        result_path = predict_video_voted(model, video_path, LABEL_NAMES, DEVICE)
        st.session_state.processed_videos[key] = {"name": name, "result_path": result_path}
        st.session_state.video_order.insert(0, key)

if st.session_state.image_order:
    st.subheader(f"Images ({len(st.session_state.image_order)})")
    start, end = images_paginator(len(st.session_state.image_order), PAGE_SIZE)
    subset_keys = st.session_state.image_order[start:end]
    n = len(subset_keys)
    i = 0
    while i < n:
        cols = st.columns(4 if n - i >= 4 else (n - i))
        for c in cols:
            item = st.session_state.processed_images[subset_keys[i]]
            img = Image.open(BytesIO(item["bytes"])).convert("RGB")
            cap = f"Prediction: {LABEL_NAMES[item['pred']]} ({item['prob']:.2f})"
            c.image(img, caption=cap, width="stretch")
            i += 1
            if i >= n:
                break

if st.session_state.video_order:
    st.subheader(f"Videos ({len(st.session_state.video_order)})")
    for idx, key in enumerate(st.session_state.video_order, 1):
        item = st.session_state.processed_videos[key]
        st.write(f"{idx}. {item['name']}")
        st.video(item["result_path"])

if not uploaded_files and not st.session_state.image_order and not st.session_state.video_order:
    st.info("You can upload one or more images or videos.")
