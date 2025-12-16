import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import time
import os
import pandas as pd
import numpy as np

# --- 1. CẤU HÌNH GIAO DIỆN ---
st.set_page_config(
    page_title="Bác sĩ Táo AI",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="collapsed" 
)

st.markdown("""
    <style>
        /* Container kính mờ */
        .glass-container {
            background: rgba(255, 255, 255, 0.95);
            padding: 25px;
            border-radius: 20px;
            box-shadow: 0px 8px 32px rgba(0,0,0,0.1);
            margin-bottom: 25px;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        /* Nút bấm hiện đại */
        .stButton>button {
            width: 100%;
            border-radius: 12px;
            height: 50px;
            font-weight: 700;
            font-size: 16px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 7px 14px rgba(0,0,0,0.15);
        }

        /* Thanh tiến trình */
        .stProgress > div > div > div > div {
            background-image: linear-gradient(90deg, #00b09b, #96c93d);
        }
        
        /* Ẩn menu mặc định và Sidebar */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        [data-testid="stSidebar"] {display: none;}
        
        /* Căn chỉnh header */
        .main-header {
            text-align: center;
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# --- 2. CẤU HÌNH HỆ THỐNG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ['Apple Scab', 'Black Rot', 'Cedar Apple Rust', 'Healthy']

# Đường dẫn (Cập nhật cho máy Local)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models') 

DESCRIPTIONS = {
    "Apple Scab": "🍂 **Bệnh ghẻ táo:** Xuất hiện đốm nâu/ô liu trên lá, có thể gây rụng lá sớm.",
    "Black Rot": "🟣 **Bệnh thối đen:** Đốm tím nhỏ lan rộng thành hình tròn, tâm màu nâu hoặc xám.",
    "Cedar Apple Rust": "🟠 **Bệnh gỉ sắt:** Đốm màu vàng cam hoặc đỏ tươi trên mặt trên của lá.",
    "Healthy": "✅ **Lá khỏe mạnh:** Màu xanh đều, không có đốm lạ hay dấu hiệu tổn thương."
}

# --- 3. HÀM TẢI ĐA MÔ HÌNH (CACHE) ---
@st.cache_resource
def load_all_models():
    models_dict = {}
    
    if not os.path.exists(MODEL_DIR):
        st.error(f"⚠️ Không tìm thấy thư mục models tại: {MODEL_DIR}")
        st.info("Vui lòng tạo thư mục 'models' và chép các file .pth vào đó.")
        return {}

    model_configs = [
        ('ResNet50', 'resnet50_best.pth', models.resnet50, 'fc'),
        ('MobileNetV2', 'mobilenet_v2_best.pth', models.mobilenet_v2, 'classifier'),
        ('EfficientNetB0', 'efficientnet_b0_best.pth', models.efficientnet_b0, 'classifier')
    ]

    for name, filename, model_func, layer_name in model_configs:
        path = os.path.join(MODEL_DIR, filename)
        if os.path.exists(path):
            try:
                model = model_func()
                if layer_name == 'fc':
                    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
                else:
                    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASSES))
                
                model.load_state_dict(torch.load(path, map_location=DEVICE))
                model.eval().to(DEVICE)
                models_dict[name] = model
            except Exception as e:
                st.error(f"Lỗi tải {name}: {e}")
    
    return models_dict

with st.spinner("🚀 Đang khởi động hệ thống AI..."):
    loaded_models = load_all_models()

if not loaded_models:
    st.error("❌ Không tải được mô hình nào. Vui lòng kiểm tra file model.")
    st.stop()

# --- 4. XỬ LÝ ẢNH ---
def process_image(image):
    transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(DEVICE)

# --- 5. GIAO DIỆN CHÍNH (NO SIDEBAR) ---

# Header
col_h1, col_h2 = st.columns([1, 8])
with col_h1:
    st.image("https://img.icons8.com/color/96/000000/apple-orchard.png", width=80)
with col_h2:
    st.title("Bác sĩ Táo AI")
    st.caption(f"Engine: {'🟢 GPU' if torch.cuda.is_available() else '🟡 CPU'}")

st.markdown("---")

# 1. Điều hướng & Cài đặt (Đưa ra màn hình chính)
col_nav1, col_nav2 = st.columns([1, 1])

with col_nav1:
    st.subheader("1. Chức năng")
    app_mode = st.radio(
        "Chế độ:",
        ["🔍 Chẩn đoán bệnh", "⚡ So sánh Hiệu năng"],
        label_visibility="collapsed"
    )

with col_nav2:
    selected_model_name = None
    if app_mode == "🔍 Chẩn đoán bệnh":
        st.subheader("2. Cấu hình Model")
        selected_model_name = st.selectbox(
            "Chọn kiến trúc AI:", 
            list(loaded_models.keys()),
            index=0,
            label_visibility="collapsed"
        )

st.markdown("---")

# 2. Input (Upload/Camera)
st.subheader("3. Nhập dữ liệu")
input_source = st.radio("Nguồn ảnh:", ["📂 Tải ảnh lên", "📷 Chụp ảnh"], horizontal=True, label_visibility="collapsed")

img_file = None
if input_source == "📂 Tải ảnh lên":
    img_file = st.file_uploader("Chọn ảnh từ thiết bị:", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
else:
    img_file = st.camera_input("Chụp ảnh lá táo")

# --- MAIN CONTENT ---

if not img_file:
    # Màn hình chờ (Welcome screen)
    if input_source == "📂 Tải ảnh lên":
        st.info("👆 Vui lòng chọn ảnh để bắt đầu.")
    else:
        st.info("👆 Vui lòng chụp ảnh để bắt đầu.")

elif app_mode == "🔍 Chẩn đoán bệnh":
    # Xử lý ảnh
    image = Image.open(img_file).convert('RGB')
    
    # Chỉ hiện ảnh preview nếu là upload (Camera đã có preview riêng)
    if input_source == "📂 Tải ảnh lên":
        with st.expander("📸 Xem ảnh gốc", expanded=True):
            st.image(image, caption="Ảnh đầu vào", use_container_width=True)
            
    img_tensor = process_image(image)

    # --- GIAO DIỆN CHẨN ĐOÁN ---
    st.header("🔍 Kết quả Phân tích")
    
    # Container kính mờ
    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    
    col_info, col_act = st.columns([3, 1])
    with col_info:
        st.info(f"Đang sử dụng mô hình: **{selected_model_name}**")
    
    with col_act:
        run_btn = st.button("🔎 Phân tích ngay", type="primary")

    if run_btn:
        model = loaded_models[selected_model_name]
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)[0]
            conf, pred_idx = torch.max(probs, 0)
        
        end_time = time.time()
        processing_time = (end_time - start_time) * 1000
        
        # Layout kết quả
        col_res1, col_res2 = st.columns([1, 3])
        
        with col_res1:
            if conf.item() > 0.8:
                st.image("https://img.icons8.com/color/96/000000/checked--v1.png", width=100)
            else:
                st.image("https://img.icons8.com/color/96/000000/high-priority.png", width=100)
        
        with col_res2:
            pred_label = CLASSES[pred_idx]
            st.success(f"### {pred_label}")
            st.progress(int(conf.item()*100), text=f"Độ tin cậy: {conf.item()*100:.1f}%")
            st.caption(f"⏱️ Thời gian xử lý: {processing_time:.0f} ms")

        # Thông tin bệnh chi tiết
        st.markdown("---")
        st.markdown(f"### 📖 Kiến thức nhà nông:")
        # Thêm color: #333333 để đảm bảo chữ không bị trắng trên nền sáng
        st.markdown(f"""
        <div style="background-color: #f0f2f6; color: #333333; padding: 15px; border-radius: 10px; border-left: 5px solid #4CAF50;">
            {DESCRIPTIONS.get(pred_label, "Chưa có thông tin.")}
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown('</div>', unsafe_allow_html=True)

elif app_mode == "⚡ So sánh Hiệu năng":
    # Xử lý ảnh
    image = Image.open(img_file).convert('RGB')
    if input_source == "📂 Tải ảnh lên":
        with st.expander("📸 Xem ảnh gốc", expanded=True):
            st.image(image, caption="Ảnh đầu vào", use_container_width=True)
    img_tensor = process_image(image)

    # --- GIAO DIỆN BENCHMARK ---
    st.header("⚡ So sánh Hiệu năng AI")
    
    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    
    st.write("Kiểm tra tốc độ thực tế của các mô hình trên thiết bị này:")
    
    if st.button("🚀 Chạy Benchmark tất cả", key="btn_bench", type="primary"):
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_models = len(loaded_models)
        
        for i, (name, model) in enumerate(loaded_models.items()):
            status_text.markdown(f"**Đang kiểm tra:** `{name}`...")
            
            # Warmup (làm nóng GPU/CPU)
            with torch.no_grad(): _ = model(img_tensor)
            
            # Benchmark loop (chạy 5 lần lấy trung bình)
            times = []
            for _ in range(5):
                start = time.time()
                with torch.no_grad():
                    out = model(img_tensor)
                    prob = torch.nn.functional.softmax(out, dim=1)[0]
                    c, p_idx = torch.max(prob, 0)
                end = time.time()
                times.append((end - start) * 1000)
            
            avg_time = sum(times) / len(times)
            param_size = sum(p.numel() for p in model.parameters()) * 4 / (1024**2)
            
            results.append({
                "Mô hình": name,
                "Dự đoán": CLASSES[p_idx],
                "Độ tin cậy": f"{c.item()*100:.1f}%",
                "Tốc độ (ms)": avg_time,
                "Kích thước (MB)": param_size
            })
            progress_bar.progress((i + 1) / total_models)
        
        status_text.success("✅ Đã hoàn tất kiểm tra!")
        
        # Xử lý hiển thị bảng đẹp hơn
        df = pd.DataFrame(results)
        
        # Highlight hàng tốt nhất
        st.dataframe(
            df.style.highlight_min(subset=["Tốc độ (ms)"], color='#d4edda')
                    .highlight_max(subset=["Độ tin cậy"], color='#cce5ff')
                    .format({"Tốc độ (ms)": "{:.1f}", "Kích thước (MB)": "{:.1f}"}),
            use_container_width=True
        )
        
        # Biểu đồ so sánh
        st.write("### 📉 Biểu đồ Tốc độ (Thấp hơn là Tốt hơn)")
        chart_data = df.set_index("Mô hình")[["Tốc độ (ms)"]]
        st.bar_chart(chart_data)
        
    st.markdown('</div>', unsafe_allow_html=True)