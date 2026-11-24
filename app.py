import os
import sys
import glob
import json
import types
import random
import zipfile
import tempfile
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

# Bypass pymatting/numba agar rembg berjalan normal di Python 3.14
for mod_name in [
    "pymatting",
    "pymatting.alpha",
    "pymatting.alpha.estimate_alpha_cf",
    "pymatting.foreground",
    "pymatting.foreground.estimate_foreground_ml",
    "pymatting.util",
    "pymatting.util.util",
]:
    if mod_name not in sys.modules:
        dummy = types.ModuleType(mod_name)
        dummy.estimate_alpha_cf = None
        dummy.estimate_foreground_ml = None
        dummy.stack_images = None
        sys.modules[mod_name] = dummy

try:
    from rembg import remove, new_session
    REMBG_AVAILABLE = True
except Exception:
    REMBG_AVAILABLE = False

# KONFIGURASI HALAMAN & UI/UX (DARK MINIMALIST THEME)
st.set_page_config(
    page_title="Tree Ring Detector & Age Estimator",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp {
        background-color: #121212;
        color: #E0E0E0;
    }
    section[data-testid="stSidebar"] {
        background-color: #1A1A1A;
        border-right: 1px solid #2C2C2C;
    }
    div[data-testid="stMetric"] {
        background-color: #1E1E1E;
        border: 1px solid #2E2E2E;
        padding: 14px 18px;
        border-radius: 10px;
    }
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_rembg_session():
    """Cache sesi model ringan u2netp (4.7 MB) agar pemrosesan cepat."""
    if REMBG_AVAILABLE:
        return new_session("u2netp")
    return None


# FUNGSI INTI COMPUTER VISION (DETEKSI LINGKARAN TAHUN ADAPTIF)
def segment_wood_no_shadow(img_bgr, is_studio_bg):
    """Memisahkan penampang kayu dari background putih maupun latar alam menggunakan AI (rembg)."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)
    img_h, img_w = img_bgr.shape[:2]

    if is_studio_bg or not REMBG_AVAILABLE:
        wood_mask = np.where((s > 25) | (v < 110), 255, 0).astype(np.uint8)
        wood_mask = cv2.morphologyEx(
            wood_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
        )
        wood_mask = cv2.morphologyEx(
            wood_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        )
    else:
        session = get_rembg_session()
        rgba = remove(img_bgr, session=session)
        alpha = rgba[:, :, 3]
        wood_mask = np.where(alpha > 128, 255, 0).astype(np.uint8)
        wood_mask = cv2.morphologyEx(
            wood_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        )
        wood_mask = cv2.morphologyEx(
            wood_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
        )

    contours, _ = cv2.findContours(wood_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_mask = np.zeros_like(wood_mask)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(clean_mask, [largest], -1, 255, thickness=cv2.FILLED)
        (_, _), max_radius = cv2.minEnclosingCircle(largest)
        max_radius = int(max_radius)
    else:
        clean_mask = wood_mask
        max_radius = min(img_h, img_w) // 2

    safe_mask = cv2.erode(
        clean_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (27, 27)), iterations=1
    )
    if cv2.countNonZero(safe_mask) < 500:
        safe_mask = clean_mask

    return safe_mask, max_radius


def direct_polar_remap(img_gray, mask, cx, cy, max_radius, num_angles=360):
    """Transformasi polar langsung dengan cv2.remap agar 100% sinkron dengan koordinat gambar."""
    angles = np.linspace(0, 2 * np.pi, num_angles, endpoint=False, dtype=np.float32)
    radii = np.arange(0, max_radius, dtype=np.float32)

    R, A = np.meshgrid(radii, angles)
    map_x = (cx + R * np.cos(A)).astype(np.float32)
    map_y = (cy + R * np.sin(A)).astype(np.float32)

    polar_img = cv2.remap(
        img_gray, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255
    )
    polar_mask = cv2.remap(
        mask, map_x, map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    return polar_img, polar_mask, angles


def find_true_pith_refined(gray, safe_mask, max_radius):
    """Mencari pusat asli lingkaran (pith) lalu mengkalibrasinya agar tepat di tengah pusaran."""
    h, w = gray.shape

    mask_for_dist = safe_mask.copy()
    mask_for_dist[0, :] = 0
    mask_for_dist[-1, :] = 0
    mask_for_dist[:, 0] = 0
    mask_for_dist[:, -1] = 0

    dist = cv2.distanceTransform(mask_for_dist, cv2.DIST_L2, 5)
    max_dist = float(np.max(dist))
    if not np.isfinite(max_dist) or max_dist <= 0:
        return w // 2, h // 2

    max_dist = min(max_dist, float(max(h, w)))

    center_region = (dist > 0.40 * max_dist).astype(np.uint8) * 255
    inner_grad_mask = (dist > 0.25 * max_dist).astype(np.uint8) * 255

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=5)
    mag = np.hypot(gx, gy)

    valid_mag = mag[inner_grad_mask > 128]
    if len(valid_mag) == 0:
        cy, cx = np.unravel_index(np.argmax(dist), dist.shape)
        return int(cx), int(cy)

    thresh_mag = np.percentile(valid_mag, 75)
    ys, xs = np.where((mag >= thresh_mag) & (inner_grad_mask > 128))

    acc = np.zeros((h, w), dtype=np.float32)
    step = max(1, len(xs) // 1800)
    line_len = int(max_dist * 0.75)

    for i in range(0, len(xs), step):
        x, y = float(xs[i]), float(ys[i])
        m = float(mag[int(y), int(x)])
        if m < 1e-4 or not np.isfinite(m):
            continue

        dx, dy = float(gx[int(y), int(x)]) / m, float(gy[int(y), int(x)]) / m
        if not (np.isfinite(dx) and np.isfinite(dy)):
            continue

        x1 = int(np.clip(round(x - dx * line_len), -10000, 10000))
        y1 = int(np.clip(round(y - dy * line_len), -10000, 10000))
        x2 = int(np.clip(round(x + dx * line_len), -10000, 10000))
        y2 = int(np.clip(round(y + dy * line_len), -10000, 10000))

        temp = np.zeros((h, w), dtype=np.uint8)
        cv2.line(temp, (x1, y1), (x2, y2), 1, thickness=2)
        acc += temp

    acc = cv2.GaussianBlur(acc, (25, 25), 0)
    acc[center_region == 0] = 0

    if np.max(acc) == 0:
        cy0, cx0 = np.unravel_index(np.argmax(dist), dist.shape)
    else:
        cy0, cx0 = np.unravel_index(np.argmax(acc), acc.shape)

    best_cx, best_cy = int(cx0), int(cy0)
    best_score = -1.0
    check_r = max(35, int(max_dist * 0.6))

    for dy in range(-14, 15, 4):
        for dx in range(-14, 15, 4):
            tx, ty = int(cx0 + dx), int(cy0 + dy)
            if 0 <= tx < w and 0 <= ty < h and safe_mask[ty, tx] > 128:
                p_img, _, _ = direct_polar_remap(blur, safe_mask, tx, ty, check_r, num_angles=72)
                prof = np.median(p_img, axis=0)

                hp = prof - gaussian_filter1d(prof, sigma=4.0)
                score = float(np.std(hp))
                if score > best_score:
                    best_score = score
                    best_cx, best_cy = tx, ty

    return best_cx, best_cy


def trace_ring_seamless_dp(valley_map, r_guide, raw_edge_radii, window=3):
    """Pelacakan jalur cincin 360° tanpa sambungan (wrapped DP) agar menempel presisi."""
    num_angles = valley_map.shape[0]
    pad = 45
    total_a = num_angles + 2 * pad
    w_size = 2 * window + 1

    scores = np.full((total_a, w_size), -1e6, dtype=np.float32)
    for step_a in range(total_a):
        a = (step_a - pad) % num_angles
        max_r = int(raw_edge_radii[a]) - 3
        base_r = int(round(r_guide[a]))
        for idx, offset in enumerate(range(-window, window + 1)):
            r = base_r + offset
            if 3 <= r <= max_r:
                scores[step_a, idx] = valley_map[a, r] - 0.12 * (offset ** 2)

    dp = np.zeros_like(scores)
    backtrack = np.zeros((total_a, w_size), dtype=np.int32)
    dp[0] = scores[0]

    for step_a in range(1, total_a):
        for j in range(w_size):
            j_min = max(0, j - 1)
            j_max = min(w_size, j + 2)
            prev_slice = dp[step_a - 1, j_min:j_max]
            best_prev = j_min + int(np.argmax(prev_slice))
            dp[step_a, j] = scores[step_a, j] + dp[step_a - 1, best_prev]
            backtrack[step_a, j] = best_prev

    path_idx = np.zeros(total_a, dtype=np.int32)
    path_idx[-1] = int(np.argmax(dp[-1]))
    for step_a in range(total_a - 2, -1, -1):
        path_idx[step_a] = backtrack[step_a + 1, path_idx[step_a + 1]]

    valid_offsets = path_idx[pad:pad + num_angles] - window
    r_traced = r_guide + valid_offsets

    vals = []
    for a in range(num_angles):
        r_idx = int(np.clip(round(r_traced[a]), 0, valley_map.shape[1] - 1))
        vals.append(valley_map[a, r_idx])
    continuity_ratio = float(np.mean(np.array(vals) > 0.03))

    return r_traced, continuity_ratio


def _extract_rings(sig_norm, valid_start, norm_len, strict_edge_radii, valley_map, raw_edge_radii,
                   prominence, min_distance, min_continuity):
    """Menjalankan pencarian puncak dan pelacakan kontur cincin berdasarkan kepadatan cincin."""
    peaks, _ = find_peaks(sig_norm, distance=min_distance, prominence=prominence)
    peaks = peaks + valid_start

    dp_window = max(2, min(4, int(min_distance // 2)))

    all_rings = []
    for r_norm in peaks:
        ratio = r_norm / float(norm_len)
        r_guide = ratio * strict_edge_radii
        r_traced, continuity = trace_ring_seamless_dp(
            valley_map, r_guide, raw_edge_radii, window=dp_window
        )

        if continuity >= min_continuity:
            r_final = gaussian_filter1d(r_traced, sigma=5.0, mode='wrap')
            r_final = np.clip(r_final, 3, raw_edge_radii - 3)
            all_rings.append(r_final)

    min_gap = max(2.2, min_distance * 0.5)
    for i in range(1, len(all_rings)):
        all_rings[i] = np.maximum(all_rings[i], all_rings[i - 1] + min_gap)
        all_rings[i] = np.clip(all_rings[i], 3, raw_edge_radii - 3)

    return all_rings


def detect_tree_rings_from_array(img_bgr):
    """Mendeteksi dan menghitung lingkaran tahun secara adaptif (mendukung cincin lebar maupun rapat)."""
    if img_bgr is None:
        raise ValueError("Data gambar tidak valid atau kosong.")

    target_size = 800
    h, w = img_bgr.shape[:2]
    scale_resize = target_size / max(h, w)
    img = cv2.resize(
        img_bgr,
        (int(w * scale_resize), int(h * scale_resize)),
        interpolation=cv2.INTER_CUBIC
    )

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 1. Ekstraksi kanal kontras tinggi (gabungan CIELAB L-channel & Blue channel)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, _, _ = cv2.split(lab)
    b_channel = img[:, :, 0]
    base_gray = cv2.addWeighted(l_channel, 0.5, b_channel, 0.5, 0)

    # 2. Bilateral Filter: haluskan pori kayu tanpa memburamkan tepi garis cincin
    denoised = cv2.bilateralFilter(base_gray, d=7, sigmaColor=35, sigmaSpace=35)

    # Deteksi otomatis jenis latar belakang (studio putih vs latar alam)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)
    border_s = np.concatenate([s[0, :], s[-1, :], s[:, 0], s[:, -1]])
    border_v = np.concatenate([v[0, :], v[-1, :], v[:, 0], v[:, -1]])
    is_studio_bg = bool(np.mean((border_s < 25) & (border_v > 200)) > 0.50)

    safe_mask, max_radius = segment_wood_no_shadow(img, is_studio_bg)
    cx, cy = find_true_pith_refined(denoised, safe_mask, max_radius)

    # 3. CLAHE + Unsharp Masking untuk menajamkan batas cincin yang samar
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(12, 12))
    clahe_img = clahe.apply(denoised)
    gaussian_sharp = cv2.GaussianBlur(clahe_img, (0, 0), 2.0)
    enhanced = cv2.addWeighted(clahe_img, 1.4, gaussian_sharp, -0.4, 0)

    num_angles = 360
    polar, polar_mask, angles = direct_polar_remap(enhanced, safe_mask, cx, cy, max_radius, num_angles)

    raw_edge_radii = np.zeros(num_angles, dtype=np.float32)
    for a in range(num_angles):
        valid_r = np.where(polar_mask[a] > 128)[0]
        raw_edge_radii[a] = valid_r[-1] if len(valid_r) > 0 else 15.0

    smooth_edge_radii = gaussian_filter1d(raw_edge_radii, sigma=8, mode='wrap')
    strict_edge_radii = np.minimum(raw_edge_radii, smooth_edge_radii)

    polar_float = polar.astype(np.float32)
    low_blur = gaussian_filter1d(polar_float, sigma=0.8, axis=1)
    high_blur = gaussian_filter1d(polar_float, sigma=3.5, axis=1)
    raw_valley = high_blur - low_blur

    local_std = np.sqrt(gaussian_filter1d(raw_valley ** 2, sigma=6.0, axis=1)) + 1.5
    valley_map = raw_valley / local_std
    valley_map[polar_mask < 128] = -10.0

    norm_len = max(int(np.median(strict_edge_radii)), 25)
    norm_valley = np.zeros((num_angles, norm_len), dtype=np.float32)
    for a in range(num_angles):
        r_end = max(int(strict_edge_radii[a]), 15)
        ray = valley_map[a, :r_end]
        norm_valley[a] = np.interp(np.linspace(0, 1, norm_len), np.linspace(0, 1, len(ray)), ray)

    radial_profile = np.median(norm_valley, axis=0)
    smoothed_profile = gaussian_filter1d(radial_profile, sigma=0.8)

    valid_start = 6
    valid_end = int(norm_len * 0.98)
    sig = smoothed_profile[valid_start:valid_end]
    sig_std = np.std(sig)
    sig_norm = (sig - np.mean(sig)) / sig_std if sig_std > 1e-6 else sig

    # Estimasi kepadatan cincin untuk menentukan jarak minimum antar puncak secara adaptif
    raw_peaks, _ = find_peaks(sig_norm, distance=4, prominence=0.22)
    if len(raw_peaks) >= 4:
        median_spacing = float(np.median(np.diff(raw_peaks)))
        adaptive_dist = int(np.clip(round(median_spacing * 0.65), 4, 12))
    else:
        adaptive_dist = 7

    if is_studio_bg:
        prominence = 0.28
        min_continuity = 0.48
    else:
        prominence = 0.24
        min_continuity = 0.45

    all_rings = _extract_rings(
        sig_norm, valid_start, norm_len, strict_edge_radii, valley_map, raw_edge_radii,
        prominence=prominence, min_distance=adaptive_dist, min_continuity=min_continuity
    )

    # Kalibrasi ulang otomatis jika cincin sangat rapat/samar sehingga terdeteksi terlalu sedikit
    if len(all_rings) < 4:
        all_rings = _extract_rings(
            sig_norm, valid_start, norm_len, strict_edge_radii, valley_map, raw_edge_radii,
            prominence=0.18, min_distance=4, min_continuity=0.40
        )

    result_img = img_rgb.copy()
    cv2.circle(result_img, (cx, cy), 5, (255, 50, 50), -1)

    line_thickness = 2 if len(all_rings) > 20 else 3

    for r_ring in all_rings:
        pts_x = (cx + r_ring * np.cos(angles)).astype(np.int32)
        pts_y = (cy + r_ring * np.sin(angles)).astype(np.int32)
        pts = np.vstack((pts_x, pts_y)).T.reshape((-1, 1, 2))
        cv2.polylines(
            result_img, [pts], isClosed=True,
            color=(0, 175, 255), thickness=line_thickness, lineType=cv2.LINE_AA
        )

    jumlah_lingkaran = len(all_rings)
    usia_tahun = jumlah_lingkaran * 1

    return img_rgb, result_img, jumlah_lingkaran, usia_tahun


def find_folder(base_dir, target_name):
    for root, dirs, _ in os.walk(base_dir):
        if target_name in dirs:
            return os.path.join(root, target_name)
    return None


def get_images_in_dir(folder):
    if not folder or not os.path.exists(folder):
        return []
    return sorted(
        glob.glob(os.path.join(folder, '*.jpg')) +
        glob.glob(os.path.join(folder, '*.jpeg')) +
        glob.glob(os.path.join(folder, '*.png'))
    )


def get_ground_truth_rings(image_path, ann_dir):
    """Membaca jumlah cincin asli (ground truth) dari file JSON pasangan gambar jika tersedia."""
    if not ann_dir or not os.path.exists(ann_dir):
        return None
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    json_path = os.path.join(ann_dir, base_name + '.json')
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return len(data.get('shapes', []))
    return None


def encode_image_to_bytes(img_rgb):
    """Konversi array RGB ke bytes PNG untuk tombol download."""
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    success, buffer = cv2.imencode('.png', img_bgr)
    return buffer.tobytes() if success else b""



# ANTARMUKA PENGGUNA (UI/UX STREAMLIT)
def main():
    st.title("🌲 Tree Ring Detector & Age Estimator")
    st.caption("Deteksi otomatis lingkaran tahun penampang kayu untuk mengestimasi usia pohon.")

    if not REMBG_AVAILABLE:
        st.warning(
            "Library `rembg` belum terpasang di environment Python ini. "
            "Jalankan `pip install rembg onnxruntime` di terminal VS Code."
        )

    #  SIDEBAR: PENGATURAN INPUT 
    with st.sidebar:
        st.header("⚙️ Pengaturan Input")

        input_mode = st.radio(
            "Pilih Sumber Gambar:",
            ["Upload Gambar Langsung", "Upload Dataset (.ZIP)"],
            index=0
        )

        st.divider()
        st.info(
            "🤖 **Deteksi Adaptif Aktif**\n\n"
            "Sistem secara otomatis mengenali latar belakang, mengukur kepadatan cincin kayu, "
            "dan menyesuaikan sensitivitas pelacakan tanpa perlu pengaturan manual."
        )

    #  MODE 1: UPLOAD GAMBAR LANGSUNG 
    if input_mode == "Upload Gambar Langsung":
        uploaded_files = st.file_uploader(
            "Unggah satu atau beberapa foto penampang kayu (JPG, JPEG, PNG)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True
        )

        if uploaded_files:
            results_list = []

            for idx, up_file in enumerate(uploaded_files):
                file_bytes = np.asarray(bytearray(up_file.read()), dtype=np.uint8)
                img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                try:
                    with st.spinner(f"Menganalisis {up_file.name}..."):
                        img_rgb, result_img, pred_rings, usia = detect_tree_rings_from_array(img_bgr)

                    results_list.append({
                        "Nama File": up_file.name,
                        "Jumlah Lingkaran": pred_rings,
                        "Estimasi Usia (Tahun)": usia
                    })

                    with st.container():
                        st.markdown(f"### 📄 {up_file.name}")
                        m1, m2, m3 = st.columns([1, 1, 1])
                        m1.metric("Jumlah Lingkaran", f"{pred_rings} cincin")
                        m2.metric("Estimasi Usia Pohon", f"{usia} tahun")
                        with m3:
                            st.write("")
                            st.download_button(
                                label="⬇️ Unduh Gambar Hasil",
                                data=encode_image_to_bytes(result_img),
                                file_name=f"hasil_{os.path.splitext(up_file.name)[0]}.png",
                                mime="image/png",
                                key=f"dl_{idx}_{up_file.name}"
                            )

                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(img_rgb, caption="Gambar Asli", use_container_width=True)
                        with col2:
                            st.image(
                                result_img,
                                caption=f"Hasil Deteksi: {pred_rings} lingkaran ({usia} tahun)",
                                use_container_width=True
                            )
                        st.divider()

                except Exception as e:
                    st.error(f"Gagal memproses {up_file.name}: {e}")

            if results_list:
                st.subheader("📊 Tabel Rekapitulasi Hasil")
                df_res = pd.DataFrame(results_list)
                st.dataframe(df_res, use_container_width=True)

                csv_data = df_res.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Tabel Hasil (.CSV)",
                    data=csv_data,
                    file_name="hasil_deteksi_lingkaran_pohon.csv",
                    mime="text/csv"
                )
        else:
            st.info("Silakan unggah foto penampang batang pohon di atas untuk langsung melihat hasilnya.")

    #  MODE 2: UPLOAD DATASET (.ZIP) 
    else:
        zip_file = st.file_uploader("Unggah file Dataset (.ZIP)", type=["zip"])

        if zip_file is not None:
            if "extracted_dir" not in st.session_state or st.session_state.get("last_zip_name") != zip_file.name:
                temp_dir = tempfile.mkdtemp(prefix="tree_rings_")
                zip_path = os.path.join(temp_dir, zip_file.name)
                with open(zip_path, "wb") as f:
                    f.write(zip_file.getbuffer())

                extract_path = os.path.join(temp_dir, "extracted")
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(extract_path)

                st.session_state["extracted_dir"] = extract_path
                st.session_state["last_zip_name"] = zip_file.name

            raw_dir = st.session_state["extracted_dir"]
            img_nobg_dir = find_folder(raw_dir, "images_no_background")
            img_orig_dir = find_folder(raw_dir, "images")
            ann_dir = find_folder(raw_dir, "annotations")
            google_dir = find_folder(raw_dir, "data_uji_google")

            list_asli = get_images_in_dir(img_orig_dir)
            list_nobg = get_images_in_dir(img_nobg_dir)
            list_google = get_images_in_dir(google_dir)

            c1, c2, c3 = st.columns(3)
            c1.metric("Data Latih (Latar Asli)", f"{len(list_asli)} gambar")
            c2.metric("Data Latih (No Background)", f"{len(list_nobg)} gambar")
            c3.metric("Data Uji Google", f"{len(list_google)} gambar")

            tab_latih, tab_uji = st.tabs(["🧪 Data Latih (Campuran)", "🔍 Data Uji (Google)"])

            with tab_latih:
                max_pair = min(len(list_asli), len(list_nobg))
                if max_pair > 0:
                    jml_sampel = st.slider(
                        "Jumlah Sampel per Jenis (Latar Asli & No Background):",
                        1, max_pair, min(3, max_pair)
                    )
                    asli_shuf = list_asli.copy()
                    nobg_shuf = list_nobg.copy()
                    random.seed(42)
                    random.shuffle(asli_shuf)
                    random.shuffle(nobg_shuf)

                    sampel_campuran = []
                    for a, b in zip(asli_shuf[:jml_sampel], nobg_shuf[:jml_sampel]):
                        sampel_campuran.append((a, "Latar Asli"))
                        sampel_campuran.append((b, "No Background"))

                    res_latih = []
                    for img_path, jenis in sampel_campuran:
                        fname = os.path.basename(img_path)
                        img_bgr = cv2.imread(img_path)
                        gt_rings = get_ground_truth_rings(img_path, ann_dir)
                        try:
                            img_rgb, result_img, pred_rings, usia = detect_tree_rings_from_array(img_bgr)
                            row = {
                                "Nama File": fname,
                                "Jenis": jenis,
                                "Jumlah Lingkaran": pred_rings,
                                "Estimasi Usia (Tahun)": usia
                            }
                            if gt_rings is not None:
                                row["Ground Truth (JSON)"] = gt_rings
                            res_latih.append(row)

                            col1, col2 = st.columns(2)
                            col1.image(img_rgb, caption=f"[{jenis}] {fname}", use_container_width=True)
                            col2.image(
                                result_img,
                                caption=f"Terdeteksi: {pred_rings} lingkaran ({usia} tahun)",
                                use_container_width=True
                            )
                            st.divider()
                        except Exception as e:
                            st.error(f"Gagal memproses {fname}: {e}")

                    if res_latih:
                        st.subheader("Tabel Hasil Data Latih")
                        st.dataframe(pd.DataFrame(res_latih), use_container_width=True)
                else:
                    st.warning("Folder `images` atau `images_no_background` tidak ditemukan di dalam ZIP.")

            with tab_uji:
                if list_google:
                    res_uji = []
                    for img_path in list_google:
                        fname = os.path.basename(img_path)
                        img_bgr = cv2.imread(img_path)
                        try:
                            img_rgb, result_img, pred_rings, usia = detect_tree_rings_from_array(img_bgr)
                            res_uji.append({
                                "Nama File": fname,
                                "Jumlah Lingkaran": pred_rings,
                                "Estimasi Usia (Tahun)": usia
                            })

                            col1, col2 = st.columns(2)
                            col1.image(img_rgb, caption=f"[Data Uji] {fname}", use_container_width=True)
                            col2.image(
                                result_img,
                                caption=f"Terdeteksi: {pred_rings} lingkaran ({usia} tahun)",
                                use_container_width=True
                            )
                            st.divider()
                        except Exception as e:
                            st.error(f"Gagal memproses {fname}: {e}")

                    if res_uji:
                        st.subheader("Tabel Hasil Akhir Data Uji Google")
                        df_uji = pd.DataFrame(res_uji)
                        st.dataframe(df_uji, use_container_width=True)
                        st.download_button(
                            label="📥 Download Hasil Data Uji (.CSV)",
                            data=df_uji.to_csv(index=False).encode("utf-8"),
                            file_name="hasil_data_uji_google.csv",
                            mime="text/csv"
                        )
                else:
                    st.warning("Folder `data_uji_google` tidak ditemukan di dalam ZIP.")


if __name__ == "__main__":
    main()