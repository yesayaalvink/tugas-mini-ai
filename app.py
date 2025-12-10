import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import av
import sqlite3
import pandas as pd
import requests
import plotly.express as px
import logging # BUAT DEBUGGING
from datetime import datetime, timedelta
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode

# --- SETUP LOGGING BIAR KETAUAN ERRORNYA ---
logger = logging.getLogger(__name__)

# --- 1. SETUP SESSION STATE ---
if 'intro_done' not in st.session_state:
    st.session_state.intro_done = False

# --- 2. INTRO PAGE ---
if not st.session_state.intro_done:
    st.set_page_config(page_title="Intro - Tugas AI", layout="centered")
    
    st.title("👋 Halo!")
    st.markdown("""
    ### Selamat Datang di AI Dashboard
    Aplikasi ini adalah **2-in-1 AI Tools** untuk tugas **Mini AI**.
    
    **Dibuat oleh:**
    *   **Nama:** Yesaya Alvin K
    *   **NIM:** 632025053
    
    ---
    **🛠️ Fitur AI:**
    1.  **🎨 AI Air Canvas:** Menggambar di udara (Computer Vision).
    2.  **🌦️ Smart Weather:** Analisis & Prediksi Cuaca Real-time.
    """)
    
    if st.button("🚀 MULAI APLIKASI", type="primary", use_container_width=True):
        st.session_state.intro_done = True
        st.rerun()
    st.stop()

# ==============================================================================
# MAIN DASHBOARD
# ==============================================================================

conn = sqlite3.connect('database_ai_v2.db', check_same_thread=False)
c = conn.cursor()
c.execute('CREATE TABLE IF NOT EXISTS galeri (waktu TEXT, event TEXT, status TEXT)')
conn.commit()

def get_wib_now():
    return datetime.utcnow() + timedelta(hours=7)

def simpan_ke_db(event, status):
    waktu = get_wib_now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO galeri VALUES (?, ?, ?)", (waktu, event, status))
    conn.commit()
    return waktu

st.set_page_config(layout="wide", page_title="Super AI Dashboard")
st.title("🤖 Artificial Intelligence Dashboard (2-in-1)")
st.caption("Dibuat oleh: Yesaya Alvin K (632025053)")

tab1, tab2 = st.tabs(["🎨 AI Air Canvas", "🌦️ Smart Weather"])

# ==========================================
# TAB 1: AI AIR CANVAS (DEBUG MODE ADDED)
# ==========================================
with tab1:
    # PESAN KHUSUS BUAT DOSEN / USER
    st.warning("""
    ⚠️ **Rekomendasi Perangkat:**
    1.  **Wajib menggunakan Laptop/PC** (Windows/Mac).
    2.  Browser **Google Chrome** atau **Edge** Terbaru.
    3.  Jika loading terus, **Matikan VPN** atau **Antivirus** sementara.
    4.  Fitur ini mungkin tidak berjalan di HP/Tablet karena batasan sistem operasi.
    """)

    col_kiri, col_kanan = st.columns([2, 1])
    with col_kanan:
        st.header("🎮 Panel Kontrol")
        st.markdown("""
        **Petunjuk:**
        1. ☝️ Telunjuk Naik = ✏️ **MENGGAMBAR**
        2. ✊ Kepal Tangan = 🛑 **STOP**
        3. ✌️ Peace / ✋ Buka = 🛑 **STOP**
        """)
        if st.button("💾 Simpan Log Aktivitas"):
            waktu = simpan_ke_db("User Save Button", "Recorded")
            st.success(f"Log berhasil disimpan: {waktu}")

        st.write("---")
        
        # --- FITUR DEBUGGING ---
        with st.expander("🛠️ Mode Teknisi (Debug)"):
            st.write("Jika kamera tidak muncul, cek status di bawah:")
            debug_status = st.empty()
            st.code("""
            Kemungkinan Penyebab Error:
            1. Browser memblokir akses kamera (Cek ikon gembok di URL).
            2. Jaringan WiFi Kampus/Kantor memblokir WebRTC.
            3. Konflik dengan OBS/Virtual Camera.
            """)

    with col_kiri:
        st.header("Kamera (Live)")
        
        class CanvasProcessor(VideoProcessorBase):
            def __init__(self):
                self.mp_hands = mp.solutions.hands
                self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
                self.canvas = None
                self.prev_x, self.prev_y = 0, 0
                
            def recv(self, frame):
                try:
                    img = frame.to_ndarray(format="bgr24")
                    img = cv2.flip(img, 1) 
                    h, w, c = img.shape
                    if self.canvas is None: self.canvas = np.zeros((h, w, 3), dtype=np.uint8)
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    result = self.hands.process(img_rgb)
                    hand_detected = False
                    
                    if result.multi_hand_landmarks:
                        for hand_lms in result.multi_hand_landmarks:
                            hand_detected = True
                            lm_list = []
                            for id, lm in enumerate(hand_lms.landmark):
                                cx, cy = int(lm.x * w), int(lm.y * h)
                                lm_list.append([id, cx, cy])
                            if len(lm_list) != 0:
                                x_index, y_index = lm_list[8][1], lm_list[8][2]
                                index_up = lm_list[8][2] < lm_list[6][2]
                                middle_up = lm_list[12][2] < lm_list[10][2]
                                mode_gambar = index_up and not middle_up
                                if mode_gambar:
                                    cv2.circle(img, (x_index, y_index), 15, (0, 0, 255), cv2.FILLED)
                                    if self.prev_x == 0 and self.prev_y == 0: self.prev_x, self.prev_y = x_index, y_index
                                    cv2.line(self.canvas, (self.prev_x, self.prev_y), (x_index, y_index), (255, 0, 255), 5)
                                    self.prev_x, self.prev_y = x_index, y_index
                                else:
                                    self.prev_x, self.prev_y = 0, 0
                                    cv2.circle(img, (x_index, y_index), 15, (0, 255, 0), cv2.FILLED)
                                    msg = "STOP"
                                    if index_up and middle_up: msg = "PEACE / BUKA"
                                    if not index_up: msg = "KEPAL"
                                    cv2.putText(img, msg, (x_index, y_index-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    if not hand_detected: self.prev_x, self.prev_y = 0, 0
                    
                    img_gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
                    _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
                    img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
                    img = cv2.bitwise_and(img, img_inv)
                    img = cv2.bitwise_or(img, self.canvas)
                    return av.VideoFrame.from_ndarray(img, format="bgr24")
                except Exception as e:
                    logger.error(f"Error processing frame: {e}")
                    return frame

        # CONFIG JARINGAN STABIL
        rtc_config = {
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
                {"urls": ["stun:global.stun.twilio.com:3478"]},
            ]
        }

        # STREAMER DENGAN CALLBACK STATE
        webrtc_ctx = webrtc_streamer(
            key="air-canvas-debug",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_config,
            video_processor_factory=CanvasProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        # INDIKATOR DEBUG
        if webrtc_ctx.state.playing:
            debug_status.success("✅ Status: KAMERA AKTIF & TERHUBUNG")
        elif webrtc_ctx.state.signaling:
            debug_status.warning("⏳ Status: SEDANG MENGHUBUNGKAN (Handshake)...")
        else:
            debug_status.error("❌ Status: KAMERA MATI / TERBLOKIR")

# ==========================================
# TAB 2: SMART WEATHER
# ==========================================
with tab2:
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.header("🌦️ Smart Weather: UKSW Salatiga")
    with col_h2:
        if st.button("🔄 Refresh Data"):
            st.rerun()
        st.caption(f"Last Updated (WIB): {get_wib_now().strftime('%H:%M:%S')}")

    LAT, LON = -7.3305, 110.5084
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={LAT}&longitude={LON}&current=temperature_2m,precipitation,weather_code,wind_speed_10m&hourly=temperature_2m,precipitation,wind_speed_10m&daily=temperature_2m_max,precipitation_sum,wind_speed_10m_max&timezone=Asia%2FBangkok"
        response = requests.get(url)
        data = response.json()
        
        st.subheader("🤖 Analisis AI Cuaca Real-Time")
        cur = data['current']
        suhu, hujan, angin = cur['temperature_2m'], cur['precipitation'], cur['wind_speed_10m']
        
        pesan, bg_color = "✅ Cuaca Normal.", "success"
        if hujan > 0.5: pesan, bg_color = "⛈️ HUJAN TERDETEKSI: Sedia payung.", "error"
        elif angin > 15: pesan, bg_color = "🌬️ ANGIN KENCANG: Hati-hati.", "warning"
        elif suhu > 30: pesan, bg_color = "🥵 SUHU PANAS: Jaga hidrasi.", "warning"
        
        if bg_color == "error": st.error(pesan)
        elif bg_color == "warning": st.warning(pesan)
        else: st.success(pesan)
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🌡️ Suhu", f"{suhu} °C")
        m2.metric("💧 Hujan", f"{hujan} mm")
        m3.metric("🌬️ Angin", f"{angin} km/h")
        with m4:
            if st.button("💾 Log Cuaca"):
                simpan_ke_db("Weather Log", f"T:{suhu}, R:{hujan}, W:{angin}")
                st.toast("Data Tersimpan!")

        st.write("---")
        st.subheader("📈 Monitoring Grafik (Interactive)")
        pilihan_waktu = st.radio("Pilih Interval Data:", ["Per Jam", "Harian"], horizontal=True)
        df_show = pd.DataFrame()
        
        if pilihan_waktu == "Per Jam":
            d = data['hourly']
            df_show = pd.DataFrame({"Waktu": d['time'], "Suhu": d['temperature_2m'], "Hujan": d['precipitation'], "Angin": d['wind_speed_10m']})
            df_show['Waktu'] = pd.to_datetime(df_show['Waktu'])
            now_wib = get_wib_now() 
            mask = (df_show['Waktu'] >= now_wib.replace(tzinfo=None)) & (df_show['Waktu'] <= (now_wib + timedelta(hours=24)).replace(tzinfo=None))
            df_show = df_show.loc[mask]
        else: 
            d = data['daily']
            df_show = pd.DataFrame({"Waktu": d['time'], "Suhu": d['temperature_2m_max'], "Hujan": d['precipitation_sum'], "Angin": d['wind_speed_10m_max']})
            df_show['Waktu'] = pd.to_datetime(df_show['Waktu'])

        def apply_layout_fix(fig, interval):
            if interval == "Per Jam":
                fig.update_xaxes(dtick=3600000, tickformat="%H:%M", tickangle=-45, tickmode='linear')
            elif interval == "Harian":
                fig.update_xaxes(tickformat="%d-%b", tickangle=-45)
            return fig

        c_g1, c_g2, c_g3 = st.columns(3)
        with c_g1: st.plotly_chart(apply_layout_fix(px.line(df_show, x="Waktu", y="Suhu", title=f"Grafik Suhu").update_traces(line_color='#FF4B4B'), pilihan_waktu), use_container_width=True)
        with c_g2: st.plotly_chart(apply_layout_fix(px.bar(df_show, x="Waktu", y="Hujan", title=f"Grafik Hujan").update_traces(marker_color='#00BFFF'), pilihan_waktu), use_container_width=True)
        with c_g3: st.plotly_chart(apply_layout_fix(px.area(df_show, x="Waktu", y="Angin", title=f"Grafik Angin").update_traces(line_color='#5D3FD3'), pilihan_waktu), use_container_width=True)

        st.write("---")
        st.subheader("📅 Prediksi Jangka Panjang (7 Hari)")
        d_pred = data['daily']
        df_pred = pd.DataFrame({"Tanggal": d_pred['time'], "Suhu Max": d_pred['temperature_2m_max'], "Total Hujan": d_pred['precipitation_sum'], "Angin Max": d_pred['wind_speed_10m_max']})
        
        c_p1, c_p2, c_p3 = st.columns(3)
        with c_p1: st.plotly_chart(px.line(df_pred, x="Tanggal", y="Suhu Max", markers=True).update_traces(line_color='orange'), use_container_width=True)
        with c_p2: st.plotly_chart(px.bar(df_pred, x="Tanggal", y="Total Hujan").update_traces(marker_color='blue'), use_container_width=True)
        with c_p3: st.plotly_chart(px.line(df_pred, x="Tanggal", y="Angin Max").update_traces(line_color='purple'), use_container_width=True)
    except:
        st.error("Gagal koneksi API.")
