import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import av
import sqlite3
import pandas as pd
import requests
import plotly.express as px
import speech_recognition as sr
from datetime import datetime, timedelta
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# --- 1. SETUP SESSION STATE (AGAR INTRO PAGE JALAN) ---
if 'intro_done' not in st.session_state:
    st.session_state.intro_done = False

# --- 2. INTRO PAGE (HALAMAN PEMBUKA) ---
if not st.session_state.intro_done:
    st.set_page_config(page_title="Intro - Tugas AI", layout="centered")
    
    st.title("👋 Halo!")
    st.markdown("""
    ### Selamat Datang di AI Dashboard
    Aplikasi Streamlit ini berisi **3 kumpulan AI Tools** yang dibentuk untuk menyelesaikan tugas mengenai pembuatan program **Mini AI**.
    
    **Dibuat oleh:**
    *   **Nama:** Yesaya Alvin K
    *   **NIM:** 632025053
    
    ---
    **🛠️ Fitur AI yang tersedia:**
    1.  **🎨 AI Air Canvas:** Menggambar di udara menggunakan Computer Vision (Hand Tracking).
    2.  **🌦️ Smart Weather:** Dashboard cuaca real-time & prediksi berbasis API.
    3.  **🎙️ Live Voice Transcriber:** Mengubah suara menjadi teks secara continuous (Real-time).
    """)
    
    if st.button("🚀 MULAI APLIKASI", type="primary", use_container_width=True):
        st.session_state.intro_done = True
        st.rerun()
    
    # Stop eksekusi kode di bawahnya kalau belum klik Mulai
    st.stop()

# ==============================================================================
# MAIN DASHBOARD (HANYA MUNCUL SETELAH KLIK MULAI)
# ==============================================================================

# --- SETUP DATABASE ---
conn = sqlite3.connect('database_ai_v2.db', check_same_thread=False)
c = conn.cursor()
c.execute('CREATE TABLE IF NOT EXISTS galeri (waktu TEXT, event TEXT, status TEXT)')
conn.commit()

def simpan_ke_db(event, status):
    waktu = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO galeri VALUES (?, ?, ?)", (waktu, event, status))
    conn.commit()
    return waktu

# --- KONFIGURASI HALAMAN UTAMA ---
st.set_page_config(layout="wide", page_title="Super AI Dashboard")
st.title("🤖 Artificial Intelligence Dashboard (3-in-1)")
st.caption("Dibuat oleh: Yesaya Alvin K (632025053)")

# --- TAB SETUP ---
tab1, tab2, tab3 = st.tabs([
    "🎨 AI Air Canvas", 
    "🌦️ Smart Weather", 
    "🎙️ Live Voice Transcriber"
])

# ==========================================
# TAB 1: AI AIR CANVAS (AMAN - TIDAK DIUBAH)
# ==========================================
with tab1:
    col_kiri, col_kanan = st.columns([2, 1])
    with col_kanan:
        st.header("🎮 Panel Kontrol")
        st.info("""
        **Petunjuk Kontrol:**
        1. ☝️ Telunjuk Naik = ✏️ **MENGGAMBAR**
        2. ✊ Kepal Tangan = 🛑 **STOP**
        3. ✌️ Peace / ✋ Buka = 🛑 **STOP**
        """)
        if st.button("💾 Simpan Log Aktivitas"):
            waktu = simpan_ke_db("User Save Button", "Recorded")
            st.success(f"Log berhasil disimpan: {waktu}")

        st.write("---")
        st.subheader("📂 Database Viewer")
        if st.checkbox("Tampilkan Data Log", value=True):
            try:
                df = pd.read_sql_query("SELECT * FROM galeri ORDER BY waktu DESC", conn)
                st.dataframe(df, use_container_width=True)
            except:
                st.write("Belum ada data.")

    with col_kiri:
        st.header("Kamera (Live)")
        class CanvasProcessor(VideoProcessorBase):
            def __init__(self):
                self.mp_hands = mp.solutions.hands
                self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
                self.canvas = None
                self.prev_x, self.prev_y = 0, 0
            def recv(self, frame):
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
        webrtc_streamer(key="air-canvas-final", video_processor_factory=CanvasProcessor, rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}), media_stream_constraints={"video": True, "audio": False})

# ==========================================
# TAB 2: SMART WEATHER (FIX JAM LOMPAT)
# ==========================================
with tab2:
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.header("🌦️ Smart Weather: UKSW Salatiga")
    with col_h2:
        if st.button("🔄 Refresh Data"):
            st.rerun()
        wib_now = datetime.utcnow() + timedelta(hours=7)
        st.caption(f"Last Updated (WIB): {wib_now.strftime('%H:%M:%S')}")

    LAT, LON = -7.3305, 110.5084
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={LAT}&longitude={LON}&current=temperature_2m,precipitation,weather_code,wind_speed_10m&hourly=temperature_2m,precipitation,wind_speed_10m&daily=temperature_2m_max,precipitation_sum,wind_speed_10m_max&timezone=Asia%2FBangkok"
        response = requests.get(url)
        data = response.json()
        
        # 1. AI ANALISIS
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

        # 2. MONITORING GRAFIK
        st.subheader("📈 Monitoring Grafik (Interactive)")
        pilihan_waktu = st.radio("Pilih Interval Data:", ["Per Jam", "Harian"], horizontal=True)
        df_show = pd.DataFrame()
        
        if pilihan_waktu == "Per Jam":
            d = data['hourly']
            df_show = pd.DataFrame({"Waktu": d['time'], "Suhu": d['temperature_2m'], "Hujan": d['precipitation'], "Angin": d['wind_speed_10m']})
            df_show['Waktu'] = pd.to_datetime(df_show['Waktu'])
            now = datetime.now()
            df_show = df_show[(df_show['Waktu'] >= now) & (df_show['Waktu'] <= now + timedelta(hours=24))]
        else: # Harian
            d = data['daily']
            df_show = pd.DataFrame({"Waktu": d['time'], "Suhu": d['temperature_2m_max'], "Hujan": d['precipitation_sum'], "Angin": d['wind_speed_10m_max']})
            df_show['Waktu'] = pd.to_datetime(df_show['Waktu'])

        def apply_layout_fix(fig, interval):
            if interval == "Per Jam":
                # FORCE EVERY HOUR TICK (TIDAK LOMPAT)
                fig.update_xaxes(
                    dtick=3600000, # 1 Jam dalam milidetik
                    tickformat="%H:%M",
                    tickangle=-45,
                    tickmode='linear' # Paksa linear agar tidak auto-skip
                )
            elif interval == "Harian":
                fig.update_xaxes(tickformat="%d-%b", tickangle=-45)
            return fig

        c_g1, c_g2, c_g3 = st.columns(3)
        with c_g1:
            fig1 = px.line(df_show, x="Waktu", y="Suhu", title=f"Grafik Suhu ({pilihan_waktu})")
            fig1.update_traces(line_color='#FF4B4B')
            st.plotly_chart(apply_layout_fix(fig1, pilihan_waktu), use_container_width=True)
        with c_g2:
            fig2 = px.bar(df_show, x="Waktu", y="Hujan", title=f"Grafik Hujan ({pilihan_waktu})")
            fig2.update_traces(marker_color='#00BFFF')
            st.plotly_chart(apply_layout_fix(fig2, pilihan_waktu), use_container_width=True)
        with c_g3:
            fig3 = px.area(df_show, x="Waktu", y="Angin", title=f"Grafik Angin ({pilihan_waktu})")
            fig3.update_traces(line_color='#5D3FD3')
            st.plotly_chart(apply_layout_fix(fig3, pilihan_waktu), use_container_width=True)

        st.write("---")
        
        # 3. PREDIKSI
        st.subheader("📅 Prediksi Jangka Panjang (7 Hari)")
        d_pred = data['daily']
        df_pred = pd.DataFrame({"Tanggal": d_pred['time'], "Suhu Max": d_pred['temperature_2m_max'], "Total Hujan": d_pred['precipitation_sum'], "Angin Max": d_pred['wind_speed_10m_max']})
        
        c_p1, c_p2, c_p3 = st.columns(3)
        with c_p1: st.plotly_chart(px.line(df_pred, x="Tanggal", y="Suhu Max", markers=True, title="Prediksi Suhu Max").update_traces(line_color='orange'), use_container_width=True)
        with c_p2: st.plotly_chart(px.bar(df_pred, x="Tanggal", y="Total Hujan", title="Prediksi Curah Hujan").update_traces(marker_color='blue'), use_container_width=True)
        with c_p3: st.plotly_chart(px.line(df_pred, x="Tanggal", y="Angin Max", markers=True, title="Prediksi Angin Kencang").update_traces(line_color='purple'), use_container_width=True)
    except:
        st.error("Gagal koneksi API.")

# ==========================================
# TAB 3: LIVE VOICE (TOMBOL PINTAR & DB AUTO)
# ==========================================
with tab3:
    st.header("🎙️ Live Voice to Text")
    st.caption("Klik Mulai, sistem akan mendengarkan terus menerus (Indo/Eng Mix). Klik Stop untuk selesai.")
    
    col_kiri, col_kanan = st.columns([1, 2])
    
    # Init State
    if 'recording' not in st.session_state: st.session_state.recording = False
    if 'full_transcript' not in st.session_state: st.session_state.full_transcript = ""

    with col_kiri:
        # LOGIKA TOMBOL GANTI-GANTI
        if not st.session_state.recording:
            # Tombol Hijau kalau belum rekam
            if st.button("▶️ MULAI MENDENGARKAN", type="primary", use_container_width=True):
                st.session_state.recording = True
                st.rerun()
        else:
            # Tombol Merah kalau sedang rekam
            if st.button("⏹️ STOP (SEDANG MENDENGARKAN)", type="secondary", use_container_width=True):
                st.session_state.recording = False
                st.rerun()
        
        st.write("---")
        if st.session_state.recording:
            st.markdown("### 👂 STATUS: MENDENGARKAN...")
            st.spinner("Jangan diam terlalu lama...")
        else:
            st.markdown("### 💤 STATUS: STANDBY")

    with col_kanan:
        transkrip_box = st.empty()
        
        if st.session_state.recording:
            r = sr.Recognizer()
            r.pause_threshold = 1.0 
            r.energy_threshold = 300 
            
            try:
                with sr.Microphone() as source:
                    r.adjust_for_ambient_noise(source, duration=0.5)
                    
                    while st.session_state.recording:
                        transkrip_box.markdown(f"**🔴 Merekam...**\n\n{st.session_state.full_transcript}")
                        try:
                            audio = r.listen(source, timeout=None, phrase_time_limit=15)
                            transkrip_box.markdown(f"**🧠 Memproses suara...**\n\n{st.session_state.full_transcript}")
                            
                            text = r.recognize_google(audio, language="id-ID")
                            
                            timestamp = datetime.now().strftime("%H:%M:%S")
                            st.session_state.full_transcript += f"[{timestamp}] {text}\n"
                            
                            transkrip_box.text_area("Live Transkrip:", value=st.session_state.full_transcript, height=400)
                            
                            # SIMPAN DATABASE OTOMATIS
                            simpan_ke_db("Voice Input", text)
                            
                        except sr.WaitTimeoutError: pass 
                        except sr.UnknownValueError: pass
                        except Exception as e: break
            except Exception as e:
                st.error("Gagal akses Microphone.")
        else:
            transkrip_box.text_area("Hasil Akhir:", value=st.session_state.full_transcript, height=400)