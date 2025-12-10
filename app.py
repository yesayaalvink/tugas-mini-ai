import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import av
import sqlite3
import pandas as pd
import requests
import plotly.express as px
from datetime import datetime, timedelta
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode

# --- SETUP DATABASE ---
conn = sqlite3.connect('database_ai_v2.db', check_same_thread=False)
c = conn.cursor()
c.execute('CREATE TABLE IF NOT EXISTS galeri (waktu TEXT, event TEXT, status TEXT)')
conn.commit()

# FUNGSI WAKTU WIB
def get_wib_now():
    return datetime.utcnow() + timedelta(hours=7)

def simpan_ke_db(event, status):
    waktu = get_wib_now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO galeri VALUES (?, ?, ?)", (waktu, event, status))
    conn.commit()
    return waktu

# --- CONFIG PAGE ---
st.set_page_config(layout="wide", page_title="Super AI Dashboard")

# --- INTRO PAGE ---
if 'intro_done' not in st.session_state:
    st.session_state.intro_done = False

if not st.session_state.intro_done:
    st.title("👋 Halo!")
    st.markdown("""
    ### Selamat Datang di AI Dashboard (2-in-1)
    **Dibuat oleh:** Yesaya Alvin K (632025053)
    
    **Fitur:**
    1. 🎨 **AI Air Canvas** (Computer Vision)
    2. 🌦️ **Smart Weather** (Data Analytics)
    """)
    if st.button("🚀 MULAI", type="primary"):
        st.session_state.intro_done = True
        st.rerun()
    st.stop()

# --- MAIN APP ---
st.title("🤖 Artificial Intelligence Dashboard")
st.caption("Dibuat oleh: Yesaya Alvin K (632025053)")

tab1, tab2 = st.tabs(["🎨 AI Air Canvas", "🌦️ Smart Weather"])

# ==========================================
# TAB 1: AI AIR CANVAS (ANTI-CRASH VERSION)
# ==========================================
with tab1:
    st.info("💡 **Info:** Gunakan Laptop untuk performa terbaik. Jika di HP, pastikan izin kamera aktif.")
    
    col_kiri, col_kanan = st.columns([2, 1])
    with col_kanan:
        st.header("🎮 Kontrol")
        st.markdown("""
        **Petunjuk:**
        1. ☝️ Telunjuk = **GAMBAR**
        2. ✊ Kepal = **STOP**
        3. ✌️ Peace = **STOP**
        """)
        if st.button("💾 Simpan Log"):
            simpan_ke_db("User Save", "Success")
            st.success("Tersimpan!")

        st.write("---")
        with st.expander("📂 Lihat Database"):
            try:
                df = pd.read_sql_query("SELECT * FROM galeri ORDER BY waktu DESC", conn)
                st.dataframe(df)
            except: st.write("Kosong")

    with col_kiri:
        st.header("Kamera (Live)")
        
        class CanvasProcessor(VideoProcessorBase):
            def __init__(self):
                self.mp_hands = mp.solutions.hands
                self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
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
                    
                    if result.multi_hand_landmarks:
                        for hand_lms in result.multi_hand_landmarks:
                            lm_list = []
                            for id, lm in enumerate(hand_lms.landmark):
                                cx, cy = int(lm.x * w), int(lm.y * h)
                                lm_list.append([id, cx, cy])
                            if len(lm_list) != 0:
                                x_index, y_index = lm_list[8][1], lm_list[8][2]
                                index_up = lm_list[8][2] < lm_list[6][2]
                                middle_up = lm_list[12][2] < lm_list[10][2]
                                if index_up and not middle_up:
                                    cv2.circle(img, (x_index, y_index), 15, (0, 0, 255), cv2.FILLED)
                                    if self.prev_x == 0 and self.prev_y == 0: self.prev_x, self.prev_y = x_index, y_index
                                    cv2.line(self.canvas, (self.prev_x, self.prev_y), (x_index, y_index), (255, 0, 255), 5)
                                    self.prev_x, self.prev_y = x_index, y_index
                                else:
                                    self.prev_x, self.prev_y = 0, 0
                                    cv2.circle(img, (x_index, y_index), 15, (0, 255, 0), cv2.FILLED)
                                    cv2.putText(img, "STOP", (x_index, y_index-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    img_gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
                    _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
                    img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
                    img = cv2.bitwise_and(img, img_inv)
                    img = cv2.bitwise_or(img, self.canvas)
                    return av.VideoFrame.from_ndarray(img, format="bgr24")
                except: return frame

        # CONFIG JARINGAN
        rtc_config = {
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:global.stun.twilio.com:3478"]},
            ]
        }

        # STREAMER
        ctx = webrtc_streamer(
            key="air-canvas-mobile-fix",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_config,
            video_processor_factory=CanvasProcessor,
            media_stream_constraints={
                "video": {
                    "width": {"min": 480, "ideal": 640},
                    "facingMode": "user" # TRIK SUPAYA HP LANGSUNG PAKAI KAMERA DEPAN
                }, 
                "audio": False
            },
            async_processing=True,
        )

        # STATUS (DENGAN PENGAMAN ANTI ERROR)
        if ctx.state: # <--- INI PENGAMANNYA (Cek dulu ctx.state ada atau ngga)
            if ctx.state.playing:
                st.success("✅ Kamera Aktif")
            elif ctx.state.signaling:
                st.warning("⏳ Menghubungkan ke Server...")
        
# ==========================================
# TAB 2: SMART WEATHER (AMAN)
# ==========================================
with tab2:
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.header("🌦️ Smart Weather: UKSW Salatiga")
    with col_h2:
        if st.button("🔄 Refresh Data"): st.rerun()
        st.caption(f"Last Updated: {get_wib_now().strftime('%H:%M:%S')}")

    LAT, LON = -7.3305, 110.5084
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={LAT}&longitude={LON}&current=temperature_2m,precipitation,weather_code,wind_speed_10m&hourly=temperature_2m,precipitation,wind_speed_10m&daily=temperature_2m_max,precipitation_sum,wind_speed_10m_max&timezone=Asia%2FBangkok"
        response = requests.get(url)
        data = response.json()
        
        cur = data['current']
        suhu, hujan, angin = cur['temperature_2m'], cur['precipitation'], cur['wind_speed_10m']
        
        pesan = "✅ Cuaca Normal."
        if hujan > 0.5: pesan = "⛈️ HUJAN TERDETEKSI"
        elif angin > 15: pesan = "🌬️ ANGIN KENCANG"
        elif suhu > 30: pesan = "🥵 SUHU PANAS"
        
        st.info(f"Analisis AI: {pesan}")
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🌡️ Suhu", f"{suhu} °C")
        m2.metric("💧 Hujan", f"{hujan} mm")
        m3.metric("🌬️ Angin", f"{angin} km/h")
        with m4:
            if st.button("💾 Log Cuaca"):
                simpan_ke_db("Weather Log", f"T:{suhu}, R:{hujan}")
                st.toast("Tersimpan!")

        st.write("---")
        pilihan = st.radio("Interval:", ["Per Jam", "Harian"], horizontal=True)
        df_show = pd.DataFrame()
        
        if pilihan == "Per Jam":
            d = data['hourly']
            df_show = pd.DataFrame({"Waktu": d['time'], "Suhu": d['temperature_2m'], "Hujan": d['precipitation'], "Angin": d['wind_speed_10m']})
            df_show['Waktu'] = pd.to_datetime(df_show['Waktu'])
            now = get_wib_now()
            df_show = df_show[(df_show['Waktu'] >= now.replace(tzinfo=None)) & (df_show['Waktu'] <= (now + timedelta(hours=24)).replace(tzinfo=None))]
        else:
            d = data['daily']
            df_show = pd.DataFrame({"Waktu": d['time'], "Suhu": d['temperature_2m_max'], "Hujan": d['precipitation_sum'], "Angin": d['wind_speed_10m_max']})

        def layout(fig, p):
            if p == "Per Jam": fig.update_xaxes(dtick=3600000, tickformat="%H:%M", tickangle=-45, tickmode='linear')
            else: fig.update_xaxes(tickformat="%d-%b", tickangle=-45)
            return fig

        c1, c2, c3 = st.columns(3)
        with c1: st.plotly_chart(layout(px.line(df_show, x="Waktu", y="Suhu", title="Suhu").update_traces(line_color='red'), pilihan), use_container_width=True)
        with c2: st.plotly_chart(layout(px.bar(df_show, x="Waktu", y="Hujan", title="Hujan").update_traces(marker_color='blue'), pilihan), use_container_width=True)
        with c3: st.plotly_chart(layout(px.area(df_show, x="Waktu", y="Angin", title="Angin").update_traces(line_color='purple'), pilihan), use_container_width=True)

    except: st.error("API Error")
