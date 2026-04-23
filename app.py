import streamlit as st
from PIL import Image
import numpy as np
import tempfile
import os
import io
from scipy.io import wavfile

st.set_page_config(page_title="Sonificazione dei Colori by loop507", layout="centered")
st.markdown("<h1>🎨🎵 Sonificazione dei Colori <span style='font-size:0.5em;'>by loop507</span></h1>", unsafe_allow_html=True)
st.write("Carica una o più foto. Il suono è generato automaticamente dall'immagine.")

# ─── COSTANTI ───────────────────────────────────────────────────────────────
SAMPLE_RATE = 44100
MIN_FREQ    = 80.0
MAX_FREQ    = 2000.0

# ─── ANALISI IMMAGINE (vettorializzata numpy) ────────────────────────────────
@st.cache_data(show_spinner="Analizzando i colori...", persist=True)
def analyze_image(image_bytes):
    """
    Analizza un'immagine e restituisce una mappa spaziale dei colori.
    Tutto vettorializzato — nessun loop Python sui pixel.
    Ritorna:
      pan_map    : array (N,) — posizione stereo per ogni zona [-1=sin, +1=dex]
      freq_map   : array (N,) — frequenza Hz per ogni zona
      amp_map    : array (N,) — ampiezza per ogni zona [0-1]
      complexity : float      — complessità cromatica [0-1]
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    # Ridimensiona per performance — max 200x200 per l'analisi
    img_small = img.resize((200, 200), Image.LANCZOS)
    arr = np.array(img_small, dtype=np.float32) / 255.0  # (200, 200, 3)

    h, w, _ = arr.shape

    # --- RGB → HSV vettorializzato ---
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin + 1e-8

    # Value (luminosità)
    v = cmax

    # Saturation
    s = np.where(cmax > 1e-8, delta / (cmax + 1e-8), 0.0)

    # Hue
    hue = np.zeros_like(r)
    mask_r = (cmax == r)
    mask_g = (cmax == g)
    mask_b = (cmax == b)
    hue[mask_r] = ((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6
    hue[mask_g] = (b[mask_g] - r[mask_g]) / delta[mask_g] + 2
    hue[mask_b] = (r[mask_b] - g[mask_b]) / delta[mask_b] + 4
    hue = hue / 6.0  # [0, 1]

    # --- Mappa spaziale ---
    # Pan stereo: posizione X del pixel [0=sinistra, 1=destra] → [-1, +1]
    x_positions = np.linspace(-1.0, 1.0, w)
    pan_full = np.tile(x_positions, (h, 1))  # (h, w)

    # Frequenza: hue → frequenza (mappatura circolare)
    freq_full = MIN_FREQ + hue * (MAX_FREQ - MIN_FREQ)

    # Ampiezza: luminosità × saturazione (colori scuri/grigi = meno presenza)
    amp_full = v * (0.3 + 0.7 * s)

    # Dividi in 20 zone orizzontali × 10 verticali = 200 zone
    n_zones_x = 20
    n_zones_y = 10
    pan_map, freq_map, amp_map = [], [], []

    for iy in range(n_zones_y):
        for ix in range(n_zones_x):
            y0 = int(iy * h / n_zones_y)
            y1 = int((iy + 1) * h / n_zones_y)
            x0 = int(ix * w / n_zones_x)
            x1 = int((ix + 1) * w / n_zones_x)

            zone_pan  = pan_full[y0:y1, x0:x1].mean()
            zone_freq = freq_full[y0:y1, x0:x1].mean()
            zone_amp  = amp_full[y0:y1, x0:x1].mean()

            pan_map.append(zone_pan)
            freq_map.append(zone_freq)
            amp_map.append(zone_amp)

    pan_map  = np.array(pan_map)
    freq_map = np.array(freq_map)
    amp_map  = np.array(amp_map)

    # Normalizza ampiezza
    if amp_map.max() > 0:
        amp_map = amp_map / amp_map.max()

    # Complessità cromatica = deviazione standard delle frequenze normalizzata
    complexity = float(np.std(freq_map) / (MAX_FREQ - MIN_FREQ))

    return pan_map, freq_map, amp_map, complexity


def generate_stereo_audio(pan_map, freq_map, amp_map, duration_sec, wave_type="sine"):
    """
    Genera audio stereo da mappe spaziali di colore.
    Pan stereo automatico dalla posizione del colore nell'immagine.
    """
    n_samples = int(SAMPLE_RATE * duration_sec)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False)

    left  = np.zeros(n_samples, dtype=np.float32)
    right = np.zeros(n_samples, dtype=np.float32)

    for pan, freq, amp in zip(pan_map, freq_map, amp_map):
        if amp < 0.01:
            continue

        # Genera onda
        if wave_type == "sine":
            wave = np.sin(2 * np.pi * freq * t).astype(np.float32)
        elif wave_type == "sawtooth":
            wave = (2 * (freq * t % 1) - 1).astype(np.float32)
        else:  # square
            wave = np.sign(np.sin(2 * np.pi * freq * t)).astype(np.float32)

        wave *= amp

        # Pan stereo automatico dalla posizione del colore
        # pan ∈ [-1, +1]: -1 = tutto sinistra, +1 = tutto destra
        gain_l = float(np.sqrt(np.clip(0.5 - pan * 0.5, 0, 1)))
        gain_r = float(np.sqrt(np.clip(0.5 + pan * 0.5, 0, 1)))

        left  += wave * gain_l
        right += wave * gain_r

    # Normalizza
    peak = max(np.abs(left).max(), np.abs(right).max(), 1e-6)
    left  = left  / peak * 0.85
    right = right / peak * 0.85

    # Fade in/out 50ms
    fade = int(SAMPLE_RATE * 0.05)
    env = np.ones(n_samples, dtype=np.float32)
    env[:fade]  = np.linspace(0, 1, fade)
    env[-fade:] = np.linspace(1, 0, fade)
    left  *= env
    right *= env

    return left, right


def crossfade(left_a, right_a, left_b, right_b, fade_samples):
    """Crossfade lineare tra due segmenti stereo."""
    fade_samples = min(fade_samples, len(left_a), len(left_b))
    fade_out = np.linspace(1, 0, fade_samples, dtype=np.float32)
    fade_in  = np.linspace(0, 1, fade_samples, dtype=np.float32)
    left_a[-fade_samples:]  *= fade_out
    right_a[-fade_samples:] *= fade_out
    left_b[:fade_samples]   = left_b[:fade_samples]  * fade_in  + left_a[-fade_samples:]
    right_b[:fade_samples]  = right_b[:fade_samples] * fade_in  + right_a[-fade_samples:]
    return left_b, right_b


def build_report(images_info, total_duration, wave_type):
    lines = ["[STUDIO_SONIFICAZIONE] // VOL_01 // MP3 // COLOR_TO_SOUND",
             f":: MOTORE: sonificazione_colori [v2.0]",
             f":: PROCESSO: Traduzione Cromatica Automatica",
             f":: ONDA: {wave_type}",
             "",
             '"L\'immagine ha parlato. Il suono e\' la sua voce."',
             "",
             "> TECHNICAL LOG SHEET:",
             f"* Foto caricate: {len(images_info)}",
             f"* Durata totale: {total_duration} sec",
             f"* Campionamento: {SAMPLE_RATE} Hz | Stereo | MP3 192kbps",
             "* Mappatura: Posizione X → Pan Stereo | Hue → Frequenza | Luminosita' × Saturazione → Ampiezza",
             ""]

    for i, info in enumerate(images_info):
        lines.append(f"* Foto {i+1} — {info['name']}: complessita' {info['complexity']:.2f} | durata {info['duration']:.1f}s")

    lines += ["",
              "> Regia e Algoritmo: Loop507",
              "",
              "#loop507 #sonificazione #colortosound #synesthesia #generativeart",
              "#experimentalaudio #colormusic #automaticcomposition #brutalistart"]

    return "\n".join(lines)


def wav_to_mp3(wav_path, bitrate="192k"):
    """Converte WAV in MP3 usando ffmpeg diretto."""
    import subprocess
    mp3_path = wav_path.replace(".wav", ".mp3")
    try:
        result = subprocess.run(
            ['ffmpeg', '-i', wav_path, '-b:a', bitrate, '-y', mp3_path],
            capture_output=True
        )
        if result.returncode == 0 and os.path.exists(mp3_path):
            return mp3_path
        else:
            return wav_path
    except Exception:
        return wav_path


# ─── INTERFACCIA ─────────────────────────────────────────────────────────────
if 'audio_ready'   not in st.session_state: st.session_state.audio_ready   = False
if 'audio_path'    not in st.session_state: st.session_state.audio_path    = ""
if 'report_data'   not in st.session_state: st.session_state.report_data   = ""
if 'audio_is_mp3'  not in st.session_state: st.session_state.audio_is_mp3  = False

uploaded_files = st.file_uploader(
    "📸 Carica una o più foto (JPG, PNG)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    if len(uploaded_files) > 10:
        st.warning("Massimo 10 foto. Saranno usate le prime 10.")
        uploaded_files = uploaded_files[:10]

    # Mostra anteprime
    cols = st.columns(min(len(uploaded_files), 5))
    for i, f in enumerate(uploaded_files):
        with cols[i % 5]:
            st.image(f, use_container_width=True, caption=f.name[:12])

st.divider()

col1, col2 = st.columns(2)
with col1:
    total_duration = st.slider("⏱️ Durata totale (secondi)", 5, 240, 30, 5)
with col2:
    wave_type = st.selectbox("🎵 Tipo di onda", ["sine", "sawtooth", "square"],
        format_func=lambda x: {"sine": "Sine — morbida", "sawtooth": "Sawtooth — brillante", "square": "Square — cava"}[x])

if st.button("🎨 Genera Suono", use_container_width=True):
    if not uploaded_files:
        st.warning("Carica almeno una foto.")
    else:
        with st.spinner("Analizzando e generando..."):
            # Analizza tutte le foto
            images_data = []
            for f in uploaded_files:
                file_bytes = f.getvalue()
                pan_map, freq_map, amp_map, complexity = analyze_image(file_bytes)
                images_data.append({
                    "name": f.name,
                    "pan_map": pan_map,
                    "freq_map": freq_map,
                    "amp_map": amp_map,
                    "complexity": complexity,
                    "bytes": file_bytes
                })

            # Distribuisce la durata proporzionalmente alla complessità cromatica
            complexities = np.array([d["complexity"] for d in images_data])
            if complexities.sum() < 1e-6:
                complexities = np.ones(len(images_data))
            weights = complexities / complexities.sum()
            durations = weights * total_duration
            # Durata minima 2 secondi per foto
            durations = np.maximum(durations, 2.0)
            durations = durations / durations.sum() * total_duration

            for i, d in enumerate(images_data):
                d["duration"] = float(durations[i])

            # Genera audio per ogni foto
            fade_samples = int(SAMPLE_RATE * 0.5)  # 0.5s crossfade
            all_left, all_right = [], []

            for d in images_data:
                l, r = generate_stereo_audio(
                    d["pan_map"], d["freq_map"], d["amp_map"],
                    d["duration"], wave_type
                )
                all_left.append(l)
                all_right.append(r)

            # Crossfade tra segmenti
            if len(all_left) > 1:
                for i in range(1, len(all_left)):
                    all_left[i], all_right[i] = crossfade(
                        all_left[i-1], all_right[i-1],
                        all_left[i],   all_right[i],
                        fade_samples
                    )

            final_left  = np.concatenate(all_left)
            final_right = np.concatenate(all_right)

            # Stereo interleaved
            stereo = np.stack([final_left, final_right], axis=1)
            stereo_int16 = (stereo * 32767).astype(np.int16)

            # Salva WAV temporaneo
            wav_path = os.path.join(tempfile.gettempdir(), f"sono_{np.random.randint(0,9999)}.wav")
            wavfile.write(wav_path, SAMPLE_RATE, stereo_int16)

            # Converti in MP3
            final_path = wav_to_mp3(wav_path)
            is_mp3 = final_path.endswith(".mp3")

            # Report
            report = build_report(images_data, total_duration, wave_type)

            st.session_state.audio_path   = final_path
            st.session_state.report_data  = report
            st.session_state.audio_ready  = True
            st.session_state.audio_is_mp3 = is_mp3

            try: os.unlink(wav_path)
            except: pass

# ─── RISULTATI PERSISTENTI ───────────────────────────────────────────────────
if st.session_state.audio_ready:
    st.divider()
    st.markdown("### 🎧 Ascolta")
    if os.path.exists(st.session_state.audio_path):
        fmt = "audio/mp3" if st.session_state.audio_is_mp3 else "audio/wav"
        ext = "mp3" if st.session_state.audio_is_mp3 else "wav"
        with open(st.session_state.audio_path, "rb") as af:
            audio_bytes = af.read()
        st.audio(audio_bytes, format=fmt)

        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                f"⬇️ Scarica ({ext.upper()})",
                data=audio_bytes,
                file_name=f"sonificazione.{ext}",
                mime=fmt,
                key="down_audio"
            )
        with c2:
            st.download_button(
                "📄 Scarica Report",
                data=st.session_state.report_data,
                file_name="report_sonificazione.txt",
                key="down_report"
            )

        st.text_area("📄 REPORT", st.session_state.report_data, height=280)

st.divider()
st.markdown("🎨🎵 **Sonificazione dei Colori** by Loop507 — *L'immagine ha parlato. Il suono è la sua voce.*")
