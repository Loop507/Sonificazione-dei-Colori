import streamlit as st
from PIL import Image
import numpy as np
import scipy.io.wavfile as wavfile
import tempfile
import os
import matplotlib.pyplot as plt
import colorsys # Per convertire RGB a HSV
from scipy import signal # Per onde quadre e a sega
import io # Importa BytesIO per gestire i file caricati

# Configurazione della pagina
st.set_page_config(page_title="🎨🎵 Sonificazione dei Colori by loop507", layout="wide")

st.markdown("<h1>🎨🎵 Sonificazione dei Colori <span style='font-size:0.5em;'>by loop507</span></h1>", unsafe_allow_html=True)
st.write("Carica una o più foto e genera un suono basato sui suoi colori, ora anche come brano sperimentale!")

# --- Mappatura delle Frequenze per Classificazione Colore ---
def get_frequency_for_color_class(h_deg, s_val, v_val):
    """
    Classifica un colore HSV medio e restituisce una frequenza discreta o interpolata.
    h_deg: Hue in gradi (0-360)
    s_val: Saturazione (0.0-1.0)
    v_val: Valore (Luminosità) (0.0-1.0)
    """
    
    # 1. Priorità: Colori Acromatici (Bianco, Nero, Grigio) - Basati su Saturazione e Luminosità
    if s_val < 0.15: # Bassa saturazione indica colori acromatici
        if v_val > 0.9: # Molto luminoso = Bianco
            return 2000 # Frequenza per il Bianco
        elif v_val < 0.1: # Molto scuro = Nero
            return 20 # Frequenza per il Nero
        else: # Luminosità intermedia, bassa saturazione = Grigio
            return 200 # Frequenza per il Grigio
            
    # 2. Priorità: Colori Speciali che dipendono molto da Saturazione/Valore
    
    # Giallo Chiaro (Hue giallo, luminosità molto alta)
    if 45 <= h_deg < 75 and v_val > 0.8:
        return 1950 # Frequenza per il Giallo Chiaro
        
    # Rosa (Hue rosso/magenta, alta luminosità, saturazione media/bassa)
    if (h_deg >= 330 or h_deg < 20) and s_val > 0.15 and v_val > 0.6 and s_val < 0.6:
        return 1150
        
    # Marrone (Hue arancione/rosso, bassa luminosità, saturazione media/alta)
    if (20 <= h_deg < 60 or h_deg >= 340 or h_deg < 20) and s_val > 0.2 and v_val < 0.4:
        return 300
    
    # 3. Interpolazione basata sull'Hue per i Colori Cromatici Standard
    hue_freq_anchors = [
        (0, 700),    # Rosso
        (60, 1900),  # Giallo
        (120, 1300), # Verde
        (180, 1600), # Ciano
        (240, 400),  # Blu
        (300, 1000), # Magenta
        (360, 700)   # Rosso (per chiudere il cerchio)
    ]
    
    if h_deg >= 359.99: 
        h_deg_for_interp = 0
    else:
        h_deg_for_interp = h_deg

    idx1 = 0
    idx2 = 1
    for i in range(len(hue_freq_anchors) - 1):
        if hue_freq_anchors[i][0] <= h_deg_for_interp < hue_freq_anchors[i+1][0]:
            idx1 = i
            idx2 = i + 1
            break
    if h_deg_for_interp == 360: # Special case for 360 degrees
        return 700 # Rosso

    h1, f1 = hue_freq_anchors[idx1]
    h2, f2 = hue_freq_anchors[idx2]

    if (h2 - h1) == 0: 
        return f1
    
    interpolation_factor = (h_deg_for_interp - h1) / (h2 - h1)
    
    interpolated_frequency = f1 + (f2 - f1) * interpolation_factor
    
    return interpolated_frequency

# --- Nuova funzione per ottenere il nome descrittivo del colore per la fascia Hue ---
def get_hue_range_name(hue_start, hue_end):
    # Hue values: 0=Red, 60=Yellow, 120=Green, 180=Cyan, 240=Blue, 300=Magenta, 360=Red
    
    # Calcola il punto medio per una migliore classificazione
    # Gestisce il wrap-around (es. da 330 a 30)
    if hue_start > hue_end: # Range che attraversa 0/360
        mid_hue = (hue_start + hue_end + 360) / 2
        if mid_hue >= 360: mid_hue -= 360
    else:
        mid_hue = (hue_start + hue_end) / 2

    # Aggiustamenti per una migliore classificazione delle fasce
    if 345 <= mid_hue <= 360 or 0 <= mid_hue < 15:
        return "Rosso"
    elif 15 <= mid_hue < 45:
        return "Rosso-Arancio"
    elif 45 <= mid_hue < 75:
        return "Giallo-Arancio"
    elif 75 <= mid_hue < 105:
        return "Giallo-Verde"
    elif 105 <= mid_hue < 135:
        return "Verde"
    elif 135 <= mid_hue < 165:
        return "Verde-Ciano"
    elif 165 <= mid_hue < 195:
        return "Ciano"
    elif 195 <= mid_hue < 225:
        return "Ciano-Blu"
    elif 225 <= mid_hue < 255:
        return "Blu"
    elif 255 <= mid_hue < 285:
        return "Blu-Violetto"
    elif 285 <= mid_hue < 315:
        return "Magenta-Violetto"
    elif 315 <= mid_hue < 345:
        return "Magenta-Rosso"
    else:
        return "Sconosciuto" # Should not happen with valid hue values

# --- Funzione per l'analisi del colore con caching ---
@st.cache_data(show_spinner="Analizzando i colori...", persist=True)
def analyze_image_and_map_to_frequencies(image_bytes_or_slice_obj, n_bins=5):
    """
    Analyzes an image (from bytes or PIL.Image object) and maps its colors to frequencies.
    Uses @st.cache_data for performance.
    
    image_bytes_or_slice_obj: Either raw bytes of an image (from st.file_uploader)
                              or a PIL.Image.Image object (for internal slicing).
    """
    try:
        # Check if input is bytes (from uploaded file) or a PIL Image object
        if isinstance(image_bytes_or_slice_obj, bytes):
            img = Image.open(io.BytesIO(image_bytes_or_slice_obj)).convert('RGB')
        else: # Assume it's a PIL Image object (for slices)
            img = image_bytes_or_slice_obj.convert('RGB')

        img_array = np.array(img)
        
        pixels_flat = img_array.reshape(-1, 3)
        
        bin_rgb_sums = np.zeros((n_bins, 3), dtype=float)
        bin_hsv_sums = np.zeros((n_bins, 3), dtype=float) 
        bin_pixel_counts = np.zeros(n_bins, dtype=int)
        
        temp_bin_edges_for_digitize = np.linspace(0, 360, n_bins + 1)
        
        hue_values_all_pixels = []

        for r, g, b in pixels_flat:
            r_norm, g_norm, b_norm = r/255., g/255., b/255.
            
            h, s, v = colorsys.rgb_to_hsv(r_norm, g_norm, b_norm)
            hue_degrees = h * 360 
            
            hue_values_all_pixels.append(hue_degrees) 

            bin_idx = np.digitize(hue_degrees, temp_bin_edges_for_digitize) - 1
            
            if bin_idx == n_bins: 
                bin_idx = n_bins - 1
            if bin_idx < 0:
                bin_idx = 0

            bin_rgb_sums[bin_idx] += [r, g, b]
            bin_hsv_sums[bin_idx] += [hue_degrees, s, v] 
            bin_pixel_counts[bin_idx] += 1
        
        hist, bin_edges_hist = np.histogram(hue_values_all_pixels, bins=n_bins, range=(0, 360)) 
        
        total_pixels = np.sum(hist)
        hist_normalized = hist / total_pixels if total_pixels > 0 else np.zeros_like(hist)
        
        frequencies_and_weights = []
        all_bin_actual_colors_hex = []

        for i in range(n_bins):
            if bin_pixel_counts[i] > 0:
                avg_rgb = bin_rgb_sums[i] / bin_pixel_counts[i]
                actual_hex_color_for_bin = '#%02x%02x%02x' % (int(avg_rgb[0]), int(avg_rgb[1]), int(avg_rgb[2]))
                
                avg_h_deg = bin_hsv_sums[i][0] / bin_pixel_counts[i]
                avg_s_val = bin_hsv_sums[i][1] / bin_pixel_counts[i]
                avg_v_val = bin_hsv_sums[i][2] / bin_pixel_counts[i] 
                
                frequency = get_frequency_for_color_class(avg_h_deg, avg_s_val, avg_v_val)
            else:
                actual_hex_color_for_bin = "#CCCCCC" 
                frequency = 20 
                avg_v_val = 0 
            
            all_bin_actual_colors_hex.append(actual_hex_color_for_bin)

            if hist_normalized[i] > 0: 
                amplitude_weight = hist_normalized[i]
                
                frequencies_and_weights.append((frequency, amplitude_weight, int(bin_edges_hist[i]), int(bin_edges_hist[i+1]), actual_hex_color_for_bin, avg_v_val))
            
        return frequencies_and_weights, hist_normalized, bin_edges_hist, all_bin_actual_colors_hex
    except Exception as e:
        st.error(f"Errore durante l'analisi dell'immagine: {e}")
        return [], np.array([]), np.array([]), []

# --- Funzione per generare un'onda sinusoidale o un accordo ---
def generate_audio_wave(frequencies_and_weights_with_vval, duration_seconds, sample_rate=44100, 
                       waveform_mode="single", single_waveform_type="sine", 
                       bright_wave="sine", medium_wave="square", dark_wave="sawtooth",
                       fade_in_duration=0, fade_out_duration=0):
    
    if not frequencies_and_weights_with_vval:
        return np.zeros(int(sample_rate * duration_seconds), dtype=np.float32)

    t = np.linspace(0., duration_seconds, int(sample_rate * duration_seconds), endpoint=False)
    
    def get_waveform_function(waveform_type_str):
        if waveform_type_str == "sine":
            return lambda t_arr, freq: np.sin(2 * np.pi * freq * t_arr)
        elif waveform_type_str == "square":
            return lambda t_arr, freq: signal.square(2 * np.pi * freq * t_arr)
        elif waveform_type_str == "sawtooth":
            return lambda t_arr, freq: signal.sawtooth(2 * np.pi * freq * t_arr)
        else: # Default
            return lambda t_arr, freq: np.sin(2 * np.pi * freq * t_arr)

    combined_amplitude = np.zeros_like(t, dtype=np.float32)
    
    total_weight = sum(w for f, w, _, _, _, v_val in frequencies_and_weights_with_vval) 
    if total_weight == 0:
        return np.zeros_like(t, dtype=np.float32) 
    
    for freq, weight, _, _, _, v_val in frequencies_and_weights_with_vval: 
        if freq > 0 and weight > 0:
            current_waveform_func = None
            
            if waveform_mode == "single":
                current_waveform_func = get_waveform_function(single_waveform_type)
            elif waveform_mode == "mixed_all":
                amplitude_component = (weight / total_weight) / 3
                
                combined_amplitude += get_waveform_function("sine")(t, freq) * amplitude_component
                combined_amplitude += get_waveform_function("square")(t, freq) * amplitude_component
                combined_amplitude += get_waveform_function("sawtooth")(t, freq) * amplitude_component
                continue 
            elif waveform_mode == "by_brightness":
                if v_val > 0.7: 
                    current_waveform_func = get_waveform_function(bright_wave)
                elif v_val < 0.3: 
                    current_waveform_func = get_waveform_function(dark_wave)
                else: 
                    current_waveform_func = get_waveform_function(medium_wave)
            
            if current_waveform_func is None:
                 current_waveform_func = get_waveform_function("sine")

            amplitude = current_waveform_func(t, freq) * (weight / total_weight)
            combined_amplitude += amplitude
        
    max_amplitude = np.max(np.abs(combined_amplitude))
    if max_amplitude > 0:
        combined_amplitude /= max_amplitude 
        combined_amplitude *= 0.9 

    if fade_in_duration > 0:
        fade_in_samples = int(fade_in_duration * sample_rate)
        if fade_in_samples > len(combined_amplitude):
            fade_in_samples = len(combined_amplitude) 
        window_in = np.linspace(0., 1., fade_in_samples)
        combined_amplitude[:fade_in_samples] *= window_in

    if fade_out_duration > 0:
        fade_out_samples = int(fade_out_duration * sample_rate)
        if fade_out_samples > len(combined_amplitude):
            fade_out_samples = len(combined_amplitude) 
        window_out = np.linspace(1., 0., fade_out_samples)
        combined_amplitude[-fade_out_samples:] *= window_out
        
    return combined_amplitude

# --- Disegno della mappatura frequenze in una colonna laterale ---
def render_frequency_map_column():
    st.sidebar.markdown("### 🗺️ Mappatura Colore-Frequenza")
    st.sidebar.markdown("""
    <style>
    .color-freq-box {
        display: flex;
        align-items: center;
        margin-bottom: 5px;
        padding: 5px;
        border-radius: 5px;
        color: white; /* Default text color for contrast */
        text-shadow: 1px 1px 2px rgba(0,0,0,0.7);
        font-weight: bold;
    }
    .color-swatch {
        width: 30px;
        height: 30px;
        border: 1px solid rgba(255,255,255,0.5);
        border-radius: 50%;
        margin-right: 10px;
        flex-shrink: 0;
    }
    .freq-label {
        flex-grow: 1;
        text-align: left;
    }
    </style>
    """, unsafe_allow_html=True)

    # Definizione manuale dei colori principali e delle loro frequenze.
    # Questi sono i punti di ancoraggio per l'interpolazione.
    # Aggiungo un 'sort_freq' per facilitare l'ordinamento
    color_map_data = [
        {"name": "Nero", "hex": "#000000", "freq": 20, "sort_freq": 20},
        {"name": "Grigio", "hex": "#808080", "freq": 200, "sort_freq": 200},
        {"name": "Marrone", "hex": "#A52A2A", "freq": 300, "sort_freq": 300}, # Aggiunto per riferimento
        {"name": "Blu (240°)", "hex": "#0000FF", "freq": 400, "sort_freq": 400},
        {"name": "Rosso (0°)", "hex": "#FF0000", "freq": 700, "sort_freq": 700},
        {"name": "Magenta (300°)", "hex": "#FF00FF", "freq": 1000, "sort_freq": 1000},
        {"name": "Rosa", "hex": "#FFC0CB", "freq": 1150, "sort_freq": 1150}, # Aggiunto per riferimento
        {"name": "Verde (120°)", "hex": "#00FF00", "freq": 1300, "sort_freq": 1300},
        {"name": "Ciano (180°)", "hex": "#00FFFF", "freq": 1600, "sort_freq": 1600},
        {"name": "Giallo (60°)", "hex": "#FFFF00", "freq": 1900, "sort_freq": 1900},
        {"name": "Giallo Chiaro", "hex": "#FFFFEE", "freq": 1950, "sort_freq": 1950}, # Aggiunto per riferimento
        {"name": "Bianco", "hex": "#FFFFFF", "freq": 2000, "sort_freq": 2000}
    ]

    # Ordina i dati per frequenza crescente
    color_map_data_sorted = sorted(color_map_data, key=lambda x: x["sort_freq"])

    for item in color_map_data_sorted:
        # Per Bianco, Giallo e Rosa, il testo nero è più leggibile
        text_color = "black" if item["hex"] in ["#FFFFFF", "#FFFF00", "#FFFFEE", "#FFC0CB"] else "white"
        
        st.sidebar.markdown(f"""
        <div class="color-freq-box" style="background-color: {item['hex']}; color: {text_color};">
            <div class="color-swatch" style="background-color: {item['hex']};"></div>
            <div class="freq-label">{item['name']}: {item['freq']} Hz</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.sidebar.markdown(f"""
        <p style="font-size:0.85em; text-align:center;"><i>Le frequenze intermedie sono interpolate.<br>
        Alcuni colori speciali hanno frequenze fisse dedicate.</i></p>
    """, unsafe_allow_html=True)


# --- Sezione principale dell'app ---

# Renderizza la mappatura delle frequenze nella sidebar
render_frequency_map_column()

# Crea due colonne per il layout principale
# MODIFICA QUI: Rende le colonne di uguale larghezza
col_left, col_right = st.columns([0.5, 0.5]) # Left for controls, right for display

with col_left:
    # Abilita il caricamento di più file
    uploaded_files = st.file_uploader("📸 Carica una o più foto (fino a 10)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

    # Inizializza processed_images_data nello stato della sessione se non esiste
    if 'processed_images_data' not in st.session_state:
        st.session_state.processed_images_data = []

    # Processa le immagini caricate immediatamente (se non già processate)
    if uploaded_files:
        # Limita il numero di file a 10
        if len(uploaded_files) > 10:
            st.warning("Hai caricato più di 10 immagini. Saranno processate solo le prime 10.")
            uploaded_files = uploaded_files[:10]
        
        new_processed_data = []
        # Usiamo un hash del contenuto per verificare se il file è lo stesso, non solo il nome
        for uploaded_file in uploaded_files:
            file_bytes = uploaded_file.getvalue()
            file_hash = hash(file_bytes) # Genera un hash del contenuto del file
            
            # Controlla se il file con questo hash è già stato processato
            found_in_session = False
            for stored_data in st.session_state.processed_images_data:
                if stored_data['hash'] == file_hash and stored_data['name'] == uploaded_file.name:
                    new_processed_data.append(stored_data)
                    found_in_session = True
                    break
            
            if not found_in_session:
                # Se il file è nuovo o modificato, analizzalo
                
                # Analisi con la funzione cacheata
                frequencies_and_weights_with_vval, hist_normalized, bin_edges, all_bin_actual_colors_hex = analyze_image_and_map_to_frequencies(
                    file_bytes, n_bins=st.session_state.get('n_bins_input_val', 5) # Usa il valore corrente di n_bins_input
                )
                
                if frequencies_and_weights_with_vval or (hist_normalized.size > 0 and np.sum(hist_normalized) > 0): 
                    new_processed_data.append({
                        'image_bytes': file_bytes, # Store bytes directly for future use with cache
                        'frequencies_and_weights': frequencies_and_weights_with_vval,
                        'name': uploaded_file.name,
                        'hash': file_hash # Store hash for quick comparison
                    })
                else:
                    st.warning(f"Nessuna frequenza generata per l'immagine '{uploaded_file.name}'. Assicurati che l'immagine non sia vuota o danneggiata.")
        
        st.session_state.processed_images_data = new_processed_data

    # --- Impostazioni Sonificazione ---
    st.markdown("### ⚙️ Impostazioni Sonificazione")
    
    sonification_mode = st.radio(
        "Modalità di Sonificazione:",
        ["Singola Immagine (un accordo per immagine)", "Brano Sperimentale (sequenza e mixaggio)", "Brano basato su Scansione Immagine"],
        key="sonification_mode_selector"
    )

    # Update scan_mode_active based on selection
    st.session_state.scan_mode_active = (sonification_mode == "Brano basato su Scansione Immagine")

    duration_input = 2.0 
    sample_rate = 44100 

    # Nuovo selettore per la velocità del brano
    st.markdown("---")
    st.markdown("#### ⏱️ Velocità del Brano")
    tempo_preset = st.selectbox(
        "Scegli la velocità generale del brano:",
        ["Normale", "Lento", "Veloce"],
        key="tempo_preset_selector"
    )

    # Imposta i valori predefiniti in base alla selezione della velocità
    if tempo_preset == "Lento":
        default_tempo_per_beat = 1.5
        default_duration_per_slice = 0.3
    elif tempo_preset == "Veloce":
        default_tempo_per_beat = 0.5
        default_duration_per_slice = 0.08
    else: # Normale
        default_tempo_per_beat = 1.0
        default_duration_per_slice = 0.15
    st.markdown("---")
    
    if sonification_mode == "Singola Immagine (un accordo per immagine)":
        duration_input = st.slider("Durata del suono (secondi)", 0.5, 60.0, 2.0, 0.5) 
        if len(uploaded_files) > 1:
            st.info("Per la modalità 'Singola Immagine', verrà utilizzata solo la prima foto caricata.")

    elif sonification_mode == "Brano Sperimentale (sequenza e mixaggio)":
        st.markdown("#### Impostazioni Brano Sperimentale:")
        col_beats, col_tempo = st.columns(2)
        with col_beats:
            beats_per_image = st.number_input(
                "Battute per Immagine",
                min_value=1,
                max_value=16,
                value=4,
                step=1,
                help="Quante battute audio assegnare a ciascuna foto nella sequenza."
            )
        with col_tempo:
            tempo_per_beat = st.slider(
                "Tempo per Battuta (secondi)",
                min_value=0.1,
                max_value=2.0,
                value=default_tempo_per_beat, # Usa il valore predefinito
                step=0.1,
                help="Durata di ogni singola battuta."
            )
        
        overlap_duration = st.slider(
            "Durata Sovrapposizione (secondi)",
            min_value=0.0,
            max_value=tempo_per_beat * beats_per_image * 0.9, 
            value=min(0.5, tempo_per_beat * beats_per_image * 0.5), 
            step=0.1,
            help="Per quanto tempo il suono di un'immagine si sovrappone a quello successivo. Controlla il grado di mixaggio."
        )

        segment_duration_raw = beats_per_image * tempo_per_beat
        
        if len(uploaded_files) > 0:
            total_estimated_duration = (len(uploaded_files) * segment_duration_raw) - ((len(uploaded_files) - 1) * overlap_duration)
            if total_estimated_duration < segment_duration_raw and len(uploaded_files) > 0: 
                total_estimated_duration = segment_duration_raw
        else:
            total_estimated_duration = 0

        st.info(f"Ogni immagine durerà {segment_duration_raw:.1f} secondi. Durata totale stimata del brano: {total_estimated_duration:.1f} secondi.")

        st.markdown("---")
        # Nuova opzione per applicare la scansione a ciascuna foto nel brano sperimentale
        apply_scan_to_each_photo = st.checkbox(
            "Applica scansione a ciascuna foto (per un brano più dinamico)",
            key="apply_scan_to_each_photo_checkbox"
        )

        if apply_scan_to_each_photo:
            st.markdown("#### Impostazioni Scansione per Foto (nel Brano Sperimentale):")
            num_slices_per_image = st.slider(
                "Numero di Fette (segmenti) per ogni immagine",
                min_value=5,
                max_value=100, # Max slices per individual image scan in this mode
                value=20,
                step=5,
                help="Definisce in quanti segmenti ogni singola immagine verrà divisa per l'analisi interna. Più fette = più dettaglio sonoro per ogni foto."
            )
            scan_direction_per_image = st.selectbox(
                "Direzione di Scansione per ogni immagine:",
                ["Sinistra a Destra", "Alto a Basso"],
                key="scan_direction_per_image_selector",
                help="Scegli la direzione con cui ogni immagine verrà 'letta' internamente."
            )
            st.info(f"Ogni fetta della scansione interna durerà circa {(segment_duration_raw / num_slices_per_image):.2f} secondi.")
        st.markdown("---")

    else: # Brano basato su Scansione Immagine
        st.markdown("#### Impostazioni Scansione Immagine:")
        if len(uploaded_files) > 1:
            st.info("Per la modalità 'Brano basato su Scansione Immagine', verrà utilizzata solo la prima foto caricata.")

        num_slices = st.slider(
            "Numero di Fette (segmenti) dell'immagine",
            min_value=10,
            max_value=200, # Aumentato il massimo per maggiore dettaglio
            value=50,
            step=5,
            help="Definisce in quanti segmenti l'immagine verrà divisa per l'analisi. Più fette = più dettaglio sonoro."
        )
        duration_per_slice = st.slider(
            "Durata di Ogni Fetta (secondi)",
            min_value=0.05, # Minore per suoni veloci
            max_value=1.0,
            value=default_duration_per_slice, # Usa il valore predefinito
            step=0.05,
            help="Quanto durerà il suono generato da ciascuna fetta dell'immagine."
        )
        scan_direction = st.selectbox(
            "Direzione di Scansione:",
            ["Sinistra a Destra", "Alto a Basso"],
            key="scan_direction_selector",
            help="Scegli la direzione con cui l'immagine verrà 'letta' per generare il suono."
        )
        
        total_estimated_duration_scan = num_slices * duration_per_slice
        st.info(f"Durata totale stimata del brano scansionato: {total_estimated_duration_scan:.1f} secondi.")


    col1, col2 = st.columns(2)
    with col1:
        min_freq_input = st.number_input("Frequenza Minima (Hz)", min_value=1, max_value=20000, value=20, key="min_f")
    with col2:
        max_freq_input = st.number_input("Frequenza Massima (Hz)", min_value=1, max_value=20000, value=2000, key="max_f")
    
    # Salva n_bins_input_val nello stato della sessione per la cache
    n_bins_input = st.slider("Numero di Fasce di Colore (Tonalità)", 1, 10, 5, 1, 
                             help="Più fasce = più frequenze diverse nel suono (suono più ricco). Meno fasce = suono più semplice.",
                             key='n_bins_input_val')
    
    if min_freq_input >= max_freq_input:
        st.warning("La Frequenza Minima deve essere inferiore alla Frequenza Massima.")
        
    st.markdown("### 🎶 Tipo di Onda Sonora")
    waveform_selection_mode = st.radio(
        "Come vuoi generare le onde?",
        ["Onda Singola per tutti i Colori", "Onda per Luminosità del Colore"],
        key="waveform_mode_selector"
    )

    selected_single_waveform = "Sine" 
    if waveform_selection_mode == "Onda Singola per tutti i Colori":
        selected_single_waveform = st.radio(
            "Scegli il tipo di onda per tutte le frequenze:",
            ["Sine", "Square", "Sawtooth", "Mixed (Sine + Square + Sawtooth)"],
            key="single_waveform_type"
        )
    
    bright_wave_type = "sine"
    medium_wave_type = "square"
    dark_wave_type = "sawtooth"

    if waveform_selection_mode == "Onda per Luminosità del Colore":
        st.markdown("---")
        st.markdown("#### Assegna Onda per Luminosità:")
        col_bright, col_medium, col_dark = st.columns(3)
        with col_bright:
            bright_wave_type = st.selectbox(
                "Colori Chiari (Luminosità Alta):",
                ["sine", "square", "sawtooth"],
                index=0, 
                key="bright_wave_type"
            )
        with col_medium:
            medium_wave_type = st.selectbox(
                "Colori Medi (Luminosità Media):",
                ["sine", "square", "sawtooth"],
                index=1, 
                key="medium_wave_type"
            )
        with col_dark:
            dark_wave_type = st.selectbox(
                "Colori Scuri (Luminosità Bassa):",
                ["sine", "square", "sawtooth"],
                index=2, 
                key="dark_wave_type"
            )
        st.markdown("---")

    st.markdown("---")
    st.markdown("### 🔍 Come i Colori diventano Suoni:")
    st.markdown(f"""
    Questa applicazione analizza i colori della tua immagine, classificandoli e assegnando una frequenza sonora.
    
    Abbiamo definito frequenze di riferimento per i colori "puri" (es. Rosso, Giallo, Blu) e per i colori acromatici (Nero, Bianco, Grigio).
    Per i colori "mistura" (come l'arancione, che è un mix di rosso e giallo), la frequenza viene **interpolata**
    tra le frequenze dei suoi colori "puri" vicini sulla ruota cromatica, creando una "fusione sonora".
    
    **Nuova Funzionalità: Tipo di Onda Sonora!**
    Oltre alla frequenza (che determina l'altezza del suono), ora puoi scegliere anche il **timbro** (la "qualità" del suono)
    selezionando diversi tipi di onde:
    
    * **Sine Wave (Seno):** Il suono più puro, senza armoniche. Suona morbido e "dolce".
    * **Square Wave (Onda Quadra):** Contiene solo armoniche dispari. Ha un suono più "cavo", simile a un clarinetto o ad alcuni sintetizzatori.
    * **Sawtooth Wave (Onda a Dente di Sega):** Contiene tutte le armoniche. Ha un suono più "brillante" e ricco, simile a un violino o ad un ottoni.
    
    Puoi scegliere un'unica onda per tutti i colori, una miscela di tutte e tre, oppure lasciare che l'applicazione
    assegni l'onda in base alla luminosità del colore, con assegnazioni personalizzabili:
    * **Colori Chiari (Luminosità Alta):** Puoi scegliere il tipo di onda.
    * **Colori Medi (Luminosità Media):** Puoi scegliere il tipo di onda.
    * **Colori Scuri (Luminosità Bassa):** Puoi scegliere il tipo di onda.
    """)
    
    st.markdown("---")


    # --- Pulsante di generazione suono ---
    if st.button("🎵 Genera Suono dai Colori"):
        if not uploaded_files: # Check if files are actually loaded before proceeding
            st.warning("Carica almeno una foto per generare il suono.")
        else:
            with st.spinner("Generando il suono..."):
                final_audio_data = np.array([], dtype=np.float32)
                
                if sonification_mode == "Singola Immagine (un accordo per immagine)":
                    # Use the first uploaded image's processed data
                    if st.session_state.processed_images_data:
                        img_data = st.session_state.processed_images_data[0]
                        frequencies_and_weights_to_use = img_data['frequencies_and_weights']
                        
                        if waveform_selection_mode == "Onda Singola per tutti i Colori":
                            if selected_single_waveform == "Mixed (Sine + Square + Sawtooth)":
                                final_audio_data = generate_audio_wave(frequencies_and_weights_to_use, duration_input, 
                                                                waveform_mode="mixed_all", sample_rate=sample_rate,
                                                                fade_in_duration=0.1, fade_out_duration=0.1) 
                            else:
                                waveform_map_internal = {
                                    "Sine": "sine",
                                    "Square": "square",
                                    "Sawtooth": "sawtooth"
                                }
                                final_audio_data = generate_audio_wave(frequencies_and_weights_to_use, duration_input, 
                                                                waveform_mode="single", 
                                                                single_waveform_type=waveform_map_internal[selected_single_waveform], sample_rate=sample_rate,
                                                                fade_in_duration=0.1, fade_out_duration=0.1) 
                        else: 
                            final_audio_data = generate_audio_wave(frequencies_and_weights_to_use, duration_input, 
                                                            waveform_mode="by_brightness",
                                                            bright_wave=bright_wave_type,
                                                            medium_wave=medium_wave_type,
                                                            dark_wave=dark_wave_type, sample_rate=sample_rate,
                                                            fade_in_duration=0.1, fade_out_duration=0.1) 
                    else:
                        st.error("Nessun dato immagine processato per la modalità Singola Immagine.")

                elif sonification_mode == "Brano Sperimentale (sequenza e mixaggio)":
                    all_raw_audio_segments = []
                    segment_duration = beats_per_image * tempo_per_beat
                    current_fade_in_duration = overlap_duration 

                    for img_data in st.session_state.processed_images_data:
                        frequencies_and_weights_to_use = img_data['frequencies_and_weights']
                        
                        audio_data_segment = None

                        if apply_scan_to_each_photo:
                            # Perform scan for this individual image
                            img_original = Image.open(io.BytesIO(img_data['image_bytes'])).convert('RGB')
                            width, height = img_original.size
                            
                            audio_segments_from_internal_scan = []
                            # Calculate duration per internal slice
                            duration_per_internal_slice = segment_duration / num_slices_per_image
                            fade_duration_per_internal_slice = duration_per_internal_slice * 0.1

                            for i in range(num_slices_per_image):
                                if scan_direction_per_image == "Sinistra a Destra":
                                    slice_width = width // num_slices_per_image
                                    left = i * slice_width
                                    right = (i + 1) * slice_width
                                    if i == num_slices_per_image - 1:
                                        right = width
                                    img_slice = img_original.crop((left, 0, right, height))
                                else: # Alto a Basso
                                    slice_height = height // num_slices_per_image
                                    top = i * slice_height
                                    bottom = (i + 1) * slice_height
                                    if i == num_slices_per_image - 1:
                                        bottom = height
                                    img_slice = img_original.crop((0, top, width, bottom))
                                
                                frequencies_and_weights_slice, _, _, _ = analyze_image_and_map_to_frequencies(img_slice, n_bins_input)
                                
                                internal_segment_audio = None
                                if waveform_selection_mode == "Onda Singola per tutti i Colori":
                                    if selected_single_waveform == "Mixed (Sine + Square + Sawtooth)":
                                        internal_segment_audio = generate_audio_wave(frequencies_and_weights_slice, duration_per_internal_slice, 
                                                                        waveform_mode="mixed_all", sample_rate=sample_rate,
                                                                        fade_in_duration=fade_duration_per_internal_slice, fade_out_duration=fade_duration_per_internal_slice)
                                    else:
                                        waveform_map_internal = {
                                            "Sine": "sine", "Square": "square", "Sawtooth": "sawtooth"
                                        }
                                        internal_segment_audio = generate_audio_wave(frequencies_and_weights_slice, duration_per_internal_slice, 
                                                                        waveform_mode="single", 
                                                                        single_waveform_type=waveform_map_internal[selected_single_waveform], sample_rate=sample_rate,
                                                                        fade_in_duration=fade_duration_per_internal_slice, fade_out_duration=fade_duration_per_internal_slice)
                                else:
                                    internal_segment_audio = generate_audio_wave(frequencies_and_weights_slice, duration_per_internal_slice, 
                                                                    waveform_mode="by_brightness",
                                                                    bright_wave=bright_wave_type,
                                                                    medium_wave=medium_wave_type,
                                                                    dark_wave=dark_wave_type, sample_rate=sample_rate,
                                                                    fade_in_duration=fade_duration_per_internal_slice, fade_out_duration=fade_duration_per_internal_slice)

                                if internal_segment_audio is not None and len(internal_segment_audio) > 0:
                                    audio_segments_from_internal_scan.append(internal_segment_audio)
                                else:
                                    audio_segments_from_internal_scan.append(np.zeros(int(duration_per_internal_slice * sample_rate), dtype=np.float32))
                            
                            if audio_segments_from_internal_scan:
                                audio_data_segment = np.concatenate(audio_segments_from_internal_scan)
                                # Ensure the segment matches the expected total duration
                                expected_samples = int(segment_duration * sample_rate)
                                if len(audio_data_segment) > expected_samples:
                                    audio_data_segment = audio_data_segment[:expected_samples]
                                elif len(audio_data_segment) < expected_samples:
                                    audio_data_segment = np.pad(audio_data_segment, (0, expected_samples - len(audio_data_segment)), 'constant')
                            else:
                                audio_data_segment = np.zeros(int(segment_duration * sample_rate), dtype=np.float32)

                        else: # No internal scan, just a single chord for the image
                            if waveform_selection_mode == "Onda Singola per tutti i Colori":
                                if selected_single_waveform == "Mixed (Sine + Square + Sawtooth)":
                                    audio_data_segment = generate_audio_wave(frequencies_and_weights_to_use, segment_duration, 
                                                                    waveform_mode="mixed_all", sample_rate=sample_rate,
                                                                    fade_in_duration=current_fade_in_duration,
                                                                    fade_out_duration=0) 
                                else:
                                    waveform_map_internal = {
                                        "Sine": "sine",
                                        "Square": "square",
                                        "Sawtooth": "sawtooth"
                                    }
                                    audio_data_segment = generate_audio_wave(frequencies_and_weights_to_use, segment_duration, 
                                                                    waveform_mode="single", 
                                                                    single_waveform_type=waveform_map_internal[selected_single_waveform], sample_rate=sample_rate,
                                                                    fade_in_duration=current_fade_in_duration,
                                                                    fade_out_duration=0) 
                            else: 
                                audio_data_segment = generate_audio_wave(frequencies_and_weights_to_use, segment_duration, 
                                                                waveform_mode="by_brightness",
                                                                bright_wave=bright_wave_type,
                                                                medium_wave=medium_wave_type,
                                                                dark_wave=dark_wave_type, sample_rate=sample_rate,
                                                                fade_in_duration=current_fade_in_duration,
                                                                fade_out_duration=0) 
                        
                        if audio_data_segment is not None:
                            all_raw_audio_segments.append(audio_data_segment)

                    if all_raw_audio_segments:
                        total_duration_calculated = segment_duration + (len(all_raw_audio_segments) - 1) * (segment_duration - overlap_duration)
                        if len(all_raw_audio_segments) == 1: 
                            total_duration_calculated = segment_duration
                        
                        total_samples_needed = int(total_duration_calculated * sample_rate)
                        
                        if total_samples_needed < 0: 
                            total_samples_needed = int(segment_duration * sample_rate) 
                        
                        final_audio_data = np.zeros(total_samples_needed, dtype=np.float32)
                        
                        current_offset_samples = 0
                        offset_per_segment_samples = int((segment_duration - overlap_duration) * sample_rate)
                        
                        if offset_per_segment_samples < 0:
                            offset_per_segment_samples = 0 
                        
                        for i, segment in enumerate(all_raw_audio_segments):
                            segment_samples = len(segment)
                            end_pos = current_offset_samples + segment_samples
                            
                            if end_pos > len(final_audio_data):
                                segment_to_add = segment[:len(final_audio_data) - current_offset_samples]
                            else:
                                segment_to_add = segment
                            
                            if len(segment_to_add) > 0:
                                final_audio_data[current_offset_samples : current_offset_samples + len(segment_to_add)] += segment_to_add
                            
                            current_offset_samples += offset_per_segment_samples
                
                else: # Brano basato su Scansione Immagine (singola immagine)
                    if not st.session_state.processed_images_data: # If no images uploaded at all
                        st.error("Carica una singola immagine per utilizzare la modalità di scansione.")
                    else:
                        # Ensure only the first image is processed for scan mode
                        image_to_scan_bytes = st.session_state.processed_images_data[0]['image_bytes']
                        
                        img_original = Image.open(io.BytesIO(image_to_scan_bytes)).convert('RGB')
                        width, height = img_original.size
                        
                        audio_segments_from_scan = []
                        
                        fade_duration_per_slice = duration_per_slice * 0.1 # 10% of slice duration for fade

                        for i in range(num_slices):
                            if scan_direction == "Sinistra a Destra":
                                slice_width = width // num_slices
                                left = i * slice_width
                                right = (i + 1) * slice_width
                                if i == num_slices - 1: # Ensure last slice covers the rest of the image
                                    right = width
                                img_slice = img_original.crop((left, 0, right, height))
                            else: # Alto a Basso
                                slice_height = height // num_slices
                                top = i * slice_height
                                bottom = (i + 1) * slice_height
                                if i == num_slices - 1: # Ensure last slice covers the rest of the image
                                    bottom = height
                                img_slice = img_original.crop((0, top, width, bottom))
                            
                            # Chiamata a analyze_image_and_map_to_frequencies con l'oggetto PIL.Image direttamente
                            frequencies_and_weights_slice, _, _, _ = analyze_image_and_map_to_frequencies(img_slice, n_bins_input)
                            
                            segment_audio = None
                            if waveform_selection_mode == "Onda Singola per tutti i Colori":
                                if selected_single_waveform == "Mixed (Sine + Square + Sawtooth)":
                                    segment_audio = generate_audio_wave(frequencies_and_weights_slice, duration_per_slice, 
                                                                    waveform_mode="mixed_all", sample_rate=sample_rate,
                                                                    fade_in_duration=fade_duration_per_slice, fade_out_duration=fade_duration_per_slice)
                                else:
                                    waveform_map_internal = {
                                        "Sine": "sine",
                                        "Square": "square",
                                        "Sawtooth": "sawtooth"
                                    }
                                    segment_audio = generate_audio_wave(frequencies_and_weights_slice, duration_per_slice, 
                                                                    waveform_mode="single", 
                                                                    single_waveform_type=waveform_map_internal[selected_single_waveform], sample_rate=sample_rate,
                                                                    fade_in_duration=fade_duration_per_slice, fade_out_duration=fade_duration_per_slice)
                            else:
                                segment_audio = generate_audio_wave(frequencies_and_weights_slice, duration_per_slice, 
                                                                waveform_mode="by_brightness",
                                                                bright_wave=bright_wave_type,
                                                                medium_wave=medium_wave_type,
                                                                dark_wave=dark_wave_type, sample_rate=sample_rate,
                                                                fade_in_duration=fade_duration_per_slice, fade_out_duration=fade_duration_per_slice)

                            if segment_audio is not None and len(segment_audio) > 0:
                                audio_segments_from_scan.append(segment_audio)
                            else: # Add silence if no valid audio to maintain timing
                                audio_segments_from_scan.append(np.zeros(int(duration_per_slice * sample_rate), dtype=np.float32))
                        
                        if audio_segments_from_scan:
                            final_audio_data = np.concatenate(audio_segments_from_scan)
                        else:
                            st.error("❌ Nessun segmento audio generato dalla scansione dell'immagine.")
                            final_audio_data = np.array([], dtype=np.float32) # Ensure final_audio_data is empty
                    
                # Final common audio processing and output
                if final_audio_data.size > 0:
                    max_amplitude = np.max(np.abs(final_audio_data))
                    if max_amplitude > 0:
                        final_audio_data /= max_amplitude 
                        final_audio_data *= 0.8 

                    audio_data_int16 = (final_audio_data * 32767).astype(np.int16) 
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_audio_file:
                        audio_output_path = tmp_audio_file.name
                        wavfile.write(audio_output_path, sample_rate, audio_data_int16) 
                    
                    st.markdown("### Ascolta il tuo Suono:")
                    st.audio(audio_output_path, format='audio/wav')
                    
                    file_name_output = "suono_scansione.wav" if sonification_mode == "Brano basato su Scansione Immagine" else \
                                       ("suono_colore_brano.wav" if sonification_mode == "Brano Sperimentale (sequenza e mixaggio)" else "suono_colore.wav")

                    st.download_button(
                        label="⬇️ Scarica il suono generato",
                        data=open(audio_output_path, 'rb').read(),
                        file_name=file_name_output,
                        mime="audio/wav"
                    )
                    
                    os.unlink(audio_output_path)
                else:
                     st.error("❌ Errore nella generazione del suono: nessun segmento audio generato o un problema non specificato.")
            
with col_right:
    if st.session_state.processed_images_data:
        for img_data in st.session_state.processed_images_data:
            st.markdown(f"#### Analisi per Immagine: {img_data['name']}")
            # L'immagine si ridimensionerà automaticamente alla nuova larghezza della colonna (50%)
            st.image(img_data['image_bytes'], caption=f"Foto: {img_data['name']}", use_container_width=True)

            frequencies_and_weights_with_vval = img_data['frequencies_and_weights']
            
            # Re-run analysis to get hist_normalized, bin_edges, all_bin_actual_colors_hex
            # This is safe because analyze_image_and_map_to_frequencies is cached.
            _, hist_normalized, bin_edges, all_bin_actual_colors_hex = analyze_image_and_map_to_frequencies(
                img_data['image_bytes'], n_bins=st.session_state.get('n_bins_input_val', 5)
            )

            col_chart1, col_chart2 = st.columns(2)

            with col_chart1:
                st.markdown("##### Distribuzione Tonalità Colore")
                fig_color, ax_color = plt.subplots(figsize=(5, 3)) # Maintained figsize
                hue_bin_labels = [f"{int(bin_edges[i])}°-{int(bin_edges[i+1])}°" for i in range(len(bin_edges)-1)]
                
                ax_color.bar(hue_bin_labels, hist_normalized * 100, color=all_bin_actual_colors_hex) 
                ax_color.set_xlabel("Fascia di Tonalità (gradi Hue)")
                ax_color.set_ylabel("Percentuale (%)")
                ax_color.set_title("Percentuale Pixels per Fascia di Tonalità")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                st.pyplot(fig_color)
                plt.close(fig_color) 

            with col_chart2:
                st.markdown("##### Frequenze Generate e Peso")
                freq_labels = [f"{f:.0f} Hz" for f, w, _, _, _, _ in frequencies_and_weights_with_vval]
                freq_weights = [w * 100 for f, w, _, _, _, _ in frequencies_and_weights_with_vval]
                
                bar_colors_freq = [item[4] for item in frequencies_and_weights_with_vval]

                fig_freq, ax_freq = plt.subplots(figsize=(5, 3)) # Maintained figsize
                ax_freq.bar(freq_labels, freq_weights, color=bar_colors_freq)
                ax_freq.set_xlabel("Frequenza (Hz)")
                ax_freq.set_ylabel("Peso nell'Accordo (%)")
                ax_freq.set_title("Frequenze e loro Peso nel Suono")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                st.pyplot(fig_freq)
                plt.close(fig_freq) 
                    
            st.markdown("---")
            st.markdown("##### Dettagli Frequenze per Fascia di Colore:")
            
            # Sort frequencies_and_weights_with_vval by frequency for better readability
            sorted_frequencies_details = sorted(frequencies_and_weights_with_vval, key=lambda x: x[0])

            for freq, weight, hue_start, hue_end, rep_hex, v_val in sorted_frequencies_details:
                hue_name = get_hue_range_name(hue_start, hue_end)
                
                percentage_str = f"{weight*100:.1f}%"
                frequency_str = f"{int(freq)} Hz" 
                brightness_str = f"{v_val:.2f}" 
                
                freq_type = ""
                if freq < 200: freq_type = "Molto Bassa" 
                elif freq < 500: freq_type = "Bassa"
                elif freq < 800: freq_type = "Medio-Bassa"
                elif freq < 1200: freq_type = "Media"
                elif freq < 1800: freq_type = "Medio-Alta"
                elif freq < 2000: freq_type = "Alta"
                else: freq_type = "Molto Alta"
                
                # Formato tipo elenco
                st.markdown(f"""
                <div style="border: 1px solid #ddd; padding: 10px; margin-bottom: 10px; border-radius: 5px;">
                    <p><strong>Fascia Colore (Hue):</strong> <span style='background-color:{rep_hex}; padding: 2px 5px; border-radius:3px; display:inline-block; vertical-align:middle;'>&nbsp;</span> {hue_name} ({hue_start}°-{hue_end}°)</p>
                    <p><strong>% Pixels:</strong> {percentage_str}</p>
                    <p><strong>Frequenza:</strong> {frequency_str}</p>
                    <p><strong>Luminosità HSV:</strong> {brightness_str}</p>
                    <p><strong>Altezza:</strong> {freq_type}</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---") # Separator between images

    else:
        st.info("👆 Carica una o più foto per iniziare la sonificazione! L'analisi apparirà qui a destra.")
        st.markdown("""
        ### Come funziona:
        1.  **Carica una o più foto** (JPG, PNG). Puoi caricarne fino a 10 per creare un brano.
            * L'analisi dei colori e i grafici appariranno **immediatamente** dopo il caricamento in questa colonna!
        2.  L'applicazione analizzerà i colori di ogni immagine, **interpolando le frequenze** per i colori misti
            e assegnando frequenze fisse per i colori primari e acromatici.
        3.  **Scegli la modalità di sonificazione** dalla colonna di sinistra:
            * "Singola Immagine" (un accordo statico per la durata scelta)
            * "Brano Sperimentale" (una sequenza di accordi dalle tue foto, con controllo su battute, tempo e **mixaggio continuo**)
            * "Brano basato su Scansione Immagine" (per creare una "melodia" scansionando una singola immagine fetta per fetta).
        4.  **Scegli il tipo di onda sonora** che vuoi utilizzare: una singola onda per tutti i colori, una miscela di tutte,
            o un'assegnazione automatica basata sulla luminosità dei colori, con selezioni personalizzabili.
        5.  **NOVITÀ:** Regola la **"Velocità del Brano"** per impostare rapidamente il ritmo generale.
        6.  Clicca su "Genera Suono dai Colori" per creare il tuo accordo o brano!
        7.  Potrai ascoltare e scaricare il suono generato!
        """)

# Footer
st.markdown("---")
st.markdown("🎨🎵 **Sonificazione dei Colori** - Trasforma le immagini in suoni!")
st.markdown("*💡 Esplora il legame tra luce e suono.*")
