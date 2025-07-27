import streamlit as st
from PIL import Image
import numpy as np
import scipy.io.wavfile as wavfile
import tempfile
import os
import matplotlib.pyplot as plt
import colorsys # Per convertire RGB a HSV
from scipy import signal # Per onde quadre e a sega

# Configurazione della pagina
st.set_page_config(page_title="🎨🎵 Sonificazione dei Colori by loop507", layout="centered")

st.markdown("<h1>🎨🎵 Sonificazione dei Colori <span style='font-size:0.5em;'>by loop507</span></h1>", unsafe_allow_html=True)
st.write("Carica una o più foto e genera un suono basato sui suoi colori, ora anche come brano sperimentale!")

# --- Mappatura delle Frequenze per Classificazione Colore ---
def get_frequency_for_color_class(h_deg, s_val, v_val):
    """
    Classifica un colore HSV medio e restituisce una frequenza discreta o interpolata.
    h_deg: Hue in gradi (0-360)
    s_val: Saturation (0.0-1.0)
    v_val: Value (0.0-1.0)
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
    if h_deg_for_interp == 360:
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
    if hue_start > hue_end: 
        # Range che attraversa 0/360, calcola il punto medio "effettivo" sul cerchio
        mid_hue = (hue_start + hue_end + 360) / 2
        if mid_hue >= 360: mid_hue -= 360
    else:
        mid_hue = (hue_start + hue_end) / 2

    if 345 <= mid_hue <= 360 or 0 <= mid_hue < 15:
        return "Rosso"
    elif 15 <= mid_hue < 45:
        return "Rosso-Arancio"
    elif 45 <= mid_hue < 75:
        return "Arancio-Giallo"
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

# --- Funzione per l'analisi del colore ---
def analyze_image_and_map_to_frequencies(image_path, n_bins=5):
    try:
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)
        
        pixels_flat = img_array.reshape(-1, 3)
        
        bin_rgb_sums = np.zeros((n_bins, 3), dtype=float)
        bin_hsv_sums = np.zeros((n_bins, 3), dtype=float) 
        bin_pixel_counts = np.zeros(n_bins, dtype=int)
        
        # Adjust bin edges to avoid 360 falling into new bin
        temp_bin_edges_for_digitize = np.linspace(0, 360, n_bins + 1)
        # Ensure the last bin includes 360 if it's the max range.
        # For hue, 360 is the same as 0, so we use 0-359.99 for bins typically.
        # np.digitize usually handles upper bounds correctly.
        
        hue_values_all_pixels = []

        for r, g, b in pixels_flat:
            r_norm, g_norm, b_norm = r/255., g/255., b/255.
            
            h, s, v = colorsys.rgb_to_hsv(r_norm, g_norm, b_norm)
            hue_degrees = h * 360 
            
            hue_values_all_pixels.append(hue_degrees) 

            # np.digitize returns an array of bin indices for each value in `hue_degrees`
            # The indices are 1-based by default for the first return.
            # We want 0-based for array indexing.
            bin_idx = np.digitize(hue_degrees, temp_bin_edges_for_digitize) - 1
            
            # Ensure bin_idx is within valid range [0, n_bins-1]
            if bin_idx == n_bins: # If a value falls exactly on the last edge
                bin_idx = n_bins - 1
            if bin_idx < 0: # Should not happen with 0 as min
                bin_idx = 0

            bin_rgb_sums[bin_idx] += [r, g, b]
            bin_hsv_sums[bin_idx] += [hue_degrees, s, v] 
            bin_pixel_counts[bin_idx] += 1
        
        # Recalculate histogram to get accurate counts per bin (for weights)
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
                actual_hex_color_for_bin = "#CCCCCC" # Default grey if bin is empty
                frequency = 20 # Default minimal frequency
                avg_v_val = 0 # Default low brightness
            
            all_bin_actual_colors_hex.append(actual_hex_color_for_bin)

            if hist_normalized[i] > 0: # Only add if there are pixels in this bin
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
                continue # Skip to next frequency if all three are mixed
            elif waveform_mode == "by_brightness":
                if v_val > 0.7: # Colori Chiari (Luminosità Alta)
                    current_waveform_func = get_waveform_function(bright_wave)
                elif v_val < 0.3: # Colori Scuri (Luminosità Bassa)
                    current_waveform_func = get_waveform_function(dark_wave)
                else: # Colori Medi (Luminosità Media)
                    current_waveform_func = get_waveform_function(medium_wave)
            
            if current_waveform_func is None:
                 current_waveform_func = get_waveform_function("sine") # Fallback to sine if nothing selected

            amplitude = current_waveform_func(t, freq) * (weight / total_weight)
            combined_amplitude += amplitude
        
    max_amplitude = np.max(np.abs(combined_amplitude))
    if max_amplitude > 0:
        combined_amplitude /= max_amplitude # Normalizza l'ampiezza per evitare clipping
        combined_amplitude *= 0.9 # Riduce leggermente per sicurezza dopo normalizzazione

    # Applica fade-in e fade-out al segmento generato
    if fade_in_duration > 0:
        fade_in_samples = int(fade_in_duration * sample_rate)
        if fade_in_samples > len(combined_amplitude):
            fade_in_samples = len(combined_amplitude) # Limit to segment length
        window_in = np.linspace(0., 1., fade_in_samples)
        combined_amplitude[:fade_in_samples] *= window_in

    if fade_out_duration > 0:
        fade_out_samples = int(fade_out_duration * sample_rate)
        if fade_out_samples > len(combined_amplitude):
            fade_out_samples = len(combined_amplitude) # Limit to segment length
        window_out = np.linspace(1., 0., fade_out_samples)
        combined_amplitude[-fade_out_samples:] *= window_out
        
    return combined_amplitude

# --- Sezione principale dell'app ---

# Abilita il caricamento di più file
uploaded_files = st.file_uploader("📸 Carica una o più foto (fino a 10)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

# Inizializza processed_images_data nello stato della sessione se non esiste
if 'processed_images_data' not in st.session_state:
    st.session_state.processed_images_data = []

# Se il numero di file caricati cambia, pulisci i dati processati per evitare incongruenze
# Compara i nomi dei file caricati con quelli nello stato della sessione
uploaded_file_names = [f.name for f in uploaded_files]
stored_file_names = [data['name'] for data in st.session_state.processed_images_data]

if uploaded_file_names != stored_file_names:
    st.session_state.processed_images_data = [] # Pulisci se i file sono cambiati o l'ordine è diverso

if uploaded_files:
    # Limita il numero di file a 10
    if len(uploaded_files) > 10:
        st.warning("Hai caricato più di 10 immagini. Saranno processate solo le prime 10.")
        uploaded_files = uploaded_files[:10]

    st.markdown("### ⚙️ Impostazioni Sonificazione")
    
    # Checkbox per scegliere la modalità: singola immagine o brano sperimentale
    sonification_mode = st.radio(
        "Modalità di Sonificazione:",
        ["Singola Immagine (un accordo per immagine)", "Brano Sperimentale (sequenza e mixaggio)"],
        key="sonification_mode_selector"
    )

    duration_input = 2.0 # Default value, will be overridden
    sample_rate = 44100 # Definizione del sample rate
    
    if sonification_mode == "Singola Immagine (un accordo per immagine)":
        duration_input = st.slider("Durata del suono (secondi)", 0.5, 60.0, 2.0, 0.5) # Aumento max durata a 60s
    else: # Brano Sperimentale
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
                value=1.0,
                step=0.1,
                help="Durata di ogni singola battuta."
            )
        
        # New slider for overlap duration
        overlap_duration = st.slider(
            "Durata Sovrapposizione (secondi)",
            min_value=0.0,
            max_value=tempo_per_beat * beats_per_image * 0.9, # Max 90% of segment duration
            value=min(0.5, tempo_per_beat * beats_per_image * 0.5), # Default 0.5s or half segment
            step=0.1,
            help="Per quanto tempo il suono di un'immagine si sovrappone a quello successivo. Controlla il grado di mixaggio."
        )

        # Durata base di ogni segmento senza sovrapposizione
        segment_duration_raw = beats_per_image * tempo_per_beat
        
        # Calcolo durata totale del brano
        if len(uploaded_files) > 0:
            total_estimated_duration = (len(uploaded_files) * segment_duration_raw) - ((len(uploaded_files) - 1) * overlap_duration)
            if total_estimated_duration < segment_duration_raw and len(uploaded_files) > 0: # Ensure minimum duration is at least one segment
                total_estimated_duration = segment_duration_raw
        else:
            total_estimated_duration = 0

        st.info(f"Ogni immagine durerà {segment_duration_raw:.1f} secondi. Durata totale stimata del brano: {total_estimated_duration:.1f} secondi.")


    col1, col2 = st.columns(2)
    with col1:
        min_freq_input = st.number_input("Frequenza Minima (Hz)", min_value=1, max_value=20000, value=20, key="min_f")
    with col2:
        max_freq_input = st.number_input("Frequenza Massima (Hz)", min_value=1, max_value=20000, value=2000, key="max_f")
    
    n_bins_input = st.slider("Numero di Fasce di Colore (Tonalità)", 1, 10, 5, 1, 
                             help="Più fasce = più frequenze diverse nel suono (suono più ricco). Meno fasce = suono più semplice.")
    
    if min_freq_input >= max_freq_input:
        st.warning("La Frequenza Minima deve essere inferiore alla Frequenza Massima.")
        
    # --- Controlli per il Tipo di Onda Sonora ---
    st.markdown("### 🎶 Tipo di Onda Sonora")
    waveform_selection_mode = st.radio(
        "Come vuoi generare le onde?",
        ["Onda Singola per tutti i Colori", "Onda per Luminosità del Colore"],
        key="waveform_mode_selector"
    )

    selected_single_waveform = "Sine" # Default
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
                index=0, # Default a sine
                key="bright_wave_type"
            )
        with col_medium:
            medium_wave_type = st.selectbox(
                "Colori Medi (Luminosità Media):",
                ["sine", "square", "sawtooth"],
                index=1, # Default a square
                key="medium_wave_type"
            )
        with col_dark:
            dark_wave_type = st.selectbox(
                "Colori Scuri (Luminosità Bassa):",
                ["sine", "square", "sawtooth"],
                index=2, # Default a sawtooth
                key="dark_wave_type"
            )
        st.markdown("---")

    # Mostra analisi e generazione solo se ci sono file caricati e i dati non sono già in session_state
    if not st.session_state.processed_images_data: # Solo se i dati non sono già stati processati
        for i, uploaded_file in enumerate(uploaded_files):
            st.markdown(f"#### Analisi per Immagine {i+1}: {uploaded_file.name}")
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_image_file:
                tmp_image_file.write(uploaded_file.read())
                image_path = tmp_image_file.name
            
            st.image(image_path, caption=f"Foto {i+1}: {uploaded_file.name}", use_container_width=True)
            
            with st.spinner(f"Analizzando i colori dell'immagine {i+1}..."):
                frequencies_and_weights_with_vval, hist_normalized, bin_edges, all_bin_actual_colors_hex = analyze_image_and_map_to_frequencies(
                    image_path, n_bins_input
                )
                
                if frequencies_and_weights_with_vval or (hist_normalized.size > 0 and np.sum(hist_normalized) > 0): 
                    st.success(f"Analisi colori per immagine {i+1} completata!")
                    
                    col_chart1, col_chart2 = st.columns(2)

                    with col_chart1:
                        st.markdown("##### Distribuzione Tonalità Colore")
                        fig_color, ax_color = plt.subplots(figsize=(6, 4))
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

                        fig_freq, ax_freq = plt.subplots(figsize=(6, 4))
                        ax_freq.bar(freq_labels, freq_weights, color=bar_colors_freq)
                        ax_freq.set_xlabel("Frequenza (Hz)")
                        ax_freq.set_ylabel("Peso nell'Accordo (%)")
                        ax_freq.set_title("Frequenze e loro Peso nel Suono")
                        plt.xticks(rotation=45, ha="right")
                        plt.tight_layout()
                        st.pyplot(fig_freq)
                        plt.close(fig_freq) 
                        
                    st.markdown("---")
                    st.markdown("##### Tabella Dettaglio Frequenze:")
                    st.markdown("| Fascia Colore (Hue) | % Pixels | Frequenza (Hz) | Luminosità HSV | Altezza |")
                    st.markdown("|:--------------------|:---------|:---------------|:---------------|:--------|")
                    
                    for freq, weight, hue_start, hue_end, rep_hex, v_val in frequencies_and_weights_with_vval:
                        hue_name = get_hue_range_name(hue_start, hue_end)
                        
                        # Added vertical-align:middle to the span for better alignment of color swatch
                        hue_range_str = f"**{hue_name}** <span style='background-color:{rep_hex}; padding: 2px 5px; border-radius:3px; display:inline-block; vertical-align:middle;'>&nbsp;</span> ({hue_start}°-{hue_end}°)"
                        percentage_str = f"{weight*100:.1f}%"
                        frequency_str = f"{int(freq)}" # Arrotonda a intero
                        brightness_str = f"{v_val:.2f}" 
                        
                        freq_type = ""
                        if freq < 200: freq_type = "Molto Bassa" 
                        elif freq < 500: freq_type = "Bassa"
                        elif freq < 800: freq_type = "Medio-Bassa"
                        elif freq < 1200: freq_type = "Media"
                        elif freq < 1800: freq_type = "Medio-Alta"
                        elif freq < 2000: freq_type = "Alta"
                        else: freq_type = "Molto Alta"
                        
                        st.markdown(f"| {hue_range_str} | {percentage_str} | {frequency_str} | {brightness_str} | {freq_type} |", unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # Memorizza i dati per la generazione audio
                    st.session_state.processed_images_data.append({
                        'image_path': image_path,
                        'frequencies_and_weights': frequencies_and_weights_with_vval,
                        'name': uploaded_file.name
                    })

                else:
                    st.warning(f"Nessuna frequenza generata per l'immagine {i+1}. Assicurati che l'immagine non sia vuota o danneggiata.")
                
    else: # If data is already in session_state, just display paths and analysis if desired (optional, for brevity not showing full analysis again)
        st.info("Le immagini sono già state analizzate. Premi 'Genera Suono dai Colori' o modifica le impostazioni.")


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
    * **Sawtooth Wave (Onda a Dente di Sega):** Contiene tutte le armoniche. Ha un suono più "brillante" e ricco, simile a un violino o a un ottoni.
    
    Puoi scegliere un'unica onda per tutti i colori, una miscela di tutte e tre, oppure lasciare che l'applicazione
    assegni l'onda in base alla luminosità del colore, con assegnazioni personalizzabili:
    * **Colori Chiari (Luminosità Alta):** Puoi scegliere il tipo di onda.
    * **Colori Medi (Luminosità Media):** Puoi scegliere il tipo di onda.
    * **Colori Scuri (Luminosità Bassa):** Puoi scegliere il tipo di onda.
    """)
    
    st.markdown("#### Mappatura Tonalità (Hue) ➡️ Frequenza Base:")
    # Using a simpler flexbox for the labels below the gradient
    st.markdown("""
    <div style="width:100%; position:relative; height:150px; border-radius: 10px; overflow: hidden; margin-bottom: 20px;">
        <div style="width:100%; height:80px; background: 
            linear-gradient(to right, 
            #FF0000 0%, #FF8000 10%, #FFFF00 20%, #80FF00 30%, #00FF00 40%, 
            #00FF80 50%, #00FFFF 60%, #0080FF 70%, #0000FF 80%, #8000FF 90%, #FF00FF 100%);">
        </div>
        <div style="display:flex; justify-content:space-between; width:100%; position:absolute; top:80px; left:0; right:0; padding: 0 10px; box-sizing: border-box;">
            <div style="text-align:center; flex-basis:16.6%; font-size:0.75em;">Red<br>(0°)<br>700Hz</div>
            <div style="text-align:center; flex-basis:16.6%; font-size:0.75em;">Yellow<br>(60°)<br>1900Hz</div>
            <div style="text-align:center; flex-basis:16.6%; font-size:0.75em;">Green<br>(120°)<br>1300Hz</div>
            <div style="text-align:center; flex-basis:16.6%; font-size:0.75em;">Cyan<br>(180°)<br>1600Hz</div>
            <div style="text-align:center; flex-basis:16.6%; font-size:0.75em;">Blue<br>(240°)<br>400Hz</div>
            <div style="text-align:center; flex-basis:16.6%; font-size:0.75em;">Magenta<br>(300°)<br>1000Hz</div>
            <div style="text-align:center; flex-basis:0%; font-size:0.75em; visibility:hidden;">Red (360°)<br>700Hz</div>
        </div>
        <div style="text-align:center; width:100%; font-size:0.75em; margin-top:10px;">
            <p style="font-size:0.85em;"><i>Le frequenze intermedie sono interpolate tra questi punti di ancoraggio.<br>
            I colori acromatici (bianco, nero, grigio) e alcuni colori speciali (rosa, marrone, giallo chiaro) hanno frequenze fisse dedicate.</i></p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")


    # --- Pulsante di generazione suono ---
    if st.button("🎵 Genera Suono dai Colori"):
        if st.session_state.processed_images_data:
            with st.spinner("Generando il suono..."):
                all_raw_audio_segments = []
                
                # Determina la durata per ogni segmento in base alla modalità
                segment_duration = 0
                current_fade_in_duration = 0 # Inizializza per uso nel loop

                if sonification_mode == "Singola Immagine (un accordo per immagine)":
                    segment_duration = duration_input # Prende il valore dallo slider diretto
                    # Per singola immagine, nessun fade-in/out speciale a meno che non sia definito altrove
                else: # Brano Sperimentale
                    segment_duration = beats_per_image * tempo_per_beat
                    current_fade_in_duration = overlap_duration # Fade-in uguale alla durata di sovrapposizione
                
                # Genera i segmenti audio INDIVIDUALI prima di miscelarli
                for img_data in st.session_state.processed_images_data:
                    frequencies_and_weights_to_use = img_data['frequencies_and_weights']
                    
                    audio_data_segment = None
                    # No fade out duration passed to generate_audio_wave for individual segments in experimental mode
                    # as fade out will be handled by final mixing logic or global fade out.
                    if waveform_selection_mode == "Onda Singola per tutti i Colori":
                        if selected_single_waveform == "Mixed (Sine + Square + Sawtooth)":
                            audio_data_segment = generate_audio_wave(frequencies_and_weights_to_use, segment_duration, 
                                                            waveform_mode="mixed_all", sample_rate=sample_rate,
                                                            fade_in_duration=current_fade_in_duration if sonification_mode == "Brano Sperimentale (sequenza e mixaggio)" else 0,
                                                            fade_out_duration=0) # No segment-level fade-out for continuous mix
                        else:
                            waveform_map_internal = {
                                "Sine": "sine",
                                "Square": "square",
                                "Sawtooth": "sawtooth"
                            }
                            audio_data_segment = generate_audio_wave(frequencies_and_weights_to_use, segment_duration, 
                                                            waveform_mode="single", 
                                                            single_waveform_type=waveform_map_internal[selected_single_waveform], sample_rate=sample_rate,
                                                            fade_in_duration=current_fade_in_duration if sonification_mode == "Brano Sperimentale (sequenza e mixaggio)" else 0,
                                                            fade_out_duration=0) # No segment-level fade-out for continuous mix
                    else: # Onda per Luminosità del Colore
                        audio_data_segment = generate_audio_wave(frequencies_and_weights_to_use, segment_duration, 
                                                        waveform_mode="by_brightness",
                                                        bright_wave=bright_wave_type,
                                                        medium_wave=medium_wave_type,
                                                        dark_wave=dark_wave_type, sample_rate=sample_rate,
                                                        fade_in_duration=current_fade_in_duration if sonification_mode == "Brano Sperimentale (sequenza e mixaggio)" else 0,
                                                        fade_out_duration=0) # No segment-level fade-out for continuous mix
                    
                    if audio_data_segment is not None:
                        all_raw_audio_segments.append(audio_data_segment)
                
                # Combinazione dei segmenti per il "Mixing Continuo"
                final_audio_data = np.array([], dtype=np.float32)
                
                if all_raw_audio_segments:
                    if sonification_mode == "Singola Immagine (un accordo per immagine)":
                        # For single image, ensure it also has a proper fade-in/out if it's the only one.
                        # For simplicity, we just take the first segment and assume generate_audio_wave already applied any requested fades based on its direct duration_input.
                        final_audio_data = all_raw_audio_segments[0] 
                    else: # Brano Sperimentale con Mixing Continuo
                        
                        # Calculate total length needed for the mixed track
                        # Total duration is length of first segment + (length of remaining segments * (segment_duration - overlap_duration))
                        total_duration_calculated = segment_duration + (len(all_raw_audio_segments) - 1) * (segment_duration - overlap_duration)
                        if len(all_raw_audio_segments) == 1: # If there's only one image in experimental mode
                             total_duration_calculated = segment_duration
                        
                        total_samples_needed = int(total_duration_calculated * sample_rate)
                        
                        if total_samples_needed < 0: # Safety check for extreme overlap
                            total_samples_needed = int(segment_duration * sample_rate) 
                        
                        final_audio_data = np.zeros(total_samples_needed, dtype=np.float32)
                        
                        current_offset_samples = 0
                        # This is the step for the start of the next segment
                        offset_per_segment_samples = int((segment_duration - overlap_duration) * sample_rate)
                        
                        # Ensure offset_per_segment_samples is not negative or zero if overlap is too large
                        if offset_per_segment_samples < 0:
                            offset_per_segment_samples = 0 # No forward progress, all segments start at same point for extreme overlap
                        
                        for i, segment in enumerate(all_raw_audio_segments):
                            segment_samples = len(segment)
                            
                            # Calculate the end position for the current segment in the final buffer
                            end_pos = current_offset_samples + segment_samples
                            
                            # If the segment goes beyond the pre-calculated total duration, clip it
                            if end_pos > len(final_audio_data):
                                segment_to_add = segment[:len(final_audio_data) - current_offset_samples]
                            else:
                                segment_to_add = segment
                            
                            if len(segment_to_add) > 0:
                                final_audio_data[current_offset_samples : current_offset_samples + len(segment_to_add)] += segment_to_add
                            
                            # Update the offset for the next segment
                            current_offset_samples += offset_per_segment_samples
                    
                    # Normalizza il volume del brano finale per evitare clipping dopo la somma
                    max_amplitude = np.max(np.abs(final_audio_data))
                    if max_amplitude > 0:
                        final_audio_data /= max_amplitude # Normalizza a 1.0 (o -1.0 a 1.0)
                        final_audio_data *= 0.8 # Un po' di margine per sicurezza

                    audio_data_int16 = (final_audio_data * 32767).astype(np.int16) 
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_audio_file:
                        audio_output_path = tmp_audio_file.name
                        wavfile.write(audio_output_path, sample_rate, audio_data_int16) 
                    
                    st.markdown("### Ascolta il tuo Suono:")
                    st.audio(audio_output_path, format='audio/wav')
                    
                    st.download_button(
                        label="⬇️ Scarica il suono generato",
                        data=open(audio_output_path, 'rb').read(),
                        file_name="suono_colore_brano.wav" if sonification_mode == "Brano Sperimentale (sequenza e mixaggio)" else "suono_colore.wav",
                        mime="audio/wav"
                    )
                    
                    os.unlink(audio_output_path)
                else:
                     st.error("❌ Errore nella generazione del suono: nessun segmento audio generato.")
            
            # Pulizia dei file temporanei delle immagini
            for img_data in st.session_state.processed_images_data:
                try:
                    os.unlink(img_data['image_path'])
                except Exception as e:
                    st.warning(f"Impossibile eliminare il file temporaneo {img_data['image_path']}: {e}")
            del st.session_state.processed_images_data # Pulisce lo stato dopo la generazione
            
        else:
            st.warning("Carica almeno una foto per generare il suono.")
            
else:
    st.info("👆 Carica una o più foto per iniziare la sonificazione!")
    st.markdown("""
    ### Come funziona:
    1.  **Carica una o più foto** (JPG, PNG). Puoi caricarne fino a 10 per creare un brano.
    2.  L'applicazione analizzerà i colori di ogni immagine, **interpolando le frequenze** per i colori misti
        e assegnando frequenze fisse per i colori primari e acromatici.
    3.  **Scegli la modalità di sonificazione:** "Singola Immagine" (un accordo statico per la durata scelta)
        o "Brano Sperimentale" (una sequenza di accordi dalle tue foto, con controllo su battute, tempo e **mixaggio continuo**).
    4.  **Scegli il tipo di onda sonora** che vuoi utilizzare: una singola onda per tutte le frequenze, una miscela di tutte,
        o un'assegnazione automatica basata sulla luminosità dei colori, con selezioni personalizzabili.
    5.  **Verranno mostrati istogrammi e una tabella** con la percentuale di ogni fascia di colore e la frequenza sonora associata per ciascuna immagine.
    6.  Clicca su "Genera Suono dai Colori" per creare il tuo accordo o brano!
    7.  Potrai ascoltare e scaricare il suono generato!
    """)

# Footer
st.markdown("---")
st.markdown("🎨🎵 **Sonificazione dei Colori** - Trasforma le immagini in suoni!")
st.markdown("*💡 Esplora il legame tra luce e suono.*")
