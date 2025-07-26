import streamlit as st
from PIL import Image
import numpy as np
import scipy.io.wavfile as wavfile
import tempfile
import os
import matplotlib.pyplot as plt
import colorsys # Per convertire RGB a HSV

# Configurazione della pagina
st.set_page_config(page_title="🎨🎵 Sonificazione dei Colori by loop507", layout="centered")

st.markdown("<h1>🎨🎵 Sonificazione dei Colori <span style='font-size:0.5em;'>by loop507</span></h1>", unsafe_allow_html=True)
st.write("Carica una foto e genera un suono basato sui suoi colori!")

# --- Mappatura delle Frequenze per Classificazione Colore ---
# Definiamo delle funzioni per classificare il colore e assegnare una frequenza
# in base alle tue richieste (es. Bianco 2000Hz, Giallo 1900Hz).
# Useremo i valori HSV (Hue, Saturation, Value) per la classificazione.

def get_frequency_for_color_class(h_deg, s_val, v_val, min_f_fallback=20, max_f_fallback=2000):
    """
    Classifica un colore HSV medio e restituisce una frequenza discreta.
    h_deg: Hue in gradi (0-360)
    s_val: Saturation (0.0-1.0)
    v_val: Value (0.0-1.0)
    min_f_fallback, max_f_fallback: Frequenze min/max per i casi di fallback o grigi non classificati.
    """
    
    # 1. Colori Acromatici (Bianco, Nero, Grigio) - priorità alta
    if s_val < 0.15: # Bassa saturazione indica colori acromatici
        if v_val > 0.9: # Molto luminoso = Bianco
            return 2000 # Frequenza per il Bianco
        elif v_val < 0.1: # Molto scuro = Nero
            return 20 # Frequenza per il Nero
        else: # Luminosità intermedia, bassa saturazione = Grigio
            return 200 # Frequenza per il Grigio (es. 200 Hz)
            
    # 2. Colori Cromatici (basati sulla tonalità/Hue)
    # Queste frequenze sono esempi per riempire il "ecc. ecc." mantenendo un ordine logico.
    # Sono state scelte per essere discrete e distinte.

    # Giallo (tra 45 e 75 gradi di Hue) e luminoso
    if 45 <= h_deg < 75 and v_val > 0.6: 
        return 1900 # Frequenza per il Giallo (come richiesto)
    
    # Arancione (tra 20 e 45 gradi)
    if 20 <= h_deg < 45 and v_val > 0.4:
        return 1700
        
    # Ciano/Azzurro (tra 165 e 210 gradi)
    if 165 <= h_deg < 210 and v_val > 0.4:
        return 1600
        
    # Verde (tra 75 e 165 gradi)
    if 75 <= h_deg < 165 and v_val > 0.4:
        return 1300
        
    # Magenta/Rosa (tra 300 e 345 gradi o molto basso < 20 per i rossi viola)
    if (300 <= h_deg < 345 or h_deg < 20) and v_val > 0.4:
        return 1000 

    # Rosso (Hue intorno a 0/360 gradi) - Scuro/Medio
    if (h_deg >= 345 or h_deg < 20) and v_val < 0.6:
        return 700

    # Blu (tra 210 e 285 gradi)
    if 210 <= h_deg < 285 and v_val > 0.2:
        return 400
        
    # Fallback: Se non rientra in nessuna categoria specifica
    normalized_value = (v_val - 0.1) / 0.9 
    if normalized_value < 0: normalized_value = 0 
    
    return min_f_fallback + normalized_value * (max_f_fallback - min_f_fallback) * 0.5 

# --- Funzione per l'analisi del colore (basata su HUE e VALUE per classificazione) ---
def analyze_image_and_map_to_frequencies(image_path, min_freq_overall=20, max_freq_overall=2000, n_bins=5):
    try:
        img = Image.open(image_path).convert('RGB')
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
        
        hist, bin_edges_hist = np.histogram(hue_values_all_pixels, bins=n_bins, range=(0, 361)) 
        
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
                
                # Assegna la frequenza usando la nuova logica di classificazione
                # Passiamo min_freq_overall e max_freq_overall al fallback
                frequency = get_frequency_for_color_class(avg_h_deg, avg_s_val, avg_v_val, min_freq_overall, max_freq_overall)
            else:
                actual_hex_color_for_bin = "#CCCCCC" 
                frequency = min_freq_overall 
            
            all_bin_actual_colors_hex.append(actual_hex_color_for_bin)

            if hist_normalized[i] > 0: 
                amplitude_weight = hist_normalized[i]
                
                frequencies_and_weights.append((frequency, amplitude_weight, int(bin_edges_hist[i]), int(bin_edges_hist[i+1]), actual_hex_color_for_bin))
            
        return frequencies_and_weights, hist_normalized, bin_edges_hist, all_bin_actual_colors_hex
    except Exception as e:
        st.error(f"Errore nell'analisi dell'immagine: {e}")
        return [], np.array([]), np.array([]), []

# --- Funzione per generare un'onda sinusoidale o un accordo ---
def generate_audio_wave(frequencies_and_weights, duration_seconds, sample_rate=44100):
    if not frequencies_and_weights:
        return np.zeros(int(sample_rate * duration_seconds), dtype=np.float32)

    t = np.linspace(0., duration_seconds, int(sample_rate * duration_seconds), endpoint=False)
    
    combined_amplitude = np.zeros_like(t, dtype=np.float32)
    
    total_weight = sum(w for f, w, _, _, _ in frequencies_and_weights) 
    if total_weight == 0:
        return np.zeros_like(t, dtype=np.float32) 
    
    for freq, weight, _, _, _ in frequencies_and_weights: 
        if freq > 0 and weight > 0: 
            amplitude = np.sin(2 * np.pi * freq * t) * (weight / total_weight)
            combined_amplitude += amplitude
        
    if np.max(np.abs(combined_amplitude)) > 0:
        combined_amplitude /= np.max(np.abs(combined_amplitude))
        
    return combined_amplitude

# --- Sezione principale dell'app ---
uploaded_file = st.file_uploader("📸 Carica una foto", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_image_file:
        tmp_image_file.write(uploaded_file.read())
        image_path = tmp_image_file.name
    
    st.image(image_path, caption="Foto Caricata", use_container_width=True)
    
    st.markdown("### ⚙️ Impostazioni Sonificazione")
    
    duration_input = st.slider("Durata del suono (secondi)", 0.5, 10.0, 2.0, 0.5)
    
    col1, col2 = st.columns(2)
    with col1:
        min_freq_input = st.number_input("Frequenza Minima (Hz)", min_value=1, max_value=20000, value=20, key="min_f")
    with col2:
        max_freq_input = st.number_input("Frequenza Massima (Hz)", min_value=1, max_value=20000, value=2000, key="max_f")
    
    n_bins_input = st.slider("Numero di Fasce di Colore (Tonalità)", 1, 10, 5, 1, 
                             help="Più fasce = più frequenze diverse nel suono (suono più ricco). Meno fasce = suono più semplice.")
    
    if min_freq_input >= max_freq_input:
        st.warning("La Frequenza Minima deve essere inferiore alla Frequenza Massima.")
        
    # --- Sezione di analisi e visualizzazione immediata ---
    st.markdown("### 📊 Analisi Colori e Frequenze Associate:")
    
    with st.spinner("Analizzando i colori della foto..."):
        frequencies_and_weights, hist_normalized, bin_edges, all_bin_actual_colors_hex = analyze_image_and_map_to_frequencies(
            image_path, min_freq_input, max_freq_input, n_bins_input
        )
        
        if frequencies_and_weights or (hist_normalized.size > 0 and np.sum(hist_normalized) > 0): 
            st.success("Analisi dei colori completata!")
            
            col_chart1, col_chart2 = st.columns(2)

            with col_chart1:
                st.markdown("#### Distribuzione Tonalità Colore")
                fig_color, ax_color = plt.subplots(figsize=(6, 4))
                hue_bin_labels = [f"{int(bin_edges[i])}°-{int(bin_edges[i+1])}°" for i in range(len(bin_edges)-1)]
                
                ax_color.bar(hue_bin_labels, hist_normalized * 100, color=all_bin_actual_colors_hex) 
                ax_color.set_xlabel("Fascia di Tonalità (Hue in gradi)")
                ax_color.set_ylabel("Percentuale (%)")
                ax_color.set_title("Percentuale Pixels per Fascia di Tonalità")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                st.pyplot(fig_color)
                plt.close(fig_color) 

            with col_chart2:
                st.markdown("#### Frequenze Generate e Peso")
                freq_labels = [f"{f:.0f} Hz" for f, w, _, _, _ in frequencies_and_weights]
                freq_weights = [w * 100 for f, w, _, _, _ in frequencies_and_weights]
                
                bar_colors_freq = [item[4] for item in frequencies_and_weights]

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
            st.markdown("#### Tabella Dettaglio Frequenze:")
            st.markdown("| Fascia Tonalità (Hue) | Percentuale | Frequenza Associata (Hz) | Tipo Frequenza |")
            st.markdown("|:----------------------:|:-----------:|:--------------------------:|:--------------:|")
            
            for freq, weight, hue_start, hue_end, rep_hex in frequencies_and_weights:
                hue_range_str = f"<span style='background-color:{rep_hex}; padding: 2px 5px; border-radius:3px;'>&nbsp;&nbsp;&nbsp;</span> {hue_start}°-{hue_end}°"
                percentage_str = f"{weight*100:.1f}%"
                frequency_str = f"{freq:.2f}"
                
                freq_type = ""
                if freq < 200: freq_type = "Molto Bassa" 
                elif freq < 500: freq_type = "Bassa"
                elif freq < 2000: freq_type = "Media"
                elif freq < 8000: freq_type = "Alta"
                else: freq_type = "Molto Alta"
                
                st.markdown(f"| {hue_range_str} | {percentage_str} | {frequency_str} | {freq_type} |", unsafe_allow_html=True)
            
            st.markdown("---")
            
            # --- Spiegazione Generale della Mappatura Aggiornata ---
            st.markdown("### 🔍 Come i Colori diventano Suoni:")
            st.markdown(f"""
            Questa applicazione analizza i colori della tua immagine, li classifica in categorie (Bianco, Giallo, Rosso, ecc.)
            e assegna una **frequenza sonora specifica e discreta** a ciascuna categoria.
            
            * **Bianco:** {get_frequency_for_color_class(0, 0, 1, min_freq_input, max_freq_input)} Hz (frequenza più alta)
            * **Giallo:** {get_frequency_for_color_class(60, 0.8, 0.9, min_freq_input, max_freq_input)} Hz
            * **Verde:** {get_frequency_for_color_class(120, 0.8, 0.5, min_freq_input, max_freq_input)} Hz
            * **Blu:** {get_frequency_for_color_class(240, 0.8, 0.5, min_freq_input, max_freq_input)} Hz
            * **Rosso:** {get_frequency_for_color_class(0, 0.8, 0.5, min_freq_input, max_freq_input)} Hz
            * **Grigio:** {get_frequency_for_color_class(0, 0.05, 0.5, min_freq_input, max_freq_input)} Hz
            * **Nero:** {get_frequency_for_color_class(0, 0, 0, min_freq_input, max_freq_input)} Hz (frequenza più bassa)
            
            Il suono finale è un 'accordo' creato dalla combinazione delle frequenze associate ai colori più rappresentativi
            nell'immagine, con l'intensità di ciascuna frequenza proporzionale alla percentuale
            di quel 'colore' nella foto.
            """)
            
            st.markdown("#### Mappatura Colore Nominato ➡️ Frequenza Discreta (Esempio)")
            hue_gradient_html = """
            <div style="width:100%; height:30px; 
                        background: linear-gradient(to right, 
                        #000000, #808080, #FF0000, #FFFF00, #00FF00, #0000FF, #FFFFFF);">
            </div>
            <div style="display:flex; justify-content:space-between; font-size:0.8em;">
                <span>Nero (20 Hz)</span>
                <span>Grigio (200 Hz)</span>
                <span>Rosso (700 Hz)</span>
                <span>Giallo (1900 Hz)</span>
                <span>Verde (1300 Hz)</span>
                <span>Blu (400 Hz)</span>
                <span>Bianco (2000 Hz)</span>
            </div>
            """
            st.markdown(hue_gradient_html, unsafe_allow_html=True)
            
            st.markdown("---")


            # --- Pulsante di generazione suono (rimane separato) ---
            if st.button("🎵 Genera Suono dai Colori"):
                with st.spinner("Generando il suono..."):
                    audio_data = generate_audio_wave(frequencies_and_weights, duration_input)
                    audio_data_int16 = (audio_data * 32767).astype(np.int16) 
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_audio_file:
                        audio_output_path = tmp_audio_file.name
                        wavfile.write(audio_output_path, 44100, audio_data_int16) 
                    
                    st.markdown("### Ascolta il tuo Suono:")
                    st.audio(audio_output_path, format='audio/wav')
                    
                    st.download_button(
                        label="⬇️ Scarica il suono generato",
                        data=open(audio_output_path, 'rb').read(),
                        file_name="suono_colore.wav",
                        mime="audio/wav"
                    )
                    
                    os.unlink(audio_output_path)
            
        else:
            st.warning("Nessuna frequenza generata. Assicurati che l'immagine non sia vuota o danneggiata.")
        
    os.unlink(image_path)
            
else:
    st.info("⬆️ Carica una foto per iniziare la sonificazione!")
    st.markdown("""
    ### Come funziona:
    1. **Carica una foto** (JPG, PNG).
    2. L'applicazione analizzerà i colori della tua immagine, li **classificherà** in categorie (es. Bianco, Giallo, Rosso)
       e assegnerà una **frequenza sonora specifica e discreta** a ciascuna categoria.
    3. **Verranno mostrati istogrammi e una tabella** con la percentuale di ogni fascia di colore e la frequenza sonora associata. I colori negli istogrammi e nella tabella rispecchieranno i colori reali della tua foto!
    4. Clicca su "Genera Suono dai Colori" per creare un **suono combinato** (un accordo) che rappresenta la distribuzione dei colori della tua immagine, della durata desiderata.
    5. Potrai ascoltare e scaricare il suono generato!
    """)

# Footer
st.markdown("---")
st.markdown("🎨🎵 **Sonificazione dei Colori** - Trasforma le immagini in suoni!")
st.markdown("*💡 Esplora il legame tra luce e suono.*")
