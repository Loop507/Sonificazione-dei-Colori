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

# --- Funzione per l'analisi del colore (basata su HUE) e mappatura alle frequenze ---
def analyze_image_and_map_to_frequencies(image_path, min_freq=50, max_freq=2000, n_bins=5):
    """
    Analizza l'immagine, divide i colori in fasce di tonalità (hue) e mappa ciascuna a una frequenza.
    
    Args:
        image_path (str): Percorso del file immagine.
        min_freq (int): Frequenza minima per la mappatura.
        max_freq (int): Frequenza massima per la mappatura.
        n_bins (int): Numero di fasce di tonalità (hue) da analizzare.
        
    Returns:
        tuple: (list_of_frequencies_and_weights, hist_normalized, bin_edges, representative_colors_hex)
        list_of_frequencies_and_weights: Una lista di tuple (frequency, amplitude_weight, hue_bin_start, hue_bin_end, representative_rgb).
        hist_normalized: Array NumPy delle percentuali di pixel per ogni bin di tonalità.
        bin_edges: Array NumPy dei bordi dei bins di tonalità (0-360).
        representative_colors_hex: Lista di stringhe esadecimali dei colori rappresentativi per i grafici.
    """
    try:
        img = Image.open(image_path).convert('RGB') # Mantieni RGB per analisi colori
        img_array = np.array(img)
        
        # Converti RGB a HSV
        # colorsys lavora con valori 0.0-1.0
        hsv_image = np.array([colorsys.rgb_to_hsv(r/255., g/255., b/255.) for r, g, b in img_array.reshape(-1, 3)])
        
        hue_values = hsv_image[:, 0] * 360 # Converti hue in gradi (0-360)
        value_values = hsv_image[:, 2] # Valore/Luminosità (0.0-1.0)
        
        # Calcola l'istogramma delle tonalità (hue)
        # range è (0, 361) per includere 360
        hist, bin_edges = np.histogram(hue_values, bins=n_bins, range=(0, 361)) 
        
        total_pixels = np.sum(hist)
        hist_normalized = hist / total_pixels if total_pixels > 0 else np.zeros_like(hist)
        
        frequencies_and_weights = []
        representative_colors_hex = []

        for i in range(n_bins):
            if hist_normalized[i] > 0: # Processa solo le fasce con pixel
                # Calcola il punto medio di ciascun bin per la mappatura
                hue_bin_midpoint = (bin_edges[i] + bin_edges[i+1]) / 2
                
                # Per la frequenza, mappiamo la posizione del hue nel cerchio a una frequenza
                # Una mappatura più sofisticata potrebbe considerare anche la luminosità o saturazione media del bin
                
                # Normalizza la tonalità (0-360) a un valore 0.0-1.0
                normalized_hue = hue_bin_midpoint / 360.0
                
                # Mappatura logaritmica per una percezione più naturale delle frequenze
                current_min_freq = max(1, min_freq) 
                
                if normalized_hue == 0:
                    normalized_hue = 0.001 
                
                freq_ratio = max_freq / current_min_freq
                if freq_ratio <= 0: freq_ratio = 1 
                
                frequency = current_min_freq * (freq_ratio**normalized_hue)
                
                # Il peso (amplitude_weight) è la percentuale di pixel in quella fascia di tonalità
                amplitude_weight = hist_normalized[i]

                # Calcola un colore RGB rappresentativo per questa fascia di tonalità
                # Potremmo prendere il colore medio dei pixel in quella fascia,
                # ma per semplicità prendiamo il punto medio del hue bin con saturazione e valore alti
                rep_rgb = colorsys.hsv_to_rgb(hue_bin_midpoint/360., 0.8, 0.8) # S e V a 0.8 per colori vivaci
                rep_hex = '#%02x%02x%02x' % (int(rep_rgb[0]*255), int(rep_rgb[1]*255), int(rep_rgb[2]*255))
                
                frequencies_and_weights.append((frequency, amplitude_weight, int(bin_edges[i]), int(bin_edges[i+1]), rep_hex))
                representative_colors_hex.append(rep_hex)
            else: # Se la fascia è vuota, aggiungi comunque un placeholder per il grafico
                representative_colors_hex.append("#CCCCCC") # Grigio chiaro per fasce vuote
        
        return frequencies_and_weights, hist_normalized, bin_edges, representative_colors_hex
    except Exception as e:
        st.error(f"Errore nell'analisi dell'immagine: {e}")
        return [], np.array([]), np.array([]), []

# --- Funzione per generare un'onda sinusoidale o un accordo ---
def generate_audio_wave(frequencies_and_weights, duration_seconds, sample_rate=44100):
    """
    Genera un'onda sonora combinando più sinusoidi.
    
    Args:
        frequencies_and_weights (list): Lista di tuple (frequency, amplitude_weight, hue_bin_start, hue_bin_end, representative_rgb).
        duration_seconds (float): Durata dell'onda in secondi.
        sample_rate (int): Frequenza di campionamento in Hz.
        
    Returns:
        numpy.ndarray: Array di campioni audio normalizzato a -1.0 a 1.0.
    """
    if not frequencies_and_weights:
        return np.zeros(int(sample_rate * duration_seconds), dtype=np.float32)

    t = np.linspace(0., duration_seconds, int(sample_rate * duration_seconds), endpoint=False)
    
    combined_amplitude = np.zeros_like(t, dtype=np.float32)
    
    # Normalizza i pesi in modo che la somma delle ampiezze non superi 1
    total_weight = sum(w for f, w, _, _, _ in frequencies_and_weights) # Ora ha 5 elementi
    if total_weight == 0:
        return np.zeros_like(t, dtype=np.float32) # Evita divisione per zero
    
    for freq, weight, _, _, _ in frequencies_and_weights: # Estrai solo freq e weight per la generazione
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
        min_freq_input = st.number_input("Frequenza Minima (Hz)", min_value=1, max_value=20000, value=50, key="min_f")
    with col2:
        max_freq_input = st.number_input("Frequenza Massima (Hz)", min_value=1, max_value=20000, value=2000, key="max_f")
    
    n_bins_input = st.slider("Numero di Fasce di Colore (Tonalità)", 1, 10, 5, 1, 
                             help="Più fasce = più frequenze diverse nel suono (suono più ricco). Meno fasce = suono più semplice.")
    
    if min_freq_input >= max_freq_input:
        st.warning("La Frequenza Minima deve essere inferiore alla Frequenza Massima.")
        
    # --- Sezione di analisi e visualizzazione immediata ---
    st.markdown("### 📊 Analisi Colori e Frequenze Associate:")
    
    with st.spinner("Analizzando i colori della foto..."):
        frequencies_and_weights, hist_normalized, bin_edges, representative_colors_hex = analyze_image_and_map_to_frequencies(
            image_path, min_freq_input, max_freq_input, n_bins_input
        )
        
        if frequencies_and_weights or (hist_normalized.size > 0 and np.sum(hist_normalized) > 0): # Check for empty results due to analysis issues
            st.success("Analisi dei colori completata!")
            
            col_chart1, col_chart2 = st.columns(2)

            with col_chart1:
                st.markdown("#### Distribuzione Tonalità Colore")
                fig_color, ax_color = plt.subplots(figsize=(6, 4))
                # Etichette dei bin per la tonalità (es. 0-36, 36-72)
                hue_bin_labels = [f"{int(bin_edges[i])}°-{int(bin_edges[i+1])}°" for i in range(len(bin_edges)-1)]
                
                ax_color.bar(hue_bin_labels, hist_normalized * 100, color=representative_colors_hex[:len(hist_normalized)]) 
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
                
                # Usa gli stessi colori rappresentativi della tonalità
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
            
            # --- Spiegazione Generale della Mappatura ---
            st.markdown("### 🔍 Come i Colori diventano Suoni:")
            st.markdown(f"""
            Questa applicazione analizza i **colori (tonalità/Hue)** della tua immagine, dividendoli in fasce (es. blu, verde, rosso, ecc.).
            
            * Le **tonalità più vicine al rosso/arancione** (Hue basso, es. 0-60°) sono associate a **frequenze più basse**.
            * Le **tonalità più vicine al blu/viola** (Hue alto, es. 240-300°) sono associate a **frequenze più alte**.
            
            Il suono finale è un 'accordo' creato dalla combinazione delle frequenze più rappresentative
            nell'immagine, con l'intensità di ciascuna frequenza proporzionale alla percentuale
            di quel 'colore' nella foto.
            """)
            
            st.markdown("#### Scala Tonalità ➡️ Frequenza (Esempio)")
            hue_gradient_html = """
            <div style="width:100%; height:30px; 
                        background: linear-gradient(to right, 
                        #FF0000, #FFFF00, #00FF00, #00FFFF, #0000FF, #FF00FF, #FF0000);">
            </div>
            <div style="display:flex; justify-content:space-between; font-size:0.8em;">
                <span>Rosso (0°) - Bassa Freq.</span>
                <span>Verde (120°)</span>
                <span>Blu (240°) - Alta Freq.</span>
                <span>Rosso (360°)</span>
            </div>
            """
            st.markdown(hue_gradient_html, unsafe_allow_html=True)
            
            st.markdown("---")


            # --- Pulsante di generazione suono (rimane separato) ---
            if st.button("🎵 Genera Suono dai Colori"):
                with st.spinner("Generando il suono..."):
                    # 2. Genera l'onda sonora combinata
                    audio_data = generate_audio_wave(frequencies_and_weights, duration_input)
                    
                    # Scala l'audio per un volume ragionevole (int16 per WAV standard)
                    audio_data_int16 = (audio_data * 32767).astype(np.int16) 
                    
                    # 3. Salva l'audio in un file temporaneo
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
                    
                    # Pulizia del file audio temporaneo
                    os.unlink(audio_output_path)
            
        else:
            st.warning("Nessuna frequenza generata. Assicurati che l'immagine non sia vuota o danneggiata.")
        
    os.unlink(image_path)
            
else:
    st.info("⬆️ Carica una foto per iniziare la sonificazione!")
    st.markdown("""
    ### Come funziona:
    1. **Carica una foto** (JPG, PNG).
    2. L'applicazione analizzerà le **tonalità (Hue)** della tua immagine, dividendole in fasce.
    3. **Verranno mostrati istogrammi e una tabella** con la percentuale di ogni fascia di tonalità e la frequenza sonora associata.
    4. Clicca su "Genera Suono dai Colori" per creare un **suono combinato** (un accordo) che rappresenta la distribuzione dei colori della tua immagine, della durata desiderata.
    5. Potrai ascoltare e scaricare il suono generato!
    """)

# Footer
st.markdown("---")
st.markdown("🎨🎵 **Sonificazione dei Colori** - Trasforma le immagini in suoni!")
st.markdown("*💡 Esplora il legame tra luce e suono.*")
