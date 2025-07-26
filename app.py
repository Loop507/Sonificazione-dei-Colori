import streamlit as st
from PIL import Image
import numpy as np
import scipy.io.wavfile as wavfile
import tempfile
import os

# Configurazione della pagina
st.set_page_config(page_title="🎨🎵 Sonificazione dei Colori by loop507", layout="centered")

st.markdown("<h1>🎨🎵 Sonificazione dei Colori <span style='font-size:0.5em;'>by loop507</span></h1>", unsafe_allow_html=True)
st.write("Carica una foto e genera un suono basato sui suoi colori!")

# --- Funzione per l'analisi del colore e mappatura alle frequenze ---
def analyze_image_and_map_to_frequencies(image_path, min_freq=50, max_freq=2000, n_bins=5):
    """
    Analizza l'immagine, divide l'intensità in fasce (bins) e mappa ciascuna a una frequenza.
    
    Args:
        image_path (str): Percorso del file immagine.
        min_freq (int): Frequenza minima per la mappatura.
        max_freq (int): Frequenza massima per la mappatura.
        n_bins (int): Numero di fasce di intensità (colori) da analizzare.
        
    Returns:
        list: Una lista di tuple (frequency, amplitude_weight, bin_start_intensity, bin_end_intensity) per l'accordo.
    """
    try:
        img = Image.open(image_path).convert('L') # Converti in scala di grigi (Luminosità)
        img_array = np.array(img)
        
        # Calcola l'istogramma delle intensità dei pixel
        # range è (0, 256) perché l'ultimo valore non è inclusivo in np.histogram
        hist, bin_edges = np.histogram(img_array.flatten(), bins=n_bins, range=(0, 256))
        
        # Normalizza l'istogramma per ottenere le percentuali
        total_pixels = np.sum(hist)
        hist_normalized = hist / total_pixels if total_pixels > 0 else np.zeros_like(hist)
        
        frequencies_and_weights = []
        for i in range(n_bins):
            # Calcola il punto medio di ciascun bin per la mappatura
            bin_midpoint_intensity = (bin_edges[i] + bin_edges[i+1]) / 2
            
            # Mappa l'intensità del punto medio al range di frequenze
            normalized_intensity = bin_midpoint_intensity / 255.0
            
            # Mappatura logaritmica per una percezione più naturale delle frequenze
            # Evita log(0) e divisione per zero
            if min_freq == 0: min_freq = 1 # Prevent issues with log scale if min_freq is 0
            if normalized_intensity == 0:
                normalized_intensity = 0.001 
            
            # Ensure max_freq / min_freq is not zero or negative for power
            freq_ratio = max_freq / min_freq if min_freq != 0 else 1
            if freq_ratio <= 0: freq_ratio = 1 # Avoid issues with non-positive base for power
            
            frequency = min_freq * (freq_ratio**normalized_intensity)
            
            # Il peso (amplitude_weight) è la percentuale di pixel in quella fascia
            amplitude_weight = hist_normalized[i]
            
            frequencies_and_weights.append((frequency, amplitude_weight, int(bin_edges[i]), int(bin_edges[i+1]-1))) # Aggiungi range intensità
        
        return frequencies_and_weights
    except Exception as e:
        st.error(f"Errore nell'analisi dell'immagine: {e}")
        return []

# --- Funzione per generare un'onda sinusoidale o un accordo ---
def generate_audio_wave(frequencies_and_weights, duration_seconds, sample_rate=44100):
    """
    Genera un'onda sonora combinando più sinusoidi.
    
    Args:
        frequencies_and_weights (list): Lista di tuple (frequency, amplitude_weight).
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
    total_weight = sum(w for f, w, _, _ in frequencies_and_weights) # Ora ha 4 elementi
    if total_weight == 0:
        return np.zeros_like(t, dtype=np.float32) # Evita divisione per zero
    
    for freq, weight, _, _ in frequencies_and_weights: # Estrai solo freq e weight per la generazione
        # L'ampiezza di ciascuna onda è proporzionale al suo peso normalizzato
        amplitude = np.sin(2 * np.pi * freq * t) * (weight / total_weight)
        combined_amplitude += amplitude
        
    # Normalizza l'intera onda combinata per evitare clipping e rimanere nel range -1.0 a 1.0
    if np.max(np.abs(combined_amplitude)) > 0:
        combined_amplitude /= np.max(np.abs(combined_amplitude))
        
    return combined_amplitude

# --- Sezione principale dell'app ---
uploaded_file = st.file_uploader("📸 Carica una foto", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Salva l'immagine temporaneamente per l'elaborazione
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_image_file:
        tmp_image_file.write(uploaded_file.read())
        image_path = tmp_image_file.name
    
    st.image(image_path, caption="Foto Caricata", use_container_width=True)
    
    st.markdown("### ⚙️ Impostazioni Sonificazione")
    
    # Controlli per la durata del suono e il range di frequenze
    duration_input = st.slider("Durata del suono (secondi)", 0.5, 10.0, 2.0, 0.5)
    
    col1, col2 = st.columns(2)
    with col1:
        min_freq_input = st.number_input("Frequenza Minima (Hz)", min_value=1, max_value=20000, value=50, key="min_f")
    with col2:
        max_freq_input = st.number_input("Frequenza Massima (Hz)", min_value=1, max_value=20000, value=2000, key="max_f")
    
    # Nuovo controllo per il numero di fasce di colore
    n_bins_input = st.slider("Numero di Fasce di Colore", 1, 10, 5, 1, 
                             help="Più fasce = più frequenze diverse nel suono (suono più ricco). Meno fasce = suono più semplice.")
    
    if min_freq_input >= max_freq_input:
        st.warning("La Frequenza Minima deve essere inferiore o uguale alla Frequenza Massima.")
    
    # --- Sezione di analisi e visualizzazione immediata ---
    st.markdown("### 📊 Analisi Colori e Frequenze Associate:")
    
    with st.spinner("Analizzando i colori della foto..."):
        # Analizza l'immagine e ottieni le frequenze e i pesi
        # Questa parte ora è fuori dal pulsante
        frequencies_and_weights = analyze_image_and_map_to_frequencies(
            image_path, min_freq_input, max_freq_input, n_bins_input
        )
        
        if frequencies_and_weights:
            st.success("Analisi dei colori completata!")
            
            # Creazione di una tabella o di un markdown strutturato per la visualizzazione
            st.markdown("| Intensità Colore (0-255) | Percentuale | Frequenza Associata (Hz) | Tipo Frequenza |")
            st.markdown("|:-----------------------:|:-----------:|:--------------------------:|:--------------:|")
            
            for freq, weight, bin_start, bin_end in frequencies_and_weights:
                color_range_str = f"{bin_start}-{bin_end}"
                percentage_str = f"{weight*100:.1f}%"
                frequency_str = f"{freq:.2f}"
                
                freq_type = ""
                if freq <= 100: freq_type = "Bassa"
                elif freq <= 800: freq_type = "Media"
                else: freq_type = "Alta"
                
                st.markdown(f"| {color_range_str} | {percentage_str} | {frequency_str} | {freq_type} |")
            
            # --- Pulsante di generazione suono (rimane separato) ---
            st.markdown("---")
            if st.button("🎵 Genera Suono dai Colori"):
                with st.spinner("Generando il suono..."):
                    # 2. Genera l'onda sonora combinata
                    audio_data = generate_audio_wave(frequencies_and_weights, duration_input)
                    
                    # Scala l'audio per un volume ragionevole (int16 per WAV standard)
                    audio_data_int16 = (audio_data * 32767).astype(np.int16) 
                    
                    # 3. Salva l'audio in un file temporaneo
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_audio_file:
                        audio_output_path = tmp_audio_file.name
                        wavfile.write(audio_output_path, 44100, audio_data_int16) # 44100 è una sample rate standard
                    
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
        
        # Pulizia del file immagine temporaneo, assicurandosi che sia fatto solo una volta
        # quando l'intero blocco `if uploaded_file is not None` è finito.
        # Per ora la lascio alla fine dell'intero blocco `if uploaded_file is not None`
        # per evitare problemi con la rianalisi quando si cambiano i parametri.
    os.unlink(image_path)
            
else:
    st.info("⬆️ Carica una foto per iniziare la sonificazione!")
    st.markdown("""
    ### Come funziona:
    1. **Carica una foto** (JPG, PNG).
    2. L'applicazione analizzerà l'**intensità dei colori** della tua immagine, dividendola in fasce (es. scuro, medio, chiaro).
    3. **Verrà mostrata una tabella** con la percentuale di ogni fascia di colore e la frequenza sonora associata.
    4. Clicca su "Genera Suono dai Colori" per creare un **suono combinato** (un accordo) che rappresenta la distribuzione dei colori della tua immagine, della durata desiderata.
    5. Potrai ascoltare e scaricare il suono generato!
    """)

# Footer
st.markdown("---")
st.markdown("🎨🎵 **Sonificazione dei Colori** - Trasforma le immagini in suoni!")
st.markdown("*💡 Esplora il legame tra luce e suono.*")
