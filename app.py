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
        list: Una lista di tuple (frequency, amplitude_weight) per l'accordo.
    """
    try:
        img = Image.open(image_path).convert('L') # Converti in scala di grigi (Luminosità)
        img_array = np.array(img)
        
        # Calcola l'istogramma delle intensità dei pixel
        # range è (0, 256) perché l'ultimo valore non è inclusivo in np.histogram
        hist, bin_edges = np.histogram(img_array.flatten(), bins=n_bins, range=(0, 256))
        
        # Normalizza l'istogramma per ottenere le percentuali
        hist_normalized = hist / np.sum(hist)
        
        frequencies_and_weights = []
        for i in range(n_bins):
            if hist_normalized[i] > 0: # Ignora le fasce senza pixel
                # Calcola il punto medio di ciascun bin per la mappatura
                # Esempio: bin_edges[i] è l'inizio della fascia, bin_edges[i+1] è la fine
                bin_midpoint_intensity = (bin_edges[i] + bin_edges[i+1]) / 2
                
                # Mappa l'intensità del punto medio al range di frequenze
                # Usiamo una mappatura lineare o potremmo sperimentare con logaritmica
                normalized_intensity = bin_midpoint_intensity / 255.0
                
                # Mappatura logaritmica per una percezione più naturale delle frequenze
                # Evita log(0)
                if normalized_intensity == 0:
                    normalized_intensity = 0.001 
                
                frequency = min_freq * ((max_freq / min_freq)**normalized_intensity)
                
                # Il peso (amplitude_weight) è la percentuale di pixel in quella fascia
                amplitude_weight = hist_normalized[i]
                
                frequencies_and_weights.append((frequency, amplitude_weight))
        
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
    total_weight = sum(w for f, w in frequencies_and_weights)
    if total_weight == 0:
        return np.zeros_like(t, dtype=np.float32) # Evita divisione per zero
    
    for freq, weight in frequencies_and_weights:
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
        min_freq_input = st.number_input("Frequenza Minima (Hz)", min_value=20, max_value=20000, value=50, key="min_f")
    with col2:
        max_freq_input = st.number_input("Frequenza Massima (Hz)", min_value=20, max_value=20000, value=2000, key="max_f")
    
    # Nuovo controllo per il numero di fasce di colore
    n_bins_input = st.slider("Numero di Fasce di Colore", 1, 10, 5, 1, 
                             help="Più fasce = più frequenze diverse nel suono (suono più ricco). Meno fasce = suono più semplice.")
    
    if min_freq_input >= max_freq_input:
        st.warning("La Frequenza Minima deve essere inferiore alla Frequenza Massima.")
        
    if st.button("🎵 Genera Suono dai Colori"):
        with st.spinner("Analizzando i colori e generando il suono..."):
            
            # 1. Analizza l'immagine e ottieni le frequenze e i pesi
            frequencies_and_weights = analyze_image_and_map_to_frequencies(
                image_path, min_freq_input, max_freq_input, n_bins_input
            )
            
            if frequencies_and_weights:
                st.success("Analisi completata!")
                st.markdown("### Frequenze associate ai colori:")
                
                # Visualizzazione delle frequenze e dei pesi
                freq_info = ""
                for freq, weight in frequencies_and_weights:
                    freq_info += f"- **{freq:.2f} Hz** (Peso: {weight*100:.1f}%) "
                    if freq <= 100: freq_info += "(Bassa) "
                    elif freq <= 800: freq_info += "(Media) "
                    else: freq_info += "(Alta) "
                    freq_info += "  \n" # Aggiungi nuova riga per markdown
                st.markdown(freq_info)
                
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
            
            # Pulizia del file immagine temporaneo
            os.unlink(image_path)
            
else:
    st.info("⬆️ Carica una foto per iniziare la sonificazione!")
    st.markdown("""
    ### Come funziona:
    1. **Carica una foto** (JPG, PNG).
    2. L'applicazione analizzerà l'**intensità dei colori** della tua immagine, dividendola in fasce (es. scuro, medio, chiaro).
    3. Ogni fascia di colore verrà mappata a una **frequenza sonora** all'interno del range che imposterai (di default 50Hz - 2000Hz).
    4. Verrà generato un **suono combinato** (un accordo) che rappresenta la distribuzione dei colori della tua immagine, della durata desiderata.
    5. Potrai ascoltare e scaricare il suono generato!
    """)

# Footer
st.markdown("---")
st.markdown("🎨🎵 **Sonificazione dei Colori** - Trasforma le immagini in suoni!")
st.markdown("*💡 Esplora il legame tra luce e suono.*")
