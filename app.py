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

# --- Funzione per l'analisi del colore e mappatura alla frequenza ---
def analyze_image_and_map_to_frequency(image_path, min_freq=50, max_freq=2000):
    """
    Analizza l'immagine, calcola l'intensità media e la mappa a una frequenza.
    
    Args:
        image_path (str): Percorso del file immagine.
        min_freq (int): Frequenza minima per la mappatura.
        max_freq (int): Frequenza massima per la mappatura.
        
    Returns:
        float: La frequenza calcolata in base all'intensità media dell'immagine.
    """
    try:
        img = Image.open(image_path).convert('L') # Converti in scala di grigi (Luminosità)
        img_array = np.array(img)
        
        # Calcola l'intensità media dei pixel (0=nero, 255=bianco)
        average_intensity = np.mean(img_array)
        
        # Mappa l'intensità media al range di frequenze
        # Una intensità di 0 (nero) mapperà a min_freq
        # Una intensità di 255 (bianco) mapperà a max_freq
        
        # Mappatura lineare:
        frequency = min_freq + (average_intensity / 255.0) * (max_freq - min_freq)
        
        return frequency
    except Exception as e:
        st.error(f"Errore nell'analisi dell'immagine: {e}")
        return None

# --- Funzione per generare un'onda sinusoidale ---
def generate_sine_wave(frequency, duration_seconds, sample_rate=44100):
    """
    Genera un'onda sinusoidale pura.
    
    Args:
        frequency (float): Frequenza dell'onda in Hz.
        duration_seconds (float): Durata dell'onda in secondi.
        sample_rate (int): Frequenza di campionamento in Hz.
        
    Returns:
        numpy.ndarray: Array di campioni audio normalizzato a -1.0 a 1.0.
    """
    t = np.linspace(0., duration_seconds, int(sample_rate * duration_seconds), endpoint=False)
    amplitude = np.sin(2 * np.pi * frequency * t)
    
    # Normalizza a float32 per compatibilità con i formati audio
    return amplitude.astype(np.float32)

# --- Sezione principale dell'app ---
uploaded_file = st.file_uploader("📸 Carica una foto", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Salva l'immagine temporaneamente per l'elaborazione
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_image_file:
        tmp_image_file.write(uploaded_file.read())
        image_path = tmp_image_file.name
    
    st.image(image_path, caption="Foto Caricata", use_container_width=True)
    
    st.markdown("### ⚙️ Impostazioni Audio")
    
    # Controlli per la durata del suono e il range di frequenze
    duration_input = st.slider("Durata del suono (secondi)", 0.5, 10.0, 2.0, 0.5)
    
    col1, col2 = st.columns(2)
    with col1:
        min_freq_input = st.number_input("Frequenza Minima (Hz)", min_value=20, max_value=20000, value=50)
    with col2:
        max_freq_input = st.number_input("Frequenza Massima (Hz)", min_value=20, max_value=20000, value=2000)
    
    if min_freq_input >= max_freq_input:
        st.warning("La Frequenza Minima deve essere inferiore alla Frequenza Massima.")
        
    if st.button("🎵 Genera Suono dai Colori"):
        with st.spinner("Analizzando i colori e generando il suono..."):
            
            # 1. Analizza l'immagine e ottieni la frequenza
            target_frequency = analyze_image_and_map_to_frequency(image_path, min_freq_input, max_freq_input)
            
            if target_frequency is not None:
                st.success(f"Frequenza calcolata: **{target_frequency:.2f} Hz**")
                
                # 2. Genera l'onda sonora
                audio_data = generate_sine_wave(target_frequency, duration_input)
                
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
            
            # Pulizia del file immagine temporaneo
            os.unlink(image_path)
            
else:
    st.info("⬆️ Carica una foto per iniziare la sonificazione!")
    st.markdown("""
    ### Come funziona:
    1. **Carica una foto** (JPG, PNG).
    2. L'applicazione analizzerà l'**intensità media dei colori** della tua immagine (dal nero al bianco).
    3. Questa intensità verrà mappata a una **frequenza sonora** all'interno del range che imposterai (di default 50Hz - 2000Hz).
    4. Verrà generato un **tono puro** (onda sinusoidale) con quella frequenza, della durata desiderata.
    5. Potrai ascoltare e scaricare il suono generato!
    """)

# Footer
st.markdown("---")
st.markdown("🎨🎵 **Sonificazione dei Colori** - Trasforma le immagini in suoni!")
st.markdown("*💡 Esplora il legame tra luce e suono.*")
