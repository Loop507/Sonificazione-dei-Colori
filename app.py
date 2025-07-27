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
st.set_page_config(page_title="🎨🎵 Color Sonification by loop507", layout="centered")

st.markdown("<h1>🎨🎵 Color Sonification <span style='font-size:0.5em;'>by loop507</span></h1>", unsafe_allow_html=True)
st.write("Upload a photo and generate a sound based on its colors!")

# --- Mappatura delle Frequenze per Classificazione Colore ---
def get_frequency_for_color_class(h_deg, s_val, v_val):
    """
    Classifica un colore HSV medio e restituisce una frequenza discreta o interpolata.
    h_deg: Hue in gradi (0-360)
    s_val: Saturation (0.0-1.0)
    v_val: Value (0.0-1.0)
    """
    
    # 1. Priorità: Colori Acromatici (Bianco, Nero, Grigio) - Basati su Saturazione e Luminosità
    if s_val < 0.15: # Low saturation indicates achromatic colors
        if v_val > 0.9: # Very bright = White
            return 2000 # Frequency for White
        elif v_val < 0.1: # Very dark = Black
            return 20 # Frequency for Black
        else: # Intermediate brightness, low saturation = Gray
            return 200 # Frequency for Gray
            
    # 2. Priorità: Colori Speciali che dipendono molto da Saturazione/Valore
    
    # Light Yellow (Yellow hue, very high brightness)
    if 45 <= h_deg < 75 and v_val > 0.8:
        return 1950 # Frequency for Light Yellow
        
    # Pink (Red/magenta hue, high brightness, medium/low saturation)
    if (h_deg >= 330 or h_deg < 20) and s_val > 0.15 and v_val > 0.6 and s_val < 0.6:
        return 1150
        
    # Brown (Orange/red hue, low brightness, medium/high saturation)
    if (20 <= h_deg < 60 or h_deg >= 340 or h_deg < 20) and s_val > 0.2 and v_val < 0.4:
        return 300
    
    # 3. Interpolation based on Hue for Standard Chromatic Colors
    hue_freq_anchors = [
        (0, 700),    # Red
        (60, 1900),  # Yellow
        (120, 1300), # Green
        (180, 1600), # Cyan
        (240, 400),  # Blue
        (300, 1000), # Magenta
        (360, 700)   # Red (to close the circle)
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
        return 700 # Red

    h1, f1 = hue_freq_anchors[idx1]
    h2, f2 = hue_freq_anchors[idx2]

    if (h2 - h1) == 0: 
        return f1
    
    interpolation_factor = (h_deg_for_interp - h1) / (h2 - h1)
    
    interpolated_frequency = f1 + (f2 - f1) * interpolation_factor
    
    return interpolated_frequency

# --- Funzione per l'analisi del colore ---
def analyze_image_and_map_to_frequencies(image_path, n_bins=5):
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
                frequency = 20 # Default min freq
                avg_v_val = 0 # Default low value
            
            all_bin_actual_colors_hex.append(actual_hex_color_for_bin)

            if hist_normalized[i] > 0: 
                amplitude_weight = hist_normalized[i]
                
                frequencies_and_weights.append((frequency, amplitude_weight, int(bin_edges_hist[i]), int(bin_edges_hist[i+1]), actual_hex_color_for_bin, avg_v_val))
            
        return frequencies_and_weights, hist_normalized, bin_edges_hist, all_bin_actual_colors_hex
    except Exception as e:
        st.error(f"Error analyzing image: {e}")
        return [], np.array([]), np.array([]), []

# --- Funzione per generare un'onda sinusoidale o un accordo ---
def generate_audio_wave(frequencies_and_weights_with_vval, duration_seconds, sample_rate=44100, 
                       waveform_mode="single", single_waveform_type="sine", 
                       bright_wave="sine", medium_wave="square", dark_wave="sawtooth"):
    
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
                if v_val > 0.7: # Bright Colors
                    current_waveform_func = get_waveform_function(bright_wave)
                elif v_val < 0.3: # Dark Colors
                    current_waveform_func = get_waveform_function(dark_wave)
                else: # Medium Colors
                    current_waveform_func = get_waveform_function(medium_wave)
            
            if current_waveform_func is None:
                 current_waveform_func = get_waveform_function("sine")

            amplitude = current_waveform_func(t, freq) * (weight / total_weight)
            combined_amplitude += amplitude
        
    max_amplitude = np.max(np.abs(combined_amplitude))
    if max_amplitude > 0:
        combined_amplitude /= max_amplitude
        
    return combined_amplitude

# --- Sezione principale dell'app ---
uploaded_file = st.file_uploader("📸 Upload a photo", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_image_file:
        tmp_image_file.write(uploaded_file.read())
        image_path = tmp_image_file.name
    
    st.image(image_path, caption="Uploaded Photo", use_container_width=True)
    
    st.markdown("### ⚙️ Sonification Settings")
    
    duration_input = st.slider("Sound Duration (seconds)", 0.5, 10.0, 2.0, 0.5)
    
    col1, col2 = st.columns(2)
    with col1:
        min_freq_input = st.number_input("Minimum Frequency (Hz)", min_value=1, max_value=20000, value=20, key="min_f")
    with col2:
        max_freq_input = st.number_input("Maximum Frequency (Hz)", min_value=1, max_value=20000, value=2000, key="max_f")
    
    n_bins_input = st.slider("Number of Color Bins (Hue)", 1, 10, 5, 1, 
                             help="More bins = more distinct frequencies in the sound (richer sound). Fewer bins = simpler sound.")
    
    if min_freq_input >= max_freq_input:
        st.warning("Minimum Frequency must be less than Maximum Frequency.")
        
    # --- New controls for Waveform Type ---
    st.markdown("### 🎶 Sound Waveform Type")
    waveform_selection_mode = st.radio(
        "How do you want to generate the waveforms?",
        ["Single Waveform for all Colors", "Waveform by Color Brightness"],
        key="waveform_mode_selector"
    )

    selected_single_waveform = "Sine" # Default
    if waveform_selection_mode == "Single Waveform for all Colors":
        selected_single_waveform = st.radio(
            "Choose the waveform type for all frequencies:",
            ["Sine", "Square", "Sawtooth", "Mixed (Sine + Square + Sawtooth)"],
            key="single_waveform_type"
        )
    
    bright_wave_type = "sine"
    medium_wave_type = "square"
    dark_wave_type = "sawtooth"

    if waveform_selection_mode == "Waveform by Color Brightness":
        st.markdown("---")
        st.markdown("#### Assign Waveform by Brightness:")
        col_bright, col_medium, col_dark = st.columns(3)
        with col_bright:
            bright_wave_type = st.selectbox(
                "Bright Colors (High Brightness):",
                ["sine", "square", "sawtooth"],
                index=0, # Default to sine
                key="bright_wave_type"
            )
        with col_medium:
            medium_wave_type = st.selectbox(
                "Medium Colors (Medium Brightness):",
                ["sine", "square", "sawtooth"],
                index=1, # Default to square
                key="medium_wave_type"
            )
        with col_dark:
            dark_wave_type = st.selectbox(
                "Dark Colors (Low Brightness):",
                ["sine", "square", "sawtooth"],
                index=2, # Default to sawtooth
                key="dark_wave_type"
            )
        st.markdown("---")


    # --- Analysis and immediate visualization section ---
    st.markdown("### 📊 Color Analysis and Associated Frequencies:")
    
    with st.spinner("Analyzing image colors..."):
        frequencies_and_weights_with_vval, hist_normalized, bin_edges, all_bin_actual_colors_hex = analyze_image_and_map_to_frequencies(
            image_path, n_bins_input
        )
        
        if frequencies_and_weights_with_vval or (hist_normalized.size > 0 and np.sum(hist_normalized) > 0): 
            st.success("Color analysis complete!")
            
            col_chart1, col_chart2 = st.columns(2)

            with col_chart1:
                st.markdown("#### Hue Distribution")
                fig_color, ax_color = plt.subplots(figsize=(6, 4))
                hue_bin_labels = [f"{int(bin_edges[i])}°-{int(bin_edges[i+1])}°" for i in range(len(bin_edges)-1)]
                
                ax_color.bar(hue_bin_labels, hist_normalized * 100, color=all_bin_actual_colors_hex) 
                ax_color.set_xlabel("Hue Band (degrees)")
                ax_color.set_ylabel("Percentage (%)")
                ax_color.set_title("Pixel Percentage per Hue Band")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                st.pyplot(fig_color)
                plt.close(fig_color) 

            with col_chart2:
                st.markdown("#### Generated Frequencies and Weight")
                freq_labels = [f"{f:.0f} Hz" for f, w, _, _, _, _ in frequencies_and_weights_with_vval]
                freq_weights = [w * 100 for f, w, _, _, _, _ in frequencies_and_weights_with_vval]
                
                bar_colors_freq = [item[4] for item in frequencies_and_weights_with_vval]

                fig_freq, ax_freq = plt.subplots(figsize=(6, 4))
                ax_freq.bar(freq_labels, freq_weights, color=bar_colors_freq)
                ax_freq.set_xlabel("Frequency (Hz)")
                ax_ylabel("Weight in Chord (%)")
                ax_freq.set_title("Frequencies and their Weight in Sound")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                st.pyplot(fig_freq)
                plt.close(fig_freq) 
                
            st.markdown("---")
            st.markdown("#### Frequency Detail Table:")
            st.markdown("| Hue Band | Percentage | Associated Frequency (Hz) | Brightness (0-1) | Frequency Type |")
            st.markdown("|:----------------------:|:-----------:|:--------------------------:|:----------------:|:--------------:|")
            
            for freq, weight, hue_start, hue_end, rep_hex, v_val in frequencies_and_weights_with_vval:
                hue_range_str = f"<span style='background-color:{rep_hex}; padding: 2px 5px; border-radius:3px;'>&nbsp;&nbsp;&nbsp;</span> {hue_start}°-{hue_end}°"
                percentage_str = f"{weight*100:.1f}%"
                frequency_str = f"{freq:.2f}"
                brightness_str = f"{v_val:.2f}" 
                
                freq_type = ""
                if freq < 200: freq_type = "Very Low" 
                elif freq < 500: freq_type = "Low"
                elif freq < 800: freq_type = "Medium-Low"
                elif freq < 1200: freq_type = "Medium"
                elif freq < 1800: freq_type = "Medium-High"
                elif freq < 2000: freq_type = "High"
                else: freq_type = "Very High"
                
                st.markdown(f"| {hue_range_str} | {percentage_str} | {frequency_str} | {brightness_str} | {freq_type} |", unsafe_allow_html=True)
            
            st.markdown("---")
            
            # --- General Mapping Explanation ---
            st.markdown("### 🔍 How Colors Become Sounds:")
            st.markdown(f"""
            This application analyzes the colors in your image, classifying them and assigning a sound frequency.
            
            We have defined reference frequencies for "pure" colors (e.g., Red, Yellow, Blue) and for achromatic colors (Black, White, Gray).
            For "mixed" colors (like orange, which is a mix of red and yellow), the frequency is **interpolated**
            between the frequencies of its "pure" neighboring colors on the color wheel, creating a "sonic fusion".
            
            **New Feature: Sound Waveform Type!**
            In addition to frequency (which determines the pitch of the sound), you can now also choose the **timbre** (the "quality" of the sound)
            by selecting different waveform types:
            
            * **Sine Wave:** The purest sound, with no overtones. Sounds soft and "sweet".
            * **Square Wave:** Contains only odd harmonics. Has a more "hollow" sound, similar to a clarinet or some synthesizers.
            * **Sawtooth Wave:** Contains all harmonics. Has a brighter sound, similar to a violin or brass instruments.
            
            You can choose a single waveform for all colors, a mix of all three, or let the application
            assign the waveform based on the color's brightness, with custom assignments:
            * **Bright Colors (High Brightness):** You choose the waveform.
            * **Medium Colors (Medium Brightness):** You choose the waveform.
            * **Dark Colors (Low Brightness):** You choose the waveform.
            
            """)
            
            st.markdown("#### Example: Color Interpolation ➡️ Frequency")
            hue_gradient_html = """
            <div style="width:100%; height:30px; 
                        background: linear-gradient(to right, 
                        #FF0000, #FF8000, #FFFF00, #80FF00, #00FF00, #00FF80, #00FFFF, #0080FF, #0000FF, #8000FF, #FF00FF, #FF0080, #FF0000);">
            </div>
            <div style="display:flex; justify-content:space-between; font-size:0.8em; flex-wrap: wrap;">
                <span>Red (700Hz)</span>
                <span>Orange (Interp.)</span>
                <span>Yellow (1900Hz)</span>
                <span>Lime (Interp.)</span>
                <span>Green (1300Hz)</span>
                <span>Turquoise (Interp.)</span>
                <span>Cyan (1600Hz)</span>
                <span>Azure (Interp.)</span>
                <span>Blue (400Hz)</span>
                <span>Violet (Interp.)</span>
                <span>Magenta (1000Hz)</span>
                <span>Rose (Interp.)</span>
                <span>Red (700Hz)</span>
            </div>
            """
            st.markdown(hue_gradient_html, unsafe_allow_html=True)
            
            st.markdown("---")


            # --- Sound generation button ---
            if st.button("🎵 Generate Sound from Colors"):
                with st.spinner("Generating sound..."):
                    audio_data = None
                    if waveform_selection_mode == "Single Waveform for all Colors":
                        if selected_single_waveform == "Mixed (Sine + Square + Sawtooth)":
                            audio_data = generate_audio_wave(frequencies_and_weights_with_vval, duration_input, 
                                                            waveform_mode="mixed_all")
                        else:
                            # Map readable name to internal code name
                            waveform_map_internal = {
                                "Sine": "sine",
                                "Square": "square",
                                "Sawtooth": "sawtooth"
                            }
                            audio_data = generate_audio_wave(frequencies_and_weights_with_vval, duration_input, 
                                                            waveform_mode="single", 
                                                            single_waveform_type=waveform_map_internal[selected_single_waveform])
                    else: # Waveform by Color Brightness
                        audio_data = generate_audio_wave(frequencies_and_weights_with_vval, duration_input, 
                                                        waveform_mode="by_brightness",
                                                        bright_wave=bright_wave_type,
                                                        medium_wave=medium_wave_type,
                                                        dark_wave=dark_wave_type)

                    if audio_data is not None:
                        audio_data_int16 = (audio_data * 32767).astype(np.int16) 
                        
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_audio_file:
                            audio_output_path = tmp_audio_file.name
                            wavfile.write(audio_output_path, 44100, audio_data_int16) 
                        
                        st.markdown("### Listen to your Sound:")
                        st.audio(audio_output_path, format='audio/wav')
                        
                        st.download_button(
                            label="⬇️ Download generated sound",
                            data=open(audio_output_path, 'rb').read(),
                            file_name="color_sound.wav",
                            mime="audio/wav"
                        )
                        
                        os.unlink(audio_output_path)
                    else:
                         st.error("❌ Error generating sound.")
            
        else:
            st.warning("No frequencies generated. Ensure the image is not empty or corrupted.")
        
    os.unlink(image_path)
            
else:
    st.info("⬆️ Upload a photo to start sonification!")
    st.markdown("""
    ### How it works:
    1.  **Upload a photo** (JPG, PNG).
    2.  The application will analyze the colors in your image, **interpolating frequencies** for mixed colors
        and assigning fixed frequencies for primary and achromatic colors.
    3.  **Choose the sound waveform type** you want to use: a single waveform for all frequencies, a mix of all,
        or an automatic assignment based on color brightness, with custom selections.
    4.  **Histograms and a table** will be displayed with the percentage of each color band and the associated sound frequency.
    5.  Click "Generate Sound from Colors" to create a **combined sound** (a chord) representing your image.
    6.  You can listen to and download the generated sound!
    """)

# Footer
st.markdown("---")
st.markdown("🎨🎵 **Color Sonification** - Transform images into sounds!")
st.markdown("*💡 Explore the link between light and sound.*")
