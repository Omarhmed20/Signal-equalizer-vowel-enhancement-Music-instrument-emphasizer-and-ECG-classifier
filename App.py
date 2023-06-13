import plotly.graph_objects as go
import pandas as pd
import numpy as np
import streamlit as st
from plotly.subplots import make_subplots
import plotly.io as pio
from PyPDF2 import PdfFileReader, PdfFileMerger
import PyPDF2
import librosa
import functions
import plotly.express as px
from scipy.fft import fft, fftfreq, ifft
import  streamlit_vertical_slider  as svs
from scipy.signal import spectrogram
from scipy import signal

from numpy.fft import rfft, rfftfreq, irfft


st. set_page_config(layout="wide")
pio.templates.default = "simple_white"
#####################################SIGNAL VIEWER#########################

with open("mystyle.css") as design:
    st.markdown(f"<style>{design.read()}</style>", unsafe_allow_html=True)
audio =True

side=st.sidebar
side.title("Equalizer")
g1Frames =[]
with side:

    st.subheader("Type")
    Mode = st.selectbox(label="", options=[
                        'Frequency', 'Vowels', 'Music Instrument', 'Medical'])
    
    
x = []

y= []



    #####################################End SIGNAL VIEWER#########################

import wave
from scipy.io import wavfile
from pydub import AudioSegment
from pydub.playback import play
    # Display a file uploader widget
uploaded_file = side.file_uploader("Choose a file", type=set(['csv', 'wav']))
if(Mode=="Medical"):
     uploaded_file2 = side.file_uploader("Choose a file", type=set(['csv']))

if (Mode!="Medical" and uploaded_file!=None):
    magnitude_at_time, sample_rate = librosa.load(uploaded_file)
    time_before = np.array(range(0, len(magnitude_at_time+1)))/(sample_rate)
pitch_step = 0
sample_freq=0
n_samples=0
samples_fft=[]
Sxx_db=[]
Sxx=[]
import sounddevice as sd
import soundfile as sf



# Check if a file was uploaded
if uploaded_file is not None:

    
    
# Downmix the stereo signal to mono

    if uploaded_file.type == "text/csv":
        File = pd.read_csv(uploaded_file)
    
    else:   
        File=0
        signal_array, sample_rate = librosa.load('C:\Files\DSP\Audios\\'+uploaded_file.name, sr=None, mono=False)
        signal_array = librosa.to_mono(signal_array)
        signal_array = librosa.util.normalize(signal_array)

        sf.write("C:\Files\DSP\Audios\Original.wav", signal_array ,sample_rate )
        samples,sample_rate=sf.read("C:\Files\DSP\Audios\Original.wav")

        #sample_rate, samples = wavfile.read('Audios\\' + uploaded_file.name)

        wav_obj = wave.open("C:\Files\DSP\Audios\Original.wav", 'rb')
        sample_freq = wav_obj.getframerate()
        n_samples = wav_obj.getnframes()
        t_audio = n_samples/sample_freq
        n_channels = wav_obj.getnchannels()
        signal_wave = wav_obj.readframes(n_samples)
        signal_array = np.frombuffer(signal_wave, dtype=np.int16)
        times = np.linspace(0, n_samples/sample_freq, num=n_samples)

        
   

   
if uploaded_file is not None:

    if (Mode=="Medical"):
    # File = pd.read_csv(uploaded_file)
        import wfdb
        from wfdb.processing import resample_singlechan
    
        record = wfdb.rdrecord('100')
        
        # Extract the ECG signal and sampling rate
        y_plots= record.p_signal[0:6000, 0]  # Assuming single-channel ECG
        
        x_plots=np.arange(len(y_plots))


    else:
        x_plots=times
        y_plots=signal_array

    
    # if(Mode=="Medical"):
    #       SpectroMag=Sxx

    # else:
    #       SpectroMag=Sxx_db

    

    # if(Mode!="Medical"):

        
    #     # signal = librosa.util.normalize(signal)
    #     # window = np.hamming(len(signal))
    #     # mono_signal_win = signal * window
    #     samples_fft = fft(signal_array.astype(np.float64))

    #st.session_state['mag_db'] = 10 * np.log10(np.abs(samples_fft))
    # Sxx_db = 10 * np.log10(np.abs(Sxx))
    # ##########Uniform Range Mode
    if Mode == 'Frequency':
                    sliders_freq_values = {"0:1000": [[0, 1000]],
                                        "1000:2000": [[1000, 2000]],
                                        "2000:3000": [[2000, 3000]],
                                        "3000:4000": [[3000, 4000]],
                                        "4000:5000": [[4000, 5000]],
                                        "5000:6000": [[5000, 6000]],
                                        "6000:7000": [[6000, 7000]],
                                        "7000:8000": [[7000, 8000]],
                                        "8000:9000": [[8000, 9000]],
                                        "9000:10000": [[9000, 10000]]
                                        }
                    values_slider = [[0, 10, 1]]*len(list(sliders_freq_values.keys()))

    elif Mode == 'Vowels':
        sliders_freq_values = {
                            "R": [[1500, 3000]],
                            "O": [[500, 2000]],
                            "L": [[500, 2000]],
                            "I": [[500, 2500]]
                            }
        values_slider = [[0, 10, 1]]*len(list(sliders_freq_values.keys()))
#
    elif Mode == 'Music Instrument':
        sliders_freq_values = {
                            "Flute": [[2300, 2500], [600, 900],[1500, 1800],[3900,4100],[3000,3200]],


                            "Piano": [[2000, 2200],[900, 1200],[2900, 3200]],

                            "violin": [[2900, 3200],[1400, 1600],[4500, 4800]],
                            }
        values_slider = [[0, 10, 1]]*len(list(sliders_freq_values.keys()))

    elif Mode == 'Medical':
        
        sliders_freq_values = {"Sinus bradycardia Amplitude": [0.67, 1.0],
                               "Sinus tachycardia Amplitude": [1.7, 2.5],
                               "Ventricular tachycardia Amplitude": [2.5, 4.0],
                                "AtriaL Fibrillation Amplitude ": [5.0, 10.0],
                                "AtriaL  ": [5.0, 10.0]
                            
                            }
        values_slider = [[-2.0, 2.0, 0.1]]*len(list(sliders_freq_values.keys()))



    # if(Mode!="Medical"):
    #     functions.processing(Mode, list(sliders_freq_values.keys()), values_slider, magnitude_at_time,
    #                     sample_rate, st.session_state.show_spec, sliders_freq_values)

    mode_sliders=functions.create_sliders(values_slider,sliders_freq_values)

    if(Mode!="Medical"):
        modified_magnitude=functions.RemoveFreq(File,Mode,list(sliders_freq_values.keys()), uploaded_file, mode_sliders,sliders_freq_values)
    File2=0
    if(Mode=="Medical"):
        File2 = pd.read_csv(uploaded_file2)
        modified_magnitude=functions.MedRemoveFreq(File,File2,mode_sliders)

    print(modified_magnitude)
        

    
    if(Mode!="Medical"):
        if Mode == "Music Instrument":
           sample_rate= sample_rate*2
        if Mode == "Frequency":
           sample_rate= int(sample_rate/2)  
        #time_after = np.array(range(0, len(modified_magnitude+1)))/(sample_rate)
        time_after = np.arange(len(modified_magnitude))
        functions.DrawGraph(time_before,time_after, magnitude_at_time,modified_magnitude)



        functions.audio_viewer(uploaded_file, modified_magnitude, sample_rate,Mode)

    else:
        import wfdb
        from wfdb.processing import resample_singlechan
    
        record = wfdb.rdrecord('100')
        
        # Extract the ECG signal and sampling rate
        y_plots2 = record.p_signal[0:6000, 0]  # Assuming single-channel ECG
        
        
        x_plots2=np.arange(len(y_plots2))
        functions.DrawGraph(x_plots,x_plots2,modified_magnitude,y_plots2)