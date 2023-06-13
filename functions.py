import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import rfft, rfftfreq, irfft
import librosa.display
import librosa
import plotly.graph_objects as go
import streamlit_vertical_slider as svs
import streamlit as st
import pandas as pd
import soundfile as sf
import plotly.express as px
import  streamlit_vertical_slider  as svs


def MedRemoveFreq(file,file2,sliders):


    for i in range(len(sliders)):
        if sliders[i] == None:
            sliders[i] = 1

    #mag = np.array(file.iloc[:2000,0])
    import wfdb
    from wfdb.processing import resample_singlechan

    record = wfdb.rdrecord('114')
    
    # Extract the ECG signal and sampling rate
    mag= record.p_signal[0:6000, 0]  # Assuming single-channel ECG
        
    
    
    t =np.arange(len(mag))
    
    sample_rate=t[1]-t[0]
    mag_fft = rfft(mag)
    
   
    #mag2=np.array(file2.iloc[:2000,0])

    #sample_rate = record.fs

    from biosppy.signals import ecg
    out = ecg.ecg(signal=mag_fft, show=False)

# Extract relevant features
    rpeaks = out['rpeaks']  # R-peak locations
    templates = out['templates']  # Normal QRS complex templates

    # Simulate arrhythmias
    arrhythmia_ecg = np.copy(mag)  # Create a copy of the normal ECG signal

    # Example: Add a premature ventricular contraction (PVC) at a specific location
    pvc_location = rpeaks  # Replace with the desired PVC location
#     pvc_duration = 20  # Replace with the desired PVC duration

#     pvc_amplitude = np.max(mag)  # Amplitude of the PVC signal
#     pvc_signal = pvc_amplitude * np.random.normal(-0.5, 0.2, size=pvc_duration)  # Generate random PVC signal
#     #arrhythmia_ecg+=abs(min(arrhythmia_ecg))
#     for i in range(len(pvc_location)):
#         arrhythmia_ecg[pvc_location[i]:pvc_location[i] + pvc_duration] *= pvc_signal
#     arrhythmia_ecg*=-1

    # Open the .dat file
    print("DAT FILE")
    # import the WFDB package
    import wfdb
    from wfdb.processing import resample_singlechan
  
    record2 = wfdb.rdrecord('100')
    
    print("length record")
    print(record2)
    

       # Extract the ECG signal and sampling rate
    mag2 = record2.p_signal[0:6000, 0]  # Assuming single-channel ECG
    mag2_fft=rfft(mag2)
    difference=mag2_fft-mag_fft
    differencefreqs=rfftfreq(len(difference), sample_rate)
    frequencies = rfftfreq(len(t[0:2000]), sample_rate)
  
    print(len(frequencies))
    counter = 0
    print(difference)
    #DEcompose signal ecg

# Get the signal data and channel names
    mag_add=[20]*200
    counter=0
    # no=0
    # min_peak=[]
    # for i in range(len(mag_fft)):
    #     if(mag_fft[i]<-0.3):
    #         min_peak.append(mag_fft[i])
    # print('min_peak')
    # print(min_peak)
    #     # Switch the minimum peak to the maximum peak
    # while(no<6)  :  
    #     minimum=min(min_peak)
    #     min_peak=np.where(min_peak==minimum,100,min_peak)
    #     mag_fft = np.where(mag_fft == minimum, np.abs(minimum), mag_fft)
    from numpy.polynomial import Polynomial

    degree = 5  # Set the degree of the polynomial equation
    coefficients = Polynomial.fit(mag_fft, mag2_fft, degree)
    print("Polynomial coefficients:", coefficients)
    print(mag_fft)
    print(len(frequencies))
    print(len(mag_fft))
    import cmath
    for value in frequencies:
    
    #     if 0.5 >value>0.0:
    #        mag_fft[counter]= complex(3046.6943377119455, -1389.9170602341355) 
    # - (complex(12406.530143777643, -3564.578925827908) * mag_fft[counter] ** 1) 
    # + (complex(13041.694985494369, -1833.1540761752967) * mag_fft[counter] ** 2) 
    # + (complex(6160.764886820631, 757.3863538507666) * mag_fft[counter] ** 3) 
    # - (complex(16740.80251130649, 3955.401498459056) * mag_fft[counter] ** 4) 
    # + (complex(6829.315288229834, 2872.6349901962944) * mag_fft[counter] ** 5)
        if value==0.0:
            mag_fft[counter] =0

        # if value==0.:
        #     mag_fft[counter] += -200

        # if mag_fft[counter]<(-0.5):
        #     counter2=counter-10
        #     for i in range(counter2, counter2+20, 1):
        #         mag_fft[i] = abs(mag_fft[i]) 
        ####MUST####
        if 0.2>value > 0.0:
            mag_fft[counter] *= -5*sliders[0]
        if 0.5 >value>0.0:
            mag_fft[counter] *=     1*sliders[1]
        for i in range(len(pvc_location)):
            mag_fft[pvc_location[i]-20:pvc_location[i] + 20]=abs(mag_fft[pvc_location[i]-20:pvc_location[i] + 20])
            mag_fft[pvc_location[i]]*=0.003
        if mag_fft[counter]<(-200):
            mag_fft[counter]*=0.00001


        
       
        if(mag_fft[counter]>max(mag2_fft)):
            mag_fft[counter]=max(mag2_fft)

        if(mag_fft[counter]<min(mag2_fft)):
            mag_fft[counter]=min(mag2_fft)
        # for i in range(-15,-1,-1):
        #     if mag_fft[counter]>(-8) and mag_fft[counter+i]>(-8)and mag_fft[counter-i]>(-8):
        #         mag_fft[counter]*=2
        # if 0.5 >value>0.0:
        #     mag_fft[counter] *= complex(-10,-200)*sliders[1]
        # if 0.05 >value>0.01:
        #     mag_fft[counter] *= complex(-10,-200)*sliders[4]
        # if 0.095>value>0.08:
        #     mag_fft[counter] *= complex(10,-30)*sliders[2]
        # if 0.15>value > 0.0:
        #     mag_fft[counter] *= -20*sliders[2]
        # if 0.04>value > 0.02:
        #     mag_fft[counter] *= 2*sliders[3]
        
        # if 0.2>value > 0.0:
        #     mag_fft[counter] += 20*sliders[3]
        # if mag_fft[counter]<(-10):
        #     mag_fft[counter]=abs(mag_fft[counter])

        # if 0.015 >value>0.0:
        #     mag_fft[counter%(200)] += -100*(sliders[4])
        




        counter += 1


    modified_amplitude = np.real(irfft(mag_fft))
    modified_difference = np.real(irfft(difference))

    col1, col2 = st.columns(2)

    
# Create a polynomial function using the fitted coefficients

    with col1:
        
        st.pyplot(Medspectogram(modified_amplitude, sample_rate))
        
    with col2:

        st.pyplot(Medspectogram(mag2, sample_rate))
      
    return modified_amplitude




def audio_viewer(original_file, modified_magnitude, sample_rate,mode):

    st.sidebar.write("## Original Audio")
    st.sidebar.audio(original_file)
    st.sidebar.write("## Modified Audio")
    sf.write("Reconstructed.wav", modified_magnitude, sample_rate)
    st.sidebar.audio("Reconstructed.wav")



def spectogram(Signal_Samples):
    D = librosa.stft(Signal_Samples) 
    
    Signal_Samples_in_DB = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    fig, ax = plt.subplots(figsize=(8,4 ))
    fig.colorbar(librosa.display.specshow(Signal_Samples_in_DB,cmap='autumn', x_axis='time', y_axis='linear'))
    return plt.gcf()

def Medspectogram(Signal_Samples, sample_rate):
    
    fig, ax = plt.subplots()

    plt.specgram(np.real(Signal_Samples), Fs=1/sample_rate, cmap='autumn', mode='magnitude')
    plt.yscale('linear')

    
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')

    
    cbar = plt.colorbar()
    cbar.set_label('Magnitude')
    return plt.gcf()

def DrawGraph(x_time,x_time2, y_amplitude,modified_magnitude,):
    df = pd.DataFrame(dict(
        x = x_time,
        y = y_amplitude,
    )) #

    df2 = pd.DataFrame(dict(
        x = x_time2,
        y = modified_magnitude,
    ))  #
    
    ##################fn 
    col3, col4 = st.columns(2)
    
    with col3:
        graph = px.line(df, x="x", y="y")
        graph.update_layout(width=500, height=300
        )
        st.plotly_chart(graph)
    with col4:
        graph2 = px.line(df2, x="x", y="y")
        graph2.update_layout(width=500, height=300
    )
        st.plotly_chart(graph2)



def RemoveFreq(file,mode,name, uploaded_file, sliders, slider_freq_values):
    # if (mode!="Medical"):
    y, sr = librosa.load('C:\Files\DSP\Audios\\'+uploaded_file.name)
    number_samples = len(y)
    T = 1 / (sr)
    magnitude = rfft(y)
    frequency = rfftfreq(number_samples, T)
    # else:
       
    #     y=np.array(file.iloc[:,1])
    #     print(y)
    #     magnitude=rfft(y)
    #     print(magnitude)
    #     x=np.array(file.iloc[:,0])
    #     sr=(x[1]-x[0])
    #     frequency = rfftfreq(len(x), 1/sr)
    
    for i in range(len(sliders)):
        if sliders[i] is None:
            sliders[i] = 1

    for i in range(len(sliders)):
        counter = 0
        for value in frequency:
            for freq_range in slider_freq_values[name[i]]:
                if freq_range[0] < value < freq_range[1]:
                    magnitude[counter] *= sliders[i]
            counter += 1
    
    modified_magnitude = irfft(magnitude)
    print(len(magnitude))
    print(len(modified_magnitude))
    col1, col2 = st.columns(2)
    
    with col1:
       
        st.pyplot(spectogram(y))
    with col2:
        st.pyplot(spectogram(modified_magnitude))


    return np.real(modified_magnitude)


def create_sliders(value_slider,slider_freq_values):

   
    cols = st.columns(len(value_slider))

    sliders = []
    
    keys=[]
    for key, val in slider_freq_values.items():
        keys.append(key)
    for i,val in enumerate(value_slider):
        key_selected=keys[i]
        with cols[i]:
            slider_val = svs.vertical_slider(key=key_selected, min_value=val[0], max_value=val[1],default_value=1, step=0.1)
            st.text(key_selected)
            sliders.append(slider_val)
            
            
    return sliders