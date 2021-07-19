import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
import seaborn as sns
from tensorflow import keras
import matplotlib.pyplot as plt
import librosa
import librosa.display
import pandas as pd
import io
import gc
import cv2

st.set_option('deprecation.showPyplotGlobalUse', False)


@st.cache(allow_output_mutation=True)
def load_model():
    model = keras.models.load_model('music_classifier_efficientnet_trimmed.h5')
    return model


def main_section():
    st.title('Music Genre Classification')
    st.write('')
    st.write('')
    st.image('https://images.pexels.com/photos/594388/vinyl-record-player-retro-594388.jpeg?auto=compress&cs=tinysrgb&dpr=2&h=650&w=940',
             use_column_width=True)
    st.markdown('This is a simple music genre classification app where one can upload a *wav* audio file and get the predicted genre. '
                'Genres are predicted based on the spectrograms generated from 3s audio clips. Therefore the uploaded audio track has to be also '
                'trimmed to 3 seconds. User has to input an offset in seconds which indicates the starting point of the 3 seconds interval. '
                'Additionally one can also display the spectrogram for the entire audio file.')

    audio_file = st.sidebar.file_uploader('Choose wav file to upload',type=['wav'])
    if audio_file is not None:
        if st.sidebar.button('Display spectrogram'):
            y, sr = librosa.load(audio_file)
            S = librosa.feature.melspectrogram(y=y, sr=sr)
            S_dB = librosa.power_to_db(S, ref=np.max)

            fig, ax = plt.subplots(figsize=(15, 5))
            img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, ax=ax)
            ax.axis('off')
            fig.colorbar(img, ax=ax, format='%+2.0f dB')
            st.pyplot(fig)

        if st.sidebar.checkbox('Prediction'):
            starting_point = st.text_input('Specify the offset')
            if st.sidebar.button('Predict genre'):
                y, sr = librosa.load(audio_file, offset=float(starting_point), duration=3)
                S = librosa.feature.melspectrogram(y=y, sr=sr)
                S_dB = librosa.power_to_db(S, ref=np.max)

                #plot to numpy
                fig, ax = plt.subplots(figsize=(15, 5), dpi=100)
                img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, ax=ax)
                ax.axis('off')
                fig.tight_layout(pad=0)
                io_buf = io.BytesIO()
                fig.savefig(io_buf, format='raw', dpi=100)
                io_buf.seek(0)
                img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                                     newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
                io_buf.close()

                class_mapping = {0: 'blues',
                                 1: 'classical',
                                 2: 'country',
                                 3: 'disco',
                                 4: 'hiphop',
                                 5: 'jazz',
                                 6: 'metal',
                                 7: 'pop',
                                 8: 'reggae',
                                 9: 'rock'}
                model = load_model()
                img_prep = np.expand_dims(cv2.resize(img_arr, (224, 224))[:, :, :3], axis=0)
                y_pred = model.predict(img_prep)
                df = pd.DataFrame(y_pred.flatten(), index=class_mapping.values(), columns=['Probabilities']).sort_values(
                    by='Probabilities', ascending=False)
                st.dataframe(df)

                fig, ax = plt.subplots()
                sns.barplot(data=df, y=df.index, x=df.Probabilities)
                st.pyplot(fig)

main_section()
gc.collect()
