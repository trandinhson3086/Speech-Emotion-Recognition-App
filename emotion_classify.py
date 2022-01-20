import pickle
import soundfile
import numpy as np
import librosa
import warnings
import os
warnings.filterwarnings('ignore')

def extract_feature(file_name, **kwargs):
    """
    Extract feature from audio file `file_name`
        Features supported:
            - MFCC (mfcc)
            - Chroma (chroma)
            - MEL Spectrogram Frequency (mel)
            - Contrast (contrast)
            - Tonnetz (tonnetz)
        e.g:
        `features = extract_feature(path, mel=True, mfcc=True)`
    """
    if not file_name.endswith(".wav"):
        target_path = file_name.split('.')[0]+'.wav'
        if os.path.exists(target_path):
            os.remove(target_path)

        os.system(f"ffmpeg -i {file_name} -ac 1 -ar 16000 {target_path}")
        file_name = target_path

    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")
    
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma or contrast:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result = np.hstack((result, mel))
        if contrast:
            contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
            result = np.hstack((result, contrast))
        if tonnetz:
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
            result = np.hstack((result, tonnetz))

    return result

class emotion_model(object):
    def __init__(self):
        self.model = pickle.load(open("mlp_classifier.model", "rb"))

    def predict(self, filename):
        # extract features and reshape it
        features = extract_feature(filename, mfcc=True, chroma=True, mel=True).reshape(1, -1)
        # predict
        result = self.model.predict(features)[0]
        # show the result !
        print(filename, result[0])
        return result

# em=emotion_model()
##predict file
# print(em.predict('0005.wav'))
# print(em.predict('0013.wav'))
# print(em.predict('0026.wav'))

##predict folder
# folder='test_audio/unlabel/' #'./data/Actor_01/'
# for i in os.listdir(folder):
#     em.predict(folder+i)