# -*- coding: utf-8 -*-
#import torch
import gradio as gr

#emotion model
from emotion_classify import emotion_model
emo = emotion_model()

def synthesize(audio):
    print('audio.name: ', audio.name)
    output=emo.predict(audio.name)
    
    return output

examples = [
 ["0005.wav"],
 ["0013.wav"],
 ["0023.wav"],
]

title = "Automatic Speech Emotion Classification"
input = gr.inputs.Audio(source="upload", type="file")  #real-time: source="microphone"
output=gr.outputs.Label(label='Output')
gr.Interface(synthesize,  input, output,
    title=title, examples=examples).launch(debug=True, share=True) 