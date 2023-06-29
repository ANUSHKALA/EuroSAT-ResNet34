import gradio as gr
import os
from functions import *

examples_dir = 'examples'
title = "Land Use Classification - ResNet34 PyTorch"
examples = [os.path.join(examples_dir, i) for i in os.listdir('examples')]

interface = gr.Interface(fn=predict, inputs=gr.Image(type= 'numpy', shape=(64, 64)).style(height= 256),
            outputs= gr.Label(num_top_classes= 5), cache_examples= False,
            examples= examples, title= title)

interface.launch()