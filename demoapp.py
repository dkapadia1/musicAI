import gradio as gr
import numpy as np
from random import choice
from os import listdir
#ui from https://github.com/facebookresearch/audiocraft/blob/72cb16f9fb239e9cf03f7bd997198c7d7a67a01c/demos/musicgen_app.py
INTERRUPTING = False
#conda activate ai && python -m demoapp --share
def run_sampler(folder_id, duration, topk, topp, temperature, cfg_coef, max):
    # Your code to run the audio sampler with the given parameters
    # For demonstration, we'll return two random audio signals
    sample_rate = 22050  # Sample rate in Hz
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    signal1 = 0.5 * np.sin(2 * np.pi * 220 * t)  # A 220 Hz sine wave
    signal2 = 0.5 * np.sin(2 * np.pi * 440 * t)  # A 440 Hz sine wave
    signals = []
    for i in range(topk):
        if i % 2 == 0:
            signals.append((sample_rate, signal1))
        else:
            signals.append((sample_rate, signal2))
    while len(signals) <= 20:
        signals.append(sample_rate, t)
    return signals
def random_fsong(folderID):
    return choice(listdir(folderID))
def ui_full(launch_kwargs):
    with gr.Blocks() as interface:
        audios = []
        gr.Markdown(
            """
            # MusicClassifier
            This is your private demo for MusicClassifier,
            a simple and controllable model for music similarities
            presented at: "Simple and Controllable Music Generation"
            """
        )
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    folder_id = gr.Text(label="folder Id", interactive=True, value = "examplemusic")
                    first_song = gr.Text(label = "first Song", interactive= True)
                with gr.Row():
                    submit = gr.Button("Process Folder")
                    def interrupt():
                        global INTERRUPTING
                        INTERRUPTING = True
                    _ = gr.Button("Get top 20").click(fn=interrupt, queue=False)
                    _ = gr.Button("Random First song").click(fn=random_fsong, queue=False, inputs=[folder_id], outputs=[first_song])
                with gr.Row():
                    duration = gr.Slider(minimum=1, maximum=120, value=10, label="Duration", interactive=True)
                with gr.Row():
                    topk = gr.Number(label="Top-k", value=20, interactive=True, maximum=50, minimum= 0, )
                    topp = gr.Number(label="Top-p", value=0, interactive=True)
                    temperature = gr.Number(label="Temperature", value=1.0, interactive=True)
                    cfg_coef = gr.Number(label="Classifier Free Guidance", value=3.0, interactive=True)
            with gr.Column():
                for i in range(topk.maximum):
                    output = gr.Audio(elem_id = i, visible = True, interactive= False, show_download_button= False, waveform_options={"show_controls" :False})
                    audios.append(output)
        # Define what happens when the "Submit" button is clicked
        submit.click(fn=run_sampler, inputs=[folder_id, duration, topk, topp, temperature, cfg_coef], outputs=audios)

        interface.queue().launch(**launch_kwargs)
import argparse
import logging
import sys
import os
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--listen',
        type=str,
        default='0.0.0.0' if 'SPACE_ID' in os.environ else '127.0.0.1',
        help='IP to listen on for connections to Gradio',
    )
    parser.add_argument(
        '--username', type=str, default='', help='Username for authentication'
    )
    parser.add_argument(
        '--password', type=str, default='', help='Password for authentication'
    )
    parser.add_argument(
        '--server_port',
        type=int,
        default=0,
        help='Port to run the server listener on',
    )
    parser.add_argument(
        '--inbrowser', action='store_true', help='Open in browser'
    )
    parser.add_argument(
        '--share', action='store_true', help='Share the gradio UI'
    )

    args = parser.parse_args()

    launch_kwargs = {}
    launch_kwargs['server_name'] = args.listen

    if args.username and args.password:
        launch_kwargs['auth'] = (args.username, args.password)
    if args.server_port:
        launch_kwargs['server_port'] = args.server_port
    if args.inbrowser:
        launch_kwargs['inbrowser'] = args.inbrowser
    if args.share:
        launch_kwargs['share'] = args.share

    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    ui_full(launch_kwargs)