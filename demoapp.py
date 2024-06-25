import gradio as gr
import numpy as np
from random import choice, shuffle
from os import listdir
#ui from https://github.com/facebookresearch/audiocraft/blob/72cb16f9fb239e9cf03f7bd997198c7d7a67a01c/demos/musicgen_app.py
#conda activate ai && python -m demoapp --share
def random_fsong(folderPath):
    return choice(listdir(folderPath))
def random_top_20(folderPath, firstSong, fun, duration):
    #asserts
    import torch
    assert torch.cuda.is_available()
    assert os.path.isdir(folderPath)
    assert firstSong in os.listdir(folderPath)
    assert fun is not None
    assert fun == 'lin' or isinstance(duration, int)
    from models import CUDAModel
    model = CUDAModel()
    distances = []
    firstsongEmb = model.get_latent_decoding(folderPath + firstSong, seconds = duration)
    for (i, song) in enumerate(shuffle(os.listdir(folderPath))):
        if song == firstSong:
            continue
        if i > 20:
            break
        distance, temp1, temp2 = model.get_similarity_file(folderPath + firstSong, folderPath + song, seconds = duration, tens1 = firstsongEmb)
        pcdistance = model.convToPercent(temp1, temp2, distance, fun=fun)
        distances.append((pcdistance, folderPath + song))
        del temp1, temp2

    return [item for pair in distances for item in pair]
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
                    folder_id = gr.Text(label="folder Path", interactive=True, value = "examplemusic")
                    first_song = gr.Text(label = "first Song", interactive= True)
                    simfunc = gr.Radio(['lin', 'sdtw'])
                with gr.Row():
                    _ = gr.Button("Random First song").click(fn=random_fsong, queue=False, inputs=[folder_id], outputs=[first_song])
                    submit = gr.Button("Get random 20 and sort")
                with gr.Row():
                    duration = gr.Slider(minimum=1, maximum=120, value=10, label="Duration", interactive=True)
                with gr.Row():
                    topk = gr.Number(label="Top-k", value=20, interactive=True, maximum=20, minimum= 0, )
                    topp = gr.Number(label="Top-p", value=0, interactive=True)
                    temperature = gr.Number(label="Temperature", value=1.0, interactive=True)
                    cfg_coef = gr.Number(label="Classifier Free Guidance", value=3.0, interactive=True)
            with gr.Column():
                for i in range(topk.maximum):
                    with gr.Row():
                        similarity = gr.Number(label = 'similarity')
                        audios.append(similarity)
                        output = gr.Audio(elem_id = i, visible = True, interactive= False, show_download_button= False, waveform_options={"show_controls" :False})
                        audios.append(output)
        # Define what happens when the "Submit" button is clicked
        submit.click(fn=random_top_20, inputs=[folder_id, first_song,simfunc, duration], outputs=audios)

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