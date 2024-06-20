from modal import Image, App, Volume
import sys
import time

image = (
    Image.debian_slim()
    .apt_install(
        "git",
        "default-jre",
        "curl",
    )
    .pip_install(
        'transformers',
        'torch-model-archiver',
        'torchserve',
        'optimum',
        'nvgpu',
        'captum',
        'torchtext'
    )
    .run_commands(
        "git clone https://github.com/cathyzbn/mps.git",
        "cd mps && pip install -r requirements.txt",
        "cd mps && python Download_Transformer_models.py",
        # "cd mps && torch-model-archiver --model-name BERTSeqClassification --version 1.0 --serialized-file ./Transformer_model/model.safetensors --handler ./Transformer_handler_generalized.py --extra-files \"./Transformer_model/config.json,./setup_config.json,./Seq_classification_artifacts/index_to_name.json\"",
        # "cd mps && mkdir /model_store",
        # "cd mps && mv BERTSeqClassification.mar /model_store",
    )
   
    .copy_local_dir(".", ".")
    .dockerfile_commands([
        # "FROM ubuntu:14.04",
        # 'RUN echo $\'#!/usr/bin/env sh\necho "hi"\nexec "$@"\' > /temp.sh',
        "RUN chmod +x /entrypoint.sh",
        "ENTRYPOINT [\"/entrypoint.sh\"]",
    ])

)

app = App(f"test", image=image)

@app.function(gpu="A10G")
def start_mps():
    
    import subprocess
    subprocess.run(["nvidia-smi"])
    subprocess.run(["su",  "-"])
    subprocess.run(["apt-get", "install", "sudo", "-y"])
    subprocess.run(["sudo", "nvidia-smi", "-c", "3"])
    # subprocess.run(["nvidia-cuda-mps-control", "-d"])
    # print("MPS started")
    # subprocess.run(["ls"])

    # # full command: torch-model-archiver --model-name BERTSeqClassification --version 1.0 --serialized-file ./Transformer_model/model.safetensors --handler ./Transformer_handler_generalized.py --extra-files "./Transformer_model/config.json,./setup_config.json,./Seq_classification_artifacts/index_to_name.json"
    # subprocess.run(["torch-model-archiver", "--model-name", "BERTSeqClassification", "--version", "1.0", "--serialized-file", "./Transformer_model/model.safetensors", "--handler", "./Transformer_handler_generalized.py", "--extra-files", "\"./Transformer_model/config.json,./setup_config.json,./Seq_classification_artifacts/index_to_name.json\""])
    # subprocess.run(["mkdir", "/model_store"])
    # subprocess.run(["mv", "BERTSeqClassification.mar", "/model_store"])

    # # full command: torchserve --start --model-store ./model_store --ts-config config.properties --models my_tc=BERTSeqClassification.mar --ncs
    # subprocess.run(["torchserve", "--start", "--model-store", "./model_store", "--models", "my_tc=BERTSeqClassification.mar", "--ncs"])
    # print("Torchserve started")

    # f = open("sample_text_captum_input.txt", "w")
    # f.write('{"text":"Bloomberg has decided to publish a new report on the global economy.", "target":1}')
    # # full command: curl -X POST http://127.0.0.1:8080/predictions/my_tc -T sample_text_captum_input.txt
    # subprocess.run(["curl", "-X", "POST", "http://127.0.0.1:8080/predictions/my_tc", "-T", "sample_text_captum_input.txt"])


@app.function(gpu="T4")
def stop_mps():
    import subprocess

    subprocess.run(["echo", 'quit', "|", "nvidia-cuda-mps-control"])

    print("MPS stopped")
 

# @app.function(gpu="T4")
# def submit_query():
#     import subprocess
    
@app.local_entrypoint()
def main():
    print(start_mps.remote())
    time.sleep(10)

    # time.sleep(10)
    
    # print(submit_query.remote())

    # print(stop_mps.remote())
