import aiohttp
import asyncio
import time
import numpy as np
import pandas as pd
import subprocess
import json
import sys
from jproperties import Properties

np.random.seed(0)

file_paths = [f'Seq_classification_artifacts/sample_text_captum_input.txt']
url = 'http://127.0.0.1:8080/predictions/my_tc'
num_requests = 8000
# arrival_rate = None 
output_file = 'experiment_logs/l4_results.csv'
compute_type = 'mps'
batch_size = 1
num_workers = 8

async def send_post_request(session, url, file_path):
    with open(file_path, 'rb') as file:
        data = file.read()
    start_time = time.time()
    async with session.post(url, data=data) as response:
        end_time = time.time()
        latency = end_time - start_time
        return response.status, latency

async def measure_throughput_and_latency(file_paths, url, num_requests, arrival_rate, output_file):
    total_time = 0
    successful_requests = 0
    latencies = []

    async with aiohttp.ClientSession() as session:
        tasks = []

        start_time = time.time()
        
        for i in range(num_requests):
            task = send_post_request(session, url, file_paths[i % len(file_paths)])
            tasks.append(task)

            if arrival_rate:
                inter_arrival_time = np.random.exponential(1 / arrival_rate)
                await asyncio.sleep(inter_arrival_time)

        responses = await asyncio.gather(*tasks)
        end_time = time.time()

        for response_status, latency in responses:
            if response_status == 200:
                successful_requests += 1
                latencies.append(latency)

        total_time = end_time - start_time

    if successful_requests > 0:


        experiment_type = 'poisson' if arrival_rate else 'constant'
        average_time_per_request = total_time / successful_requests
        throughput = successful_requests / total_time

        average_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        latency_90 = sorted(latencies)[int(0.9 * len(latencies))]
        latency_95 = sorted(latencies)[int(0.95 * len(latencies))]
        latency_99 = sorted(latencies)[int(0.99 * len(latencies))]
        
        # if output file doesn't exist, create one with pandas
        try:
            df = pd.read_csv(output_file)
        except:
            df = pd.DataFrame(columns=['compute_type', 'num_workers', 'batch_size','experiment_type', 'arrival_rate', 'num_requests', 'total_time', 'successful_requests', 'average_time_per_request', 'throughput', 'average_latency', 'min_latency', 'max_latency', 'latency_90', 'latency_95', 'latency_99'])
        
        df = pd.concat([df, pd.DataFrame([[compute_type, num_workers, batch_size, experiment_type, arrival_rate, num_requests, total_time, successful_requests, average_time_per_request, throughput, average_latency, min_latency, max_latency, latency_90, latency_95, latency_99]], columns=df.columns)], ignore_index=True)
        df.to_csv(output_file, index=False)
    else:
        print("No successful requests")


def update_batch_size(batch_size):
    config_file_path = 'config.properties'

    # Read the existing config file
    with open(config_file_path, 'r') as file:
        lines = file.readlines()

    # Modify the batch size
    updated_lines = []
    for line in lines:
        if '"batchSize":' in line:
            # Find the position of the current batch size value
            start_index = line.index('"batchSize":') + len('"batchSize":')
            end_index = line.index(',', start_index)
            # Replace the current batch size value with the new value (16)
            line = line[:start_index] + f" {batch_size}" + line[end_index:]
        updated_lines.append(line)

    # Write the updated config back to the file
    with open(config_file_path, 'w') as file:
        file.writelines(updated_lines)

asyncio.run(measure_throughput_and_latency(file_paths, url, num_requests, None, output_file))

# for bs in [1, 2, 4, 8, 16, 32]:
#     batch_size = bs
#     subprocess.run('bash -c "source activate dev; python -V"', shell=True)
#     subprocess.run(["torchserve", "--stop"])
#     update_batch_size(bs)
#     subprocess.run("export JAVA_HOME=$HOME/.jdk/jdk-11.0.23+9/ && export PATH=$PATH:$JAVA_HOME/bin", shell=True)
#     subprocess.run(["torchserve", "--start", "--model-store", "./model_store", "--models", f"my_tc=BERTSeqClassification{bs}.mar", "--ncs"])
#     time.sleep(10)

#     asyncio.run(measure_throughput_and_latency(file_paths, url, num_requests, None, output_file))
#     asyncio.run(measure_throughput_and_latency(file_paths, url, num_requests, 500, output_file))