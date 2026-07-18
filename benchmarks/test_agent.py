import argparse
import json
import time
import csv
import requests
import subprocess
import os
import signal
from pathlib import Path


# -----------------------------
# CONFIG
# -----------------------------

MODELS = [
    "Q4_K_M",
    "Q5_K_M",
    "Q8_0",
    "F16",
]


TASKS = [
    {
        "id": "math",
        "prompt": """
You are an agent.

Calculate:
(83 + 68) * 2

Return only the number.
""",
        "answer": "302"
    },

    {
        "id": "reasoning",
        "prompt": """
A shop has 120 apples.
It sells 30 apples every day.

How many days until all apples are sold?

Return only the number.
""",
        "answer": "4"
    },

    {
        "id": "memory",
        "prompt": """
Remember this information:

Company revenue = 50 million.
Company expenses = 20 million.

What is the profit?

Return only the number.
""",
        "answer": "30"
    }
]


SERVER_PORT = 8080
SERVER_HOST = "127.0.0.1"


# -----------------------------
# GPU MEMORY
# -----------------------------

def gpu_memory():

    try:
        result = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits"
            ]
        )

        return int(result.decode().strip())

    except:
        return None



# -----------------------------
# START SERVER
# -----------------------------

def start_server(model_path):

    cmd = [
        "llama-server",
        "--model",
        model_path,
        "--port",
        str(SERVER_PORT),
        "--n-gpu-layers",
        "999",
        "--ctx-size",
        "8192",
        "--flash-attn",
    ]


    print("Starting:")
    print(" ".join(cmd))


    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )


    # wait until server ready

    for _ in range(60):

        try:
            r = requests.get(
                f"http://{SERVER_HOST}:{SERVER_PORT}"
            )

            if r.status_code == 200:
                break

        except:
            pass

        time.sleep(1)


    return process



def stop_server(process):

    process.send_signal(signal.SIGTERM)

    try:
        process.wait(timeout=10)
    except:
        process.kill()



# -----------------------------
# LLM CALL
# -----------------------------

def ask_llm(prompt):

    payload = {

        "messages":[
            {
                "role":"user",
                "content":prompt
            }
        ],

        "temperature":0,

        "max_tokens":100
    }


    start=time.time()


    response=requests.post(
        f"http://{SERVER_HOST}:{SERVER_PORT}/v1/chat/completions",
        json=payload,
        timeout=300
    )


    latency=time.time()-start


    data=response.json()


    text=data["choices"][0]["message"]["content"]


    return text, latency



# -----------------------------
# RUN TASKS
# -----------------------------

def run_tasks():

    results=[]


    for task in TASKS:

        print("\nTask:",task["id"])


        answer, latency = ask_llm(
            task["prompt"]
        )


        success = task["answer"] in answer


        print(
            "Output:",
            answer
        )


        results.append({

            "task":task["id"],

            "success":success,

            "latency":round(latency,3),

            "answer":answer

        })


    return results



# -----------------------------
# MAIN
# -----------------------------

def main():

    parser=argparse.ArgumentParser()


    parser.add_argument(
        "model_dir",
        help="folder containing GGUF models"
    )


    args=parser.parse_args()



    all_results=[]


    for quant in MODELS:


        print("\n========================")
        print("Testing:",quant)
        print("========================")


        model_file=None


        for f in Path(args.model_dir).glob("*.gguf"):

            if quant.lower() in f.name.lower():

                model_file=f
                break



        if not model_file:

            print(
                "Missing model:",
                quant
            )

            continue



        before=gpu_memory()


        server=start_server(
            str(model_file)
        )


        time.sleep(5)


        after=gpu_memory()



        try:

            results=run_tasks()


        finally:

            stop_server(server)



        success=sum(
            x["success"]
            for x in results
        )


        total=len(results)


        summary={

            "model":quant,

            "file":str(model_file),

            "success_rate":
                round(success/total,3),

            "avg_latency":
                round(
                    sum(
                        x["latency"]
                        for x in results
                    )/total,
                    3
                ),

            "gpu_memory_mb":
                after-before
                if before and after
                else after

        }


        print(summary)


        all_results.append(summary)



    with open(
        "quant_results.csv",
        "w",
        newline=""
    ) as f:


        writer=csv.DictWriter(
            f,
            fieldnames=all_results[0].keys()
        )

        writer.writeheader()

        writer.writerows(all_results)



    print("\nFinished")
    print(
        json.dumps(
            all_results,
            indent=2
        )
    )



if __name__=="__main__":
    main()