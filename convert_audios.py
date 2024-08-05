import subprocess
import os

for i in os.listdir("../data/wei"):
    # convert to wav, 1 channels, sr 22050
    name = i.split(".")[0]
    subprocess.run(f'ffmpeg -i ../data/wei/{i} -ac 1 -ar 22050 data/wei/{name}.wav -y', shell=True)
