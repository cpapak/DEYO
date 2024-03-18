import os
import torch
import sys
import numpy as np

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from ultralytics import RTDETR


# Eval the model
model = RTDETR("../pretrained/deyo/deyo-x.pt")
# model.val(data = "coco.yaml")  # for DEYO-tiny: 37.6 AP

device = torch.device("cuda")
model.to(device)
dummy_input = torch.randn(1, 3,640,640, dtype=torch.float).to(device)

# INIT LOGGERS
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
repetitions = 300
timings=np.zeros((repetitions,1))
#GPU-WARM-UP
for _ in range(10):
    _ = model(dummy_input)
# MEASURE PERFORMANCE
with torch.no_grad():
    for rep in range(repetitions):
        starter.record()
        _ = model(dummy_input)
        ender.record()
        # WAIT FOR GPU SYNC
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        timings[rep] = curr_time

mean_syn = np.sum(timings) / repetitions
std_syn = np.std(timings)
print(1000.0/mean_syn)