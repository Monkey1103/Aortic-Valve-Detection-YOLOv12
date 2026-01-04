import locale
def getpreferredencoding(do_setlocale = True):
    return "UTF-8"
locale.getpreferredencoding = getpreferredencoding

from ultralytics import YOLO
import os
import shutil

if __name__ == "__main__":
    model = YOLO(r"./yolo12x.yaml")
    results = model.train(data="./aortic_valve_colab.yaml",
                epochs=500,
                batch=6,
                imgsz=640,
                save=True,
                device=0,
                workers=10,
                project='run/yolo12x-neg',
                seed=1,
                verbose=True,
                )
