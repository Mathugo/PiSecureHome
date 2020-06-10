from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tflite_runtime.interpreter as tflite

import numpy as np
from PIL import Image
path="model.tflite"
import time

class LiteModel:
    def __init__(self, path):
        print("[!] Loading {} ..".format(str(path)))
        self.interpreter = tflite.Interpreter(path)
        print("[*] Done")
        self.interpreter.allocate_tensors()
        _, self.height, self.width, _ = self.interpreter.get_input_details()[0]['shape']
        # Get input and output tensors.
        #self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        print("[*] Image width {} height {}".format(self.width, self.height))
        
    def load_image(self, image_path):
        self.img = Image.open(image_path).resize((self.width, self.height))

    def set_image(self, image):
        self.img = image
        
    def set_input_tensor(self):
        self.tensor_index = self.interpreter.get_input_details()[0]['index']
        self.input_tensor = self.interpreter.tensor(self.tensor_index)()[0]
        self.input_tensor[:, :] = self.img

    def classify_image(self, top_k=1):
        self.set_input_tensor()
        self.interpreter.invoke()
        output_details = self.interpreter.get_output_details()[0]
        output = np.squeeze(self.interpreter.get_tensor(output_details['index']))

        # If the model is quantized (uint8 data), then dequantize the results
        if output_details['dtype'] == np.uint8:
            scale, zero_point = output_details['quantization']
            output = scale * (output - zero_point)

        ordered = np.argpartition(-output, top_k)
        return [(i, output[i]) for i in ordered[:top_k]]

    def run(self):
        start_time = time.time()
        results = self.classify_image()
        elapsed_ms = (time.time() - start_time) * 1000
        label_id, prob = results[0]
        print("[*] Label id {} Proba {} Elapsed time {}".format(label_id, prob, elapsed_ms))

#lite = Lite_Model(path)
#lite.load_image("TEST.jpg")
#lite.run()




