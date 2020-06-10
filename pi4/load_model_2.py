from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import io
import time
import numpy as np
import picamera

from PIL import Image
from tflite_runtime.interpreter import Interpreter

def set_input_tensor(interpreter, image):
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image


def classify_image(interpreter, image, top_k=1):
  """Returns a sorted array of classification results."""
  set_input_tensor(interpreter, image)
  interpreter.invoke()
  output_details = interpreter.get_output_details()[0]
  output = np.squeeze(interpreter.get_tensor(output_details['index']))

  # If the model is quantized (uint8 data), then dequantize the results
  if output_details['dtype'] == np.uint8:
    scale, zero_point = output_details['quantization']
    output = scale * (output - zero_point)

  ordered = np.argpartition(-output, top_k)
  return [(i, output[i]) for i in ordered[:top_k]]

def main():
    #path_model = "HUGONETWORK_224_224_25_0.00001_16_13000.tflite"
    path_model = "HUGONETWORK_224_224_50_0.0001_16_14000.tflite"
    image_path = "TEST.jpg"
    print("[!] Loading {} ..".format(str(path_model)))
    interpreter = Interpreter(path_model)
    print("[*] Done")
    interpreter.allocate_tensors()
    _, height, width, _ = interpreter.get_input_details()[0]['shape']
    image = Image.open(image_path).resize((width, height))

    start_time = time.time()
    results = classify_image(interpreter, image)
    elapsed_ms = (time.time() - start_time) * 1000
    label_id, prob = results[0]
    print("[*] Label id {} Proba {} Elapsed time {} ms".format(label_id, prob, elapsed_ms))

main()
