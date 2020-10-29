import  eval_ckpt_main as eval_ckpt
import tensorflow.compat.v1 as tf
import sys
sys.path.append('/content/tpu/models/official/efficientnet')
sys.path.append('/content/tpu/models/common')
labels_map_file = "/home/kathy/models/noisystudent/ckpt/teacher_ckpt/labels_map.json"
model_name = "efficientnet-b0"
ckpt_dir = "/home/kathy/models/noisystudent/ckpt/teacher_ckpt/efficientnet-b0"

image_files = ["/home/kathy/mnt/data/frames_small_format/frames/Clean_Animals2/011.jpg"]
eval_driver = eval_ckpt.get_eval_driver(model_name)
pred_idx, pred_prob = eval_driver.eval_example_images(
    ckpt_dir, image_files, labels_map_file)
