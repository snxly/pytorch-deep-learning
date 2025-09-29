import models, predict
import random
from PIL import Image
from pathlib import Path

class_names= ['pizza', 'steak', 'sushi']

model, model_transform = models.load_efficient_b2()

# Get a list of all test image filepaths
test_data_paths = list(Path('data/pizza_steak_sushi_20_percent/test').glob("*/*.jpg"))

# Randomly select a test image path
random_image_path = random.sample(test_data_paths, k=1)[0]

# Open the target image
image = Image.open(random_image_path)
print(f"[INFO] Predicting on image at path: {random_image_path}\n")

# Predict on the target image and print out the outputs
pred, prob = predict.predict(img=image, model=model, model_transform=model_transform)
print(class_names[pred], prob.max().item())
# print(f"Prediction label and probability dictionary: \n{pred_dict}")
# print(f"Prediction time: {pred_time} seconds")