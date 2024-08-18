import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model


def makeInference(inputImage):
    # Loading the custom model
    classifier = load_model("./model/cat_dog_classifier.keras")
    # Load and preprocess the new image
    img_path = inputImage
    new_image = image.load_img(img_path, target_size=(28, 28))  # Same size as the training images
    new_image = image.img_to_array(new_image)
    new_image = np.expand_dims(new_image, axis=0)  # Add batch dimension

    # Predict the class
    prediction = classifier.predict(new_image)
    print(prediction)
    # Interpreting the result
    if prediction[0][0] > 0.5:
        print("The image is classified as Class 1 (e.g., dog).")
        return "The image belongs to a Dog 🐶🐶🐶"
    else:
        print("The image is classified as Class 0 (e.g., cat).")
        return "The image belongs to a Cat 😸😸😸😸"

# makeInference('E:/datasets/catanddog/1655430860853.jpeg')