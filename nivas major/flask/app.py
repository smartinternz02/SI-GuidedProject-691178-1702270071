from flask import Flask, render_template, request
import os
from tensorflow.keras.models import load_model
import numpy as np
from keras.preprocessing import image

# Initialize Flask app
app = Flask(__name__)

# Load the saved model
model = load_model("seanaimal.h5")

# Define a function to preprocess the input image(s) if needed
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(351, 351))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to match batch size
    return img_array

# Define route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Define route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the uploaded image file
        uploaded_file = request.files['file']
        # Save the uploaded image file
        image_path = os.path.join('static/uploads', uploaded_file.filename)
        uploaded_file.save(image_path)
        # Preprocess the input image
        processed_image = preprocess_image(image_path)
        # Make predictions
        predictions = model.predict(processed_image)
        # Get the class labels
        class_names = [
            'Penguin', 'Clams', 'Otter', 'Eel', 'Corals', 'Puffers', 'Squid', 'Whale',
            'Sea Urchins', 'Crabs', 'Starfish', 'Seal', 'Octopus', 'Shrimp', 'Sharks',
            'Sea Rays', 'Fish', 'Seahorse', 'Nudibranchs', 'Dolphin', 'Turtle_Tortoise',
            'Jelly Fish'
        ]
        # Get the predicted class index
        predicted_class_index = np.argmax(predictions)
        # Get the predicted class label
        predicted_class_label = class_names[predicted_class_index]
        # Delete the uploaded image file
        # Pass the predicted class label to the HTML template
        return render_template('index.html',image_path=image_path,prediction=predicted_class_label)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
