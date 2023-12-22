from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from tkinter import ttk  # Import themed widgets module
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np


# Load the model with compile=False
model = load_model(r'E:\Alogorithms\my_cnn_model.h5', compile=False)


import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np

class_names = ['apple', 'blueberry', 'kiwi', 'orange', 'pineapple']

def blueberry_rectangle(image_path):
    original_image = cv2.imread(image_path)
    hsv = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
    lower_blueberry = np.array([100, 50, 50])
    upper_blueberry = np.array([130, 255, 255])
    blueberry_mask = cv2.inRange(hsv, lower_blueberry, upper_blueberry)
    
    contours, _ = cv2.findContours(blueberry_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    if contours:
        max_contour = contours[0]
        x, y, w, h = cv2.boundingRect(max_contour)
        cv2.rectangle(original_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        modified_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        return modified_image
 
def orange_rectangle(image_path):
    original_image = cv2.imread(image_path)
    hsv = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
    lower_orange = np.array([10, 100, 20])
    upper_orange = np.array([25, 255, 255])
    orange_mask = cv2.inRange(hsv, lower_orange, upper_orange)
    
    contours, _ = cv2.findContours(orange_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    if contours:
        max_contour = contours[0]
        x, y, w, h = cv2.boundingRect(max_contour)
        cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 165, 255), 2)

        modified_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        return modified_image
    
def kiwi_rectangle(image_path):
    original_image = cv2.imread(image_path)
    hsv = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
    lower_outer_kiwi = np.array([30, 40, 40])
    upper_outer_kiwi = np.array([90, 255, 255])
    outer_kiwi_mask = cv2.inRange(hsv, lower_outer_kiwi, upper_outer_kiwi)
    lower_inner_kiwi = np.array([10, 40, 40])
    upper_inner_kiwi = np.array([30, 255, 255])
    inner_kiwi_mask = cv2.inRange(hsv, lower_inner_kiwi, upper_inner_kiwi)

    kiwi_mask = cv2.bitwise_or(outer_kiwi_mask, inner_kiwi_mask)
    
    contours, _ = cv2.findContours(kiwi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours based on their areas in descending order
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    if contours:
        max_contour = contours[0]
        x, y, w, h = cv2.boundingRect(max_contour)
        cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 165, 255), 2)
        max_contour = contours[0]
        area = cv2.contourArea(max_contour)
        
        if area > 250:  # Adjust the threshold based on your needs
            # Draw a rectangle around the detected kiwi region
            x, y, w, h = cv2.boundingRect(max_contour)
            cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        modified_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        return modified_image    

def apple_rectangle(image_path):
    original_image = cv2.imread(image_path)
    hsv = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
    lower_outer_apple = np.array([30, 40, 40])
    upper_outer_apple = np.array([90, 255, 255])
    outer_apple_mask = cv2.inRange(hsv, lower_outer_apple, upper_outer_apple)

    # Define the range of HSV values for the inner part of kiwi (darker brown)
    lower_inner_apple = np.array([10, 40, 40])
    upper_inner_apple = np.array([30, 255, 255])
    inner_apple_mask = cv2.inRange(hsv, lower_inner_apple, upper_inner_apple)

    # Combine the masks for the outer and inner parts of kiwi
    apple_mask = cv2.bitwise_or(outer_apple_mask, inner_apple_mask)

    # Find contours in the binary image
    contours, _ = cv2.findContours(apple_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours based on their areas in descending order
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    if contours:
        max_contour = contours[0]
        area = cv2.contourArea(max_contour)
        x, y, w, h = cv2.boundingRect(max_contour)
        cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    modified_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    return modified_image

def pineapple_rectangle(image_path):
    original_image = cv2.imread(image_path)

    # Convert the image to the HSV color space
    hsv = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)

    # Define a broad range of HSV values for detecting pineapples
    lower_pineapple = np.array([15, 30, 30])
    upper_pineapple = np.array([90, 255, 255])
    pineapple_mask = cv2.inRange(hsv, lower_pineapple, upper_pineapple)

    # Find contours in the binary image
    contours, _ = cv2.findContours(pineapple_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours based on their areas in descending order
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    if contours:
        max_contour = contours[0]
        area = cv2.contourArea(max_contour)

        # Draw a rectangle around the detected pineapple region
        x, y, w, h = cv2.boundingRect(max_contour)
        cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 255), 2)

        modified_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        return modified_image   
    
def select(file_path, class_name):
    if class_name == 'blueberry':
        return blueberry_rectangle(file_path)
    elif class_name == 'orange':
        return orange_rectangle(file_path)
    elif class_name == 'kiwi':
        return kiwi_rectangle(file_path)
    elif class_name == 'apple':
        return apple_rectangle(file_path)
    elif class_name == 'pineapple':
        return pineapple_rectangle(file_path)

def predict_and_apply_choice(file_path):
    img = Image.open(file_path)
    img = img.resize((201, 163), Image.ANTIALIAS)

    img_for_prediction = image.load_img(file_path, target_size=(201, 163))
    img_array = image.img_to_array(img_for_prediction)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_names[predicted_class_index]
    
    # Update the prediction label text
    predicted_label.config(text=f"Predicted class: {predicted_class_name}")

    modified_img = select(file_path,predicted_class_name)
    if modified_img is not None:
        modified_img = Image.fromarray(modified_img)
        modified_img_tk = ImageTk.PhotoImage(modified_img)
        modified_image_label.img_tk = modified_img_tk
        modified_image_label.config(image=modified_img_tk)
    else:
        print(f"No {predicted_class_name}s detected in the image.")
        
def predict_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        predict_and_apply_choice(file_path)

root = tk.Tk()
root.title("Image Classifier")
root.geometry("600x500")
root.configure(bg='gray')

# Create a label to display the original image
image_label = tk.Label(root, bg='gray')
image_label.place(relx=0.5, rely=0.4, anchor='center')

# Create a label to display the modified image with blueberry rectangle
modified_image_label = tk.Label(root, bg='gray')
modified_image_label.place(relx=0.5, rely=0.6, anchor='center')

predicted_label = tk.Label(root, text="", bg='gray', font=("Arial", 14))
predicted_label.place(relx=0.5, rely=0.85, anchor='center')

# Create a button to trigger image selection and prediction
upload_button = ttk.Button(root, text="Choose Image", command=predict_image)
upload_button.place(relx=0.5, rely=0.2, anchor='center')

root.mainloop()
