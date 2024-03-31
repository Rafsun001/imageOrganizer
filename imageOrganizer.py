import tkinter as tk
from tkinter import filedialog
import cv2
import face_recognition
import shutil
import os
import numpy as np

# Initialize folder paths and image path
folder_path_1 = ""
folder_path_2 = ""
image_path = ""

# Function to set folder path 1
def select_folder_path_1():
    global folder_path_1
    folder_path_1 = filedialog.askdirectory(title="Select Folder Path 1")
    folder_path_entry_1.config(text=folder_path_1)

# Function to set folder path 2
def select_folder_path_2():
    global folder_path_2
    folder_path_2 = filedialog.askdirectory(title="Select Folder Path 2")
    folder_path_entry_2.config(text=folder_path_2)

# Function to set image path
def select_image():
    global image_path
    image_path = filedialog.askopenfilename(title="Select Image File", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    image_path_entry.config(text=image_path)

# Your backend function
# Backend function for image processing
def backend_function():
    global folder_path_1, folder_path_2, image_path

    # Ensure paths are not empty
    if not folder_path_1 or not folder_path_2 or not image_path:
        result_label.config(text="Invalid paths")
        return

    # Read the image
    img = cv2.imread(image_path)

    #### Get the target image ####
    # Function to display the image and allow user to draw a square
    def draw_square(image):
        # Create a window to display the image
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        # Set window size
        cv2.resizeWindow("Image", 550, 550)
        # Display the image
        cv2.imshow("Image", image)

        # Initialize variables for drawing the square
        rectangle = [(0, 0), (0, 0)]
        drawing = False

        # Mouse callback function
        def mouse_callback(event, x, y, flags, param):
            nonlocal rectangle, drawing

            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                rectangle[0] = (x, y)
            elif event == cv2.EVENT_MOUSEMOVE:
                if drawing:
                    rectangle[1] = (x, y)
            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False
                rectangle[1] = (x, y)

        # Set the mouse callback function
        cv2.setMouseCallback("Image", mouse_callback)

        # Wait for the user to draw the square
        while True:
            # Create a copy of the original image
            img_copy = image.copy()

            # Draw the square on the copy of the image
            cv2.rectangle(img_copy, rectangle[0], rectangle[1], (0, 255, 0), 10)

            # Display the image with the square
            cv2.imshow("Image", img_copy)

            # Wait for a key press
            key = cv2.waitKey(1) & 0xFF

            # If the 'q' key is pressed, break from the loop
            if key == ord("q"):
                break

        # Close all OpenCV windows
        cv2.destroyAllWindows()

        # Extract the part of the image inside the square
        x1, y1 = rectangle[0]
        x2, y2 = rectangle[1]
        cropped_image = image[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)]

        cropped_image_resized = cv2.resize(cropped_image, (0, 0), None, fx=0.25, fy=0.25)
        # Return the cropped image
        return cropped_image_resized

    # Check if the image is read successfully
    if img is None:
        result_label.config(text="Error: Could not read the image.")
    else:
        # Call the function to allow the user to draw a square
        cropped_image = draw_square(img)

        # Display the cropped image
        cv2.imshow("Cropped Image", cropped_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    ############ Face ##############
    unknown_faces_folder = os.path.join(folder_path_2, "unknown_faces")
    os.makedirs(unknown_faces_folder, exist_ok=True)

    ## Get target image face encodings
    img_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
    img_loc = face_recognition.face_locations(img_rgb)
    target_img_encode = face_recognition.face_encodings(img_rgb, img_loc)

    image_path_list = os.listdir(folder_path_1)
    img_list = []
    for path in image_path_list:
        img_list.append(cv2.imread(os.path.join(folder_path_1, path)))

    for imgg, path in zip(img_list, image_path_list):
        if imgg is None:
            print(f"Error: Unable to read image '{path}'.")
            continue

        if imgg.size == 0:
            print(f"Error: Image '{path}' has empty dimensions.")
            continue

        img2 = cv2.resize(imgg, (0, 0), None, fx=0.25, fy=0.25)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        img_loc2 = face_recognition.face_locations(img2)
        img_encode2 = face_recognition.face_encodings(img2, img_loc2)

        for encode_face, face_loc in zip(target_img_encode, img_loc):
            matches = face_recognition.compare_faces(img_encode2, encode_face)
            face_dis = face_recognition.face_distance(img_encode2, encode_face)
            if len(face_dis) == 0:
                src_path = os.path.join(folder_path_1, path)
                if os.path.exists(src_path):
                    shutil.copy(src_path, unknown_faces_folder)
                else:
                    print(f"File '{src_path}' does not exist.")
                continue

            match_index = np.argmin(face_dis)
            if matches[match_index]:
                src_path = os.path.join(folder_path_1, path)
                if os.path.exists(src_path):
                    shutil.copy(src_path, folder_path_2)
            else:
                print('No known face detected')
                src_path = os.path.join(folder_path_1, path)
                if os.path.exists(src_path):
                    shutil.copy(src_path, unknown_faces_folder)
                else:
                    print(f"File '{src_path}' does not exist.")

    result_label.config(text="Processing complete")


# Function to submit the form
def submit_form():
    backend_function()

# Create main window
root = tk.Tk()
root.title("Image Processing App")

# Create and place widgets for selecting folder paths
folder_label_1 = tk.Label(root, text="Folder Path 1:")
folder_label_1.grid(row=0, column=0, padx=5, pady=5, sticky="e")

folder_path_entry_1 = tk.Label(root, text="")
folder_path_entry_1.grid(row=0, column=1, padx=5, pady=5)

select_folder_button_1 = tk.Button(root, text="Select Folder 1", command=select_folder_path_1)
select_folder_button_1.grid(row=0, column=2, padx=5, pady=5)

folder_label_2 = tk.Label(root, text="Folder Path 2:")
folder_label_2.grid(row=1, column=0, padx=5, pady=5, sticky="e")

folder_path_entry_2 = tk.Label(root, text="")
folder_path_entry_2.grid(row=1, column=1, padx=5, pady=5)

select_folder_button_2 = tk.Button(root, text="Select Folder 2", command=select_folder_path_2)
select_folder_button_2.grid(row=1, column=2, padx=5, pady=5)

# Create and place widgets for selecting an image file
image_label = tk.Label(root, text="Image File:")
image_label.grid(row=2, column=0, padx=5, pady=5, sticky="e")

image_path_entry = tk.Label(root, text="")
image_path_entry.grid(row=2, column=1, padx=5, pady=5)

select_button = tk.Button(root, text="Select Image", command=select_image)
select_button.grid(row=2, column=2, padx=5, pady=5)

# Button to submit the form
submit_button = tk.Button(root, text="Submit", command=submit_form)
submit_button.grid(row=3, column=1, padx=5, pady=5)

root.mainloop()
