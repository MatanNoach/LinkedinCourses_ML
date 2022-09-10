import PIL.Image
import PIL.ImageDraw
# See requirements.txt file if you are having trouble installing dlib dependency
import face_recognition
import os
base_path = 'face_recognition\\datasets\\Ch03'

# Load the jpg file into an array
image = face_recognition.load_image_file(os.path.join(base_path, "people.jpg"))

# Find all faces locations in the image
# Each location is marked with 4 indexes - top,right,bottom,left
face_locations = face_recognition.face_locations(image)

# Check the number of faces found in the image
num_of_faces = len(face_locations)
print(f"{num_of_faces} faces found in thie photograph")

# Convert to PIL image to perform PIL operations
pil_image = PIL.Image.fromarray(image)


for face_location in face_locations:
    # Print the 4 indexes for each face
    top,right,bottom,left = face_location
    print(f"A face is located at pixel location: Top - {top}, Bottom - {bottom}, Left - {left}, Right - {right}")

    # Draw a rectangle above each image by their indexes
    draw = PIL.ImageDraw.Draw(pil_image)
    draw.rectangle([left,top,right,bottom],outline="red")
# Show the image with the rectangles
pil_image.show()
