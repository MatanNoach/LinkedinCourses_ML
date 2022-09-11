import PIL.Image
import PIL.ImageDraw
import face_recognition
import os
base_path = 'face_recognition\\datasets\\Ch03'

# Load the jpg file into numpy array
image = face_recognition.load_image_file(os.path.join(base_path, "people.jpg"))

# Find all the features in all faces in the image
face_landmarks_list = face_recognition.face_landmarks(image)

number_of_faces = len(face_landmarks_list)

pil_image = PIL.Image.fromarray(image)

draw = PIL.ImageDraw.Draw(pil_image)

for face_landmarks in face_landmarks_list:
    for name,list_of_points in face_landmarks.items():
        print(f"The {name} of the face has the following points: {list_of_points}")
        draw.line(list_of_points,fill='red',width=2)

pil_image.show()
