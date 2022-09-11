import face_recognition
import os
base_path = 'face_recognition\\datasets\\Ch05'

image = face_recognition.load_image_file(os.path.join(base_path,'person.jpg'))

# Generate the face encodings
face_encodings = face_recognition.face_encodings(image)

if len(face_encodings) ==0:
    print("No faces found in the image")
else:
    first_face_encoding = face_encodings[0]

    print(first_face_encoding)