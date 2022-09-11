from unittest import result
import face_recognition
import os
base_path = 'face_recognition\\datasets\\Ch06'

# Load the known images

image_1 = face_recognition.load_image_file(os.path.join(base_path,'person_1.jpg'))
image_2 = face_recognition.load_image_file(os.path.join(base_path,'person_2.jpg'))
image_3 = face_recognition.load_image_file(os.path.join(base_path,'person_3.jpg'))

# Get the face encodings for each person
# The [0] is because we know we only have 1 person in every image
# If we have more, we can't do that
encodings_1 = face_recognition.face_encodings(image_1)[0]
encodings_2 = face_recognition.face_encodings(image_2)[0]
encodings_3 = face_recognition.face_encodings(image_3)[0]

known_face_encodings = [encodings_1,encodings_2,encodings_3]

# Encode an unknown image
unknown_image = face_recognition.load_image_file(os.path.join(base_path,'unknown_2.jpg'))
unknown_face_encodings = face_recognition.face_encodings(unknown_image)

# For each encoding found in the unknown image (maybe there are multiple faces - unknown_2 contains all 3 people)
for unknown_face_encoding in unknown_face_encodings:
    # Compare the unknown face to our list of faces
    results = face_recognition.compare_faces(known_face_encodings,unknown_face_encoding)
    
    # Print who did you find
    name="unknown"
    if results[0]:
        name = "person 1"
    elif results[1]:
        name = "person 2"
    elif results[2]:
        name = "person 3"
    print(name)

# Tuning The System
# This image is a small and problematic image so the system won't work
bad_unknown_image = face_recognition.load_image_file(os.path.join(base_path,'unknown_7.jpg'))
# Instead of usin this line:
# bad_unknown_face_encodings = face_recognition.face_encodings(bad_unknown_image)
# we will split it to 2 steps so we'll have more control over the process

# Locate and upscale the image
bad_face_locations = face_recognition.face_locations(bad_unknown_image,number_of_times_to_upsample=2)
# Tell the system where to look for the image to encode better
bad_unknown_face_encodings = face_recognition.face_encodings(bad_unknown_image,known_face_locations=bad_face_locations)

# For each encoding found in the unknown image (maybe there are multiple faces - unknown_2 contains all 3 people)
for unknown_face_encoding in bad_unknown_face_encodings:
    # Compare the unknown face to our list of faces
    results = face_recognition.compare_faces(known_face_encodings,unknown_face_encoding)
    
    # Print who did you find
    name="unknown"
    if results[0]:
        name = "person 1"
    elif results[1]:
        name = "person 2"
    elif results[2]:
        name = "person 3"
    print(name)




