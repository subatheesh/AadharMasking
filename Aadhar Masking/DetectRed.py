import io
import os
#
# # Imports the Google Cloud client library
# from google.cloud import vision
# from google.cloud.vision import types
#
# # Instantiates a client
# client = vision.ImageAnnotatorClient()
#
# # The name of the image file to annotate
# # file_name = os.path.join(
# #     os.path.dirname(__file__),
# #     'resources/wakeupcat.jpg')
#
# file_name = "p.png"
#
# # Loads the image into memory
# with io.open(file_name, 'rb') as image_file:
#     content = image_file.read()
#
# image = types.Image(content=content)
#
# # Performs label detection on the image file
# response = client.document_text_detection(image=image)
# print(response)
# labels = response.label_annotations
#
# print('Labels:')
# for label in labels:
#     print(label.description)

from google.cloud import vision
client = vision.ImageAnnotatorClient()

file_name = "Test/test.jpg"

with io.open(file_name, 'rb') as image_file:
    content = image_file.read()

image = vision.types.Image(content=content)

response = client.text_detection(image=image)
print(response)
texts = response.text_annotations
print('Texts:')

for text in texts:
    print('\n"{}"'.format(text.description))

    vertices = (['({},{})'.format(vertex.x, vertex.y)
                for vertex in text.bounding_poly.vertices])

    print('bounds: {}'.format(','.join(vertices)))
