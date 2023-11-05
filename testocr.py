import easyocr

IMAGE_PATH = 'tests/ocr/6.png'
reader = easyocr.Reader(['en'])
result = reader.readtext(IMAGE_PATH)
for detection in result:
    if detection[2] > 0.5:
        print(detection[1])
    print(detection)