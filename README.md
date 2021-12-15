# AI Based Smart Exam Proctoring Package

It takes image (base64) as input:
Provide Output as: 
1) Detection of Mobile phone.
2) Detection of More than 1 person in the exam.
3) Gaze Estimation: Estimating the position of student body & eyes movements.

# DOWNLOAD LINK OF YOLO V3 MODEL:
https://pjreddie.com/media/files/yolov3.weights

# DOWNLOAD LINK OF shape_predictor_68_face_landmarks.dat MODEL:
https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat?raw=true
# Code Sample Working
```python
from proctoring.proctoring import get_analysis, yolov3_model_v3_path

# insert the path of yolov3 model [mandatory]
yolov3_model_v3_path("yolov3.weights_model_path")

# insert the image of base64 format
imgData = "base64_image_format"
proctorData = get_analysis(imgData, "shape_predictor_68_face_landmarks.dat_model_path")
print(proctorData)

```
# Code Sample Output

```
{'mob_status': 'Not Mobile Phone detected', 'person_status': 'Normal', 'user_move1': 'Head up', 'user_move2': 'Head right', 'eye_movements': 'Blinking'}
```