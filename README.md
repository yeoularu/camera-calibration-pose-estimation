# camera-calibration-pose-estimation
**Camera Calibration with chessboard video and simple pose estimation using opencv-python**


![sample](./sample.gif)

<br />

original video: data/chessboard.mov

---

I calibrate my MacBook Air M2 webcam with OpenCV, the result with 45 images is 
```
K = [[1.43624645e+03 0.00000000e+00 8.21399444e+02]
     [0.00000000e+00 1.43814864e+03 5.27026483e+02]
     [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
dist_coeff = [ 4.60598063e-02  1.99846779e-01 -9.12081076e-04  6.43551674e-04  -8.00178409e-01]
```

The chessboard image(data/chessboard.jpeg) has 11x7 squares(10x6 verticies). It fit well with iPad mini 6th size (175x115mm, 15mm square)

after that, I floated a symbol similar to a butterfly on a chessboard using solvePnP of OpenCV
