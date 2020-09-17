# keras2ncnn

## This project is still WIP. 

### If you want me to support a model, you can create an issue and attach the keras file. 
Currently there is no plan for detection or model complex model, but welcome for PR.

---

## Supported Op
- InputLayer
- Conv2D
- DepthwiseConv2D
- Add
- ZeroPadding2D
- ReLU (No leaky clip relu support)
- GlobalAveragePooling2D
- MaxAveragePooling2D
- AveragePooling2D
- MaxPooling2D
- BatchNormalization
- Dense (With or withour softmax activation)

## Ops that dont work but have done coding


## Ops that have done coding but not checked yet


## Current status
- [X] Be able to convert MobilenetV2, load and excute
- [X] Get all current ops working
- [ ] Support more ops

## A sample conversion and compare result from the keras2ncnn.py -> Model: MobilenetV2-1.0-224.h5
```
==================================
Layer Name: Conv1, Layer Shape: keras->(1, 112, 112, 32) ncnn->(32, 112, 112)
Max:    keras->497.192 ncnn->529.577    Min: keras->-591.230 ncnn->-590.439
Mean:   keras->-15.327 ncnn->-15.309    Var: keras->97.025 ncnn->97.257
Cosine Similarity: 0.00164
Keras Feature Map:      [26.017 21.204 12.724 11.609  9.72  10.486  5.091  6.731  5.089  9.807]
Ncnn Feature Map:       [31.604 22.974  8.5    9.72  17.139 10.659  5.642  0.998  4.327  8.196]
==================================
Layer Name: bn_Conv1, Layer Shape: keras->(1, 112, 112, 32) ncnn->(32, 112, 112)
Max:    keras->2434.938 ncnn->2593.356  Min: keras->-2135.899 ncnn->-2130.922
Mean:   keras->-44.069 ncnn->-44.011    Var: keras->294.513 ncnn->296.122
Cosine Similarity: 0.00298
Keras Feature Map:      [48.01  39.557 24.666 22.707 19.39  20.735 11.26  14.14  11.257 19.543]
Ncnn Feature Map:       [57.821 42.666 17.248 19.389 32.419 21.04  12.228  4.073  9.92  16.713]
==================================
Layer Name: Conv1_relu, Layer Shape: keras->(1, 112, 112, 32) ncnn->(32, 112, 112)
Max:    keras->6.000 ncnn->6.000        Min: keras->0.000 ncnn->0.000
Mean:   keras->2.260 ncnn->2.269        Var: keras->2.845 ncnn->2.850
Cosine Similarity: 0.02350
Keras Feature Map:      [6. 6. 6. 6. 6. 6. 6. 6. 6. 6.]
Ncnn Feature Map:       [6.    6.    6.    6.    6.    6.    6.    4.073 6.    6.   ]
==================================
Layer Name: expanded_conv_depthwise, Layer Shape: keras->(1, 112, 112, 32) ncnn->(32, 112, 112)
Max:    keras->280.475 ncnn->278.144    Min: keras->-288.847 ncnn->-288.847
Mean:   keras->1.669 ncnn->1.658        Var: keras->23.166 ncnn->23.488
Cosine Similarity: 0.06401
Keras Feature Map:      [74.557 30.85  30.85  30.85  30.85  30.85  30.85  30.85  30.85  30.85 ]
Ncnn Feature Map:       [74.557 30.85  30.85  30.85  30.85  30.85  28.526  3.447 44.541 30.85 ]
==================================
Layer Name: expanded_conv_depthwise_BN, Layer Shape: keras->(1, 112, 112, 32) ncnn->(32, 112, 112)
Max:    keras->38.413 ncnn->38.342      Min: keras->-230.972 ncnn->-230.972
Mean:   keras->-3.192 ncnn->-3.189      Var: keras->29.318 ncnn->29.302
Cosine Similarity: 0.00132
Keras Feature Map:      [9.608 4.816 4.816 4.816 4.816 4.816 4.816 4.816 4.816 4.816]
Ncnn Feature Map:       [9.608 4.816 4.816 4.816 4.816 4.816 4.561 1.812 6.317 4.816]
==================================
Layer Name: expanded_conv_depthwise_relu, Layer Shape: keras->(1, 112, 112, 32) ncnn->(32, 112, 112)
Max:    keras->6.000 ncnn->6.000        Min: keras->0.000 ncnn->0.000
Mean:   keras->1.701 ncnn->1.708        Var: keras->1.862 ncnn->1.884
Cosine Similarity: 0.04312
Keras Feature Map:      [6.    4.816 4.816 4.816 4.816 4.816 4.816 4.816 4.816 4.816]
Ncnn Feature Map:       [6.    4.816 4.816 4.816 4.816 4.816 4.561 1.812 6.    4.816]
==================================
Layer Name: expanded_conv_project, Layer Shape: keras->(1, 112, 112, 16) ncnn->(16, 112, 112)
Max:    keras->8.220 ncnn->8.055        Min: keras->-9.847 ncnn->-10.327
Mean:   keras->-0.430 ncnn->-0.425      Var: keras->2.955 ncnn->2.987
Cosine Similarity: 0.05073
Keras Feature Map:      [0.164 2.679 2.413 1.057 3.501 2.567 2.168 1.826 1.166 5.222]
Ncnn Feature Map:       [0.507 2.555 2.355 3.644 5.137 3.235 3.546 2.002 0.857 6.214]
==================================
Layer Name: expanded_conv_project_BN, Layer Shape: keras->(1, 112, 112, 16) ncnn->(16, 112, 112)
Max:    keras->40.270 ncnn->41.348      Min: keras->-52.520 ncnn->-53.048
Mean:   keras->0.826 ncnn->0.835        Var: keras->9.311 ncnn->9.649
Cosine Similarity: 0.17899
Keras Feature Map:      [-1.003 11.171  9.884  3.32  15.148 10.628  8.695  7.041  3.845 23.478]
Ncnn Feature Map:       [ 0.658 10.57   9.6   15.838 23.065 13.862 15.364  7.895  2.35  28.276]
==================================
Layer Name: block_1_expand, Layer Shape: keras->(1, 112, 112, 96) ncnn->(96, 112, 112)
Max:    keras->33.430 ncnn->30.350      Min: keras->-30.575 ncnn->-32.065
Mean:   keras->0.140 ncnn->0.139        Var: keras->5.018 ncnn->5.212
Cosine Similarity: 0.17189
Keras Feature Map:      [5.454 3.778 3.796 4.599 3.877 3.119 4.261 4.342 4.295 3.976]
Ncnn Feature Map:       [5.485 3.759 3.795 4.639 3.816 2.953 4.292 3.506 4.485 3.721]
==================================
Layer Name: block_1_expand_BN, Layer Shape: keras->(1, 112, 112, 96) ncnn->(96, 112, 112)
Max:    keras->56.411 ncnn->56.430      Min: keras->-47.697 ncnn->-48.239
Mean:   keras->1.048 ncnn->1.043        Var: keras->5.363 ncnn->5.569
Cosine Similarity: 0.17103
Keras Feature Map:      [4.963 3.684 3.698 4.311 3.76  3.182 4.053 4.114 4.079 3.836]
Ncnn Feature Map:       [4.986 3.67  3.697 4.341 3.714 3.055 4.076 3.477 4.224 3.641]
==================================
Layer Name: block_1_expand_relu, Layer Shape: keras->(1, 112, 112, 96) ncnn->(96, 112, 112)
Max:    keras->6.000 ncnn->6.000        Min: keras->0.000 ncnn->0.000
Mean:   keras->1.740 ncnn->1.760        Var: keras->1.797 ncnn->1.820
Cosine Similarity: 0.08144
Keras Feature Map:      [4.963 3.684 3.698 4.311 3.76  3.182 4.053 4.114 4.079 3.836]
Ncnn Feature Map:       [4.986 3.67  3.697 4.341 3.714 3.055 4.076 3.477 4.224 3.641]
==================================
Layer Name: block_1_depthwise, Layer Shape: keras->(1, 56, 56, 96) ncnn->(96, 56, 56)
Max:    keras->56.350 ncnn->56.163      Min: keras->-55.251 ncnn->-55.074
Mean:   keras->-1.372 ncnn->-1.390      Var: keras->13.744 ncnn->13.857
Cosine Similarity: 0.02637
Keras Feature Map:      [-11.512 -13.881 -11.341 -12.985 -14.472 -12.519 -13.442 -11.298 -13.965
 -11.823]
Ncnn Feature Map:       [-11.428 -14.466 -12.129 -11.641 -12.8   -12.192 -13.454 -11.943 -13.571
 -11.593]
==================================
Layer Name: block_1_depthwise_BN, Layer Shape: keras->(1, 56, 56, 96) ncnn->(96, 56, 56)
Max:    keras->13.397 ncnn->13.061      Min: keras->-18.742 ncnn->-18.742
Mean:   keras->0.086 ncnn->0.080        Var: keras->2.249 ncnn->2.292
Cosine Similarity: 0.06648
Keras Feature Map:      [-2.012 -2.441 -1.982 -2.279 -2.547 -2.195 -2.361 -1.974 -2.456 -2.069]
Ncnn Feature Map:       [-1.997 -2.546 -2.124 -2.036 -2.245 -2.136 -2.363 -2.09  -2.385 -2.027]
==================================
Layer Name: block_1_depthwise_relu, Layer Shape: keras->(1, 56, 56, 96) ncnn->(96, 56, 56)
Max:    keras->6.000 ncnn->6.000        Min: keras->0.000 ncnn->0.000
Mean:   keras->0.789 ncnn->0.805        Var: keras->1.169 ncnn->1.186
Cosine Similarity: 0.05257
Keras Feature Map:      [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
Ncnn Feature Map:       [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
==================================
Layer Name: block_1_project, Layer Shape: keras->(1, 56, 56, 24) ncnn->(24, 56, 56)
Max:    keras->9.754 ncnn->9.189        Min: keras->-10.740 ncnn->-10.547
Mean:   keras->0.543 ncnn->0.533        Var: keras->2.347 ncnn->2.413
Cosine Similarity: 0.03144
Keras Feature Map:      [-0.344 -1.041 -0.088 -0.837 -1.526 -0.545 -1.569 -1.251 -1.256 -1.072]
Ncnn Feature Map:       [-0.398 -1.19   0.032  0.012 -1.421 -0.576 -1.612 -0.906 -1.415 -1.156]
==================================
Layer Name: block_1_project_BN, Layer Shape: keras->(1, 56, 56, 24) ncnn->(24, 56, 56)
Max:    keras->32.667 ncnn->30.751      Min: keras->-30.061 ncnn->-29.466
Mean:   keras->2.558 ncnn->2.557        Var: keras->6.991 ncnn->7.246
Cosine Similarity: 0.04500
Keras Feature Map:      [6.689 3.867 7.726 4.692 1.906 5.876 1.731 3.018 3.    3.745]
Ncnn Feature Map:       [6.471 3.267 8.212 8.131 2.332 5.75  1.559 4.413 2.356 3.405]
==================================
Layer Name: block_2_expand, Layer Shape: keras->(1, 56, 56, 144) ncnn->(144, 56, 56)
Max:    keras->19.606 ncnn->20.331      Min: keras->-18.509 ncnn->-20.511
Mean:   keras->0.188 ncnn->0.187        Var: keras->3.537 ncnn->3.667
Cosine Similarity: 0.04085
Keras Feature Map:      [ 0.699 -0.726 -1.033 -2.695 -2.522 -1.101 -0.468 -0.567 -0.213 -1.439]
Ncnn Feature Map:       [ 0.016 -1.823 -1.387 -3.46  -2.35  -1.615 -1.628 -0.867 -0.18   0.284]
==================================
Layer Name: block_2_expand_BN, Layer Shape: keras->(1, 56, 56, 144) ncnn->(144, 56, 56)
Max:    keras->16.836 ncnn->15.942      Min: keras->-17.446 ncnn->-18.943
Mean:   keras->0.744 ncnn->0.741        Var: keras->2.658 ncnn->2.724
Cosine Similarity: 0.03089
Keras Feature Map:      [ 1.245  0.45   0.278 -0.649 -0.553  0.24   0.594  0.539  0.736  0.051]
Ncnn Feature Map:       [ 0.864 -0.163  0.08  -1.076 -0.457 -0.047 -0.054  0.371  0.754  1.013]
==================================
Layer Name: block_2_expand_relu, Layer Shape: keras->(1, 56, 56, 144) ncnn->(144, 56, 56)
Max:    keras->6.000 ncnn->6.000        Min: keras->0.000 ncnn->0.000
Mean:   keras->1.435 ncnn->1.454        Var: keras->1.539 ncnn->1.566
Cosine Similarity: 0.02308
Keras Feature Map:      [1.245 0.45  0.278 0.    0.    0.24  0.594 0.539 0.736 0.051]
Ncnn Feature Map:       [0.864 0.    0.08  0.    0.    0.    0.    0.371 0.754 1.013]
==================================
Layer Name: block_2_depthwise, Layer Shape: keras->(1, 56, 56, 144) ncnn->(144, 56, 56)
Max:    keras->29.210 ncnn->29.364      Min: keras->-32.795 ncnn->-31.545
Mean:   keras->-0.718 ncnn->-0.724      Var: keras->3.783 ncnn->3.813
Cosine Similarity: 0.09767
Keras Feature Map:      [ 1.579 -0.819 -0.011 -0.289 -0.256 -0.33   0.098 -0.496  0.493 -0.676]
Ncnn Feature Map:       [ 1.348 -0.981  0.131 -0.083 -0.129 -0.462 -0.516  0.011  0.099  0.717]
==================================
Layer Name: block_2_depthwise_BN, Layer Shape: keras->(1, 56, 56, 144) ncnn->(144, 56, 56)
Max:    keras->15.019 ncnn->15.040      Min: keras->-36.505 ncnn->-34.366
Mean:   keras->0.263 ncnn->0.258        Var: keras->2.023 ncnn->2.043
Cosine Similarity: 0.04934
Keras Feature Map:      [2.32  1.006 1.449 1.296 1.314 1.274 1.508 1.183 1.725 1.084]
Ncnn Feature Map:       [2.194 0.917 1.527 1.409 1.384 1.201 1.172 1.461 1.509 1.848]
==================================
Layer Name: block_2_depthwise_relu, Layer Shape: keras->(1, 56, 56, 144) ncnn->(144, 56, 56)
Max:    keras->6.000 ncnn->6.000        Min: keras->0.000 ncnn->0.000
Mean:   keras->0.823 ncnn->0.827        Var: keras->1.225 ncnn->1.229
Cosine Similarity: 0.03737
Keras Feature Map:      [2.32  1.006 1.449 1.296 1.314 1.274 1.508 1.183 1.725 1.084]
Ncnn Feature Map:       [2.194 0.917 1.527 1.409 1.384 1.201 1.172 1.461 1.509 1.848]
==================================
Layer Name: block_2_project, Layer Shape: keras->(1, 56, 56, 24) ncnn->(24, 56, 56)
Max:    keras->10.754 ncnn->11.460      Min: keras->-8.685 ncnn->-8.611
Mean:   keras->0.002 ncnn->0.003        Var: keras->3.019 ncnn->3.030
Cosine Similarity: 0.01676
Keras Feature Map:      [-4.932 -6.166 -6.191 -2.435 -4.375 -4.226 -5.348 -3.984 -4.682 -5.825]
Ncnn Feature Map:       [-4.601 -6.095 -6.431 -2.633 -3.629 -4.757 -5.283 -4.632 -4.228 -5.402]
==================================
Layer Name: block_2_project_BN, Layer Shape: keras->(1, 56, 56, 24) ncnn->(24, 56, 56)
Max:    keras->31.454 ncnn->31.045      Min: keras->-35.277 ncnn->-35.486
Mean:   keras->-0.035 ncnn->-0.037      Var: keras->5.020 ncnn->5.047
Cosine Similarity: 0.17616
Keras Feature Map:      [ -5.707 -12.975 -13.125   9.002  -2.425  -1.546  -8.158  -0.123  -4.235
 -10.964]
Ncnn Feature Map:       [ -3.757 -12.558 -14.536   7.834   1.971  -4.675  -7.773  -3.941  -1.56
  -8.473]
==================================
Layer Name: block_3_expand, Layer Shape: keras->(1, 56, 56, 144) ncnn->(144, 56, 56)
Max:    keras->32.243 ncnn->34.155      Min: keras->-30.736 ncnn->-32.651
Mean:   keras->0.333 ncnn->0.306        Var: keras->5.098 ncnn->5.164
Cosine Similarity: 0.11152
Keras Feature Map:      [ -9.138 -12.581  -7.876  -4.882 -11.556  -8.262 -10.391  -6.598  -7.35
  -9.38 ]
Ncnn Feature Map:       [ -8.818 -14.048  -8.736  -5.631  -8.475  -9.623 -11.418  -8.07   -7.255
  -9.572]
==================================
Layer Name: block_3_expand_BN, Layer Shape: keras->(1, 56, 56, 144) ncnn->(144, 56, 56)
Max:    keras->19.432 ncnn->15.720      Min: keras->-18.111 ncnn->-16.139
Mean:   keras->0.400 ncnn->0.389        Var: keras->2.425 ncnn->2.447
Cosine Similarity: 0.09866
Keras Feature Map:      [-3.079 -3.989 -2.745 -1.954 -3.718 -2.847 -3.41  -2.408 -2.606 -3.143]
Ncnn Feature Map:       [-2.994 -4.377 -2.973 -2.152 -2.904 -3.207 -3.682 -2.797 -2.581 -3.194]
==================================
Layer Name: block_3_expand_relu, Layer Shape: keras->(1, 56, 56, 144) ncnn->(144, 56, 56)
Max:    keras->6.000 ncnn->6.000        Min: keras->0.000 ncnn->0.000
Mean:   keras->1.105 ncnn->1.106        Var: keras->1.417 ncnn->1.423
Cosine Similarity: 0.07342
Keras Feature Map:      [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
Ncnn Feature Map:       [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
==================================
Layer Name: block_3_depthwise, Layer Shape: keras->(1, 28, 28, 144) ncnn->(144, 28, 28)
Max:    keras->55.093 ncnn->54.854      Min: keras->-45.010 ncnn->-45.010
Mean:   keras->-1.357 ncnn->-1.368      Var: keras->9.868 ncnn->9.929
Cosine Similarity: 0.02394
Keras Feature Map:      [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
Ncnn Feature Map:       [0.   0.53 0.   0.   0.   0.   0.   0.   0.   0.  ]
==================================
Layer Name: block_3_depthwise_BN, Layer Shape: keras->(1, 28, 28, 144) ncnn->(144, 28, 28)
Max:    keras->10.205 ncnn->11.761      Min: keras->-10.628 ncnn->-10.463
Mean:   keras->0.802 ncnn->0.798        Var: keras->2.094 ncnn->2.114
Cosine Similarity: 0.03641
Keras Feature Map:      [0.098 0.098 0.098 0.098 0.098 0.098 0.098 0.098 0.098 0.098]
Ncnn Feature Map:       [0.098 0.299 0.098 0.098 0.098 0.098 0.098 0.098 0.098 0.098]
==================================
Layer Name: block_3_depthwise_relu, Layer Shape: keras->(1, 28, 28, 144) ncnn->(144, 28, 28)
Max:    keras->6.000 ncnn->6.000        Min: keras->0.000 ncnn->0.000
Mean:   keras->1.195 ncnn->1.191        Var: keras->1.429 ncnn->1.434
Cosine Similarity: 0.02897
Keras Feature Map:      [0.098 0.098 0.098 0.098 0.098 0.098 0.098 0.098 0.098 0.098]
Ncnn Feature Map:       [0.098 0.299 0.098 0.098 0.098 0.098 0.098 0.098 0.098 0.098]
==================================
Layer Name: block_3_project, Layer Shape: keras->(1, 28, 28, 32) ncnn->(32, 28, 28)
Max:    keras->13.341 ncnn->13.347      Min: keras->-13.860 ncnn->-13.582
Mean:   keras->0.683 ncnn->0.633        Var: keras->3.813 ncnn->3.841
Cosine Similarity: 0.02006
Keras Feature Map:      [1.194 2.517 1.627 1.734 2.637 3.416 2.514 3.417 1.821 1.396]
Ncnn Feature Map:       [1.297 2.566 1.924 2.264 3.882 3.539 2.574 3.78  2.082 1.748]
==================================
Layer Name: block_3_project_BN, Layer Shape: keras->(1, 28, 28, 32) ncnn->(32, 28, 28)
Max:    keras->26.176 ncnn->28.892      Min: keras->-32.849 ncnn->-32.084
Mean:   keras->-1.101 ncnn->-1.254      Var: keras->8.114 ncnn->8.226
Cosine Similarity: 0.04558
Keras Feature Map:      [ 1.822  7.855  3.794  4.284  8.402 11.958  7.84  11.962  4.682  2.741]
Ncnn Feature Map:       [ 2.29   8.081  5.149  6.701 14.082 12.518  8.118 13.616  5.87   4.349]
==================================
Layer Name: block_4_expand, Layer Shape: keras->(1, 28, 28, 192) ncnn->(192, 28, 28)
Max:    keras->15.516 ncnn->15.165      Min: keras->-15.786 ncnn->-15.530
Mean:   keras->0.045 ncnn->0.043        Var: keras->4.075 ncnn->4.164
Cosine Similarity: 0.02885
Keras Feature Map:      [ 1.2   -1.057  1.185 -0.447 -0.647 -1.683  0.435  0.01  -1.053 -0.692]
Ncnn Feature Map:       [ 1.174 -0.847  1.021 -0.467 -0.782 -1.834  0.282 -0.412 -1.048 -0.987]
==================================
Layer Name: block_4_expand_BN, Layer Shape: keras->(1, 28, 28, 192) ncnn->(192, 28, 28)
Max:    keras->8.950 ncnn->8.590        Min: keras->-10.094 ncnn->-10.155
Mean:   keras->0.795 ncnn->0.793        Var: keras->2.309 ncnn->2.353
Cosine Similarity: 0.02240
Keras Feature Map:      [ 1.387 -0.094  1.376  0.306  0.175 -0.504  0.884  0.606 -0.091  0.146]
Ncnn Feature Map:       [ 1.37   0.044  1.269  0.293  0.086 -0.604  0.785  0.329 -0.088 -0.048]
==================================
Layer Name: block_4_expand_relu, Layer Shape: keras->(1, 28, 28, 192) ncnn->(192, 28, 28)
Max:    keras->6.000 ncnn->6.000        Min: keras->0.000 ncnn->0.000
Mean:   keras->1.361 ncnn->1.372        Var: keras->1.425 ncnn->1.448
Cosine Similarity: 0.01887
Keras Feature Map:      [1.387 0.    1.376 0.306 0.175 0.    0.884 0.606 0.    0.146]
Ncnn Feature Map:       [1.37  0.044 1.269 0.293 0.086 0.    0.785 0.329 0.    0.   ]
==================================
Layer Name: block_4_depthwise, Layer Shape: keras->(1, 28, 28, 192) ncnn->(192, 28, 28)
Max:    keras->16.013 ncnn->16.448      Min: keras->-17.923 ncnn->-19.044
Mean:   keras->-0.554 ncnn->-0.556      Var: keras->2.946 ncnn->2.960
Cosine Similarity: 0.04600
Keras Feature Map:      [ 2.756  1.978  3.948 -0.102  0.852  1.825  2.31   0.533  1.15   2.323]
Ncnn Feature Map:       [3.158 2.797 3.813 0.085 0.565 1.217 2.046 0.952 1.475 2.059]
==================================
Layer Name: block_4_depthwise_BN, Layer Shape: keras->(1, 28, 28, 192) ncnn->(192, 28, 28)
Max:    keras->10.443 ncnn->10.553      Min: keras->-15.467 ncnn->-15.198
Mean:   keras->-0.416 ncnn->-0.413      Var: keras->2.405 ncnn->2.417
Cosine Similarity: 0.03073
Keras Feature Map:      [ 1.796  1.291  2.568 -0.058  0.561  1.192  1.507  0.354  0.754  1.515]
Ncnn Feature Map:       [2.057 1.823 2.481 0.063 0.375 0.797 1.335 0.625 0.965 1.343]
==================================
Layer Name: block_4_depthwise_relu, Layer Shape: keras->(1, 28, 28, 192) ncnn->(192, 28, 28)
Max:    keras->6.000 ncnn->6.000        Min: keras->0.000 ncnn->0.000
Mean:   keras->0.579 ncnn->0.580        Var: keras->1.049 ncnn->1.055
Cosine Similarity: 0.04255
Keras Feature Map:      [1.796 1.291 2.568 0.    0.561 1.192 1.507 0.354 0.754 1.515]
Ncnn Feature Map:       [2.057 1.823 2.481 0.063 0.375 0.797 1.335 0.625 0.965 1.343]
==================================
Layer Name: block_4_project, Layer Shape: keras->(1, 28, 28, 32) ncnn->(32, 28, 28)
Max:    keras->5.372 ncnn->5.313        Min: keras->-5.753 ncnn->-5.682
Mean:   keras->-0.354 ncnn->-0.351      Var: keras->1.512 ncnn->1.517
Cosine Similarity: 0.02631
Keras Feature Map:      [-1.483 -0.992 -2.551 -2.447 -1.916 -1.382 -2.169 -2.119 -2.191 -2.265]
Ncnn Feature Map:       [-1.382 -0.997 -2.379 -2.359 -1.921 -1.226 -2.178 -2.069 -1.87  -2.021]
==================================
Layer Name: block_4_project_BN, Layer Shape: keras->(1, 28, 28, 32) ncnn->(32, 28, 28)
Max:    keras->18.959 ncnn->20.461      Min: keras->-23.120 ncnn->-24.415
Mean:   keras->-0.132 ncnn->-0.136      Var: keras->4.081 ncnn->4.197
Cosine Similarity: 0.06806
Keras Feature Map:      [-1.433  1.184 -7.127 -6.574 -3.741 -0.895 -5.091 -4.825 -5.208 -5.6  ]
Ncnn Feature Map:       [-0.897  1.155 -6.211 -6.105 -3.767 -0.066 -5.14  -4.559 -3.496 -4.303]
==================================
Layer Name: block_5_expand, Layer Shape: keras->(1, 28, 28, 192) ncnn->(192, 28, 28)
Max:    keras->15.460 ncnn->13.610      Min: keras->-17.670 ncnn->-18.343
Mean:   keras->0.061 ncnn->0.037        Var: keras->3.762 ncnn->3.810
Cosine Similarity: 0.05036
Keras Feature Map:      [ -4.027  -5.846  -8.157  -6.977  -6.592  -6.676 -10.199  -5.452  -7.474
  -6.414]
Ncnn Feature Map:       [ -5.194  -4.714  -8.9    -8.205  -7.598  -7.175 -10.859  -6.05   -5.956
  -7.837]
==================================
Layer Name: block_5_expand_BN, Layer Shape: keras->(1, 28, 28, 192) ncnn->(192, 28, 28)
Max:    keras->9.167 ncnn->9.007        Min: keras->-10.003 ncnn->-10.419
Mean:   keras->0.568 ncnn->0.555        Var: keras->2.025 ncnn->2.051
Cosine Similarity: 0.03198
Keras Feature Map:      [-2.996 -4.012 -5.302 -4.643 -4.428 -4.475 -6.443 -3.792 -4.921 -4.329]
Ncnn Feature Map:       [-3.648 -3.379 -5.717 -5.329 -4.99  -4.753 -6.811 -4.126 -4.073 -5.124]
==================================
Layer Name: block_5_expand_relu, Layer Shape: keras->(1, 28, 28, 192) ncnn->(192, 28, 28)
Max:    keras->6.000 ncnn->6.000        Min: keras->0.000 ncnn->0.000
Mean:   keras->1.146 ncnn->1.149        Var: keras->1.224 ncnn->1.234
Cosine Similarity: 0.02337
Keras Feature Map:      [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
Ncnn Feature Map:       [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
==================================
Layer Name: block_5_depthwise, Layer Shape: keras->(1, 28, 28, 192) ncnn->(192, 28, 28)
Max:    keras->21.718 ncnn->21.719      Min: keras->-22.918 ncnn->-23.187
Mean:   keras->-0.433 ncnn->-0.436      Var: keras->3.358 ncnn->3.361
Cosine Similarity: 0.03346
Keras Feature Map:      [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
Ncnn Feature Map:       [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
==================================
Layer Name: block_5_depthwise_BN, Layer Shape: keras->(1, 28, 28, 192) ncnn->(192, 28, 28)
Max:    keras->13.431 ncnn->13.247      Min: keras->-16.809 ncnn->-17.059
Mean:   keras->-0.613 ncnn->-0.615      Var: keras->2.298 ncnn->2.295
Cosine Similarity: 0.05461
Keras Feature Map:      [-0.532 -0.532 -0.532 -0.532 -0.532 -0.532 -0.532 -0.532 -0.532 -0.532]
Ncnn Feature Map:       [-0.532 -0.532 -0.532 -0.532 -0.532 -0.532 -0.532 -0.532 -0.532 -0.532]
==================================
Layer Name: block_5_depthwise_relu, Layer Shape: keras->(1, 28, 28, 192) ncnn->(192, 28, 28)
Max:    keras->6.000 ncnn->6.000        Min: keras->0.000 ncnn->0.000
Mean:   keras->0.503 ncnn->0.500        Var: keras->1.052 ncnn->1.049
Cosine Similarity: 0.06603
Keras Feature Map:      [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
Ncnn Feature Map:       [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
==================================
Layer Name: block_5_project, Layer Shape: keras->(1, 28, 28, 32) ncnn->(32, 28, 28)
Max:    keras->5.105 ncnn->4.807        Min: keras->-3.813 ncnn->-4.189
Mean:   keras->0.155 ncnn->0.140        Var: keras->1.131 ncnn->1.121
Cosine Similarity: 0.06823
Keras Feature Map:      [-1.556 -2.3   -2.089 -2.473 -2.365 -3.019 -2.449 -2.012 -2.204 -2.243]
Ncnn Feature Map:       [-1.496 -2.036 -1.999 -2.591 -2.578 -2.986 -2.337 -2.342 -2.18  -2.566]
==================================
Layer Name: block_5_project_BN, Layer Shape: keras->(1, 28, 28, 32) ncnn->(32, 28, 28)
Max:    keras->17.586 ncnn->18.889      Min: keras->-19.626 ncnn->-21.118
Mean:   keras->-0.064 ncnn->-0.133      Var: keras->3.869 ncnn->3.878
Cosine Similarity: 0.10541
Keras Feature Map:      [ -5.048  -8.644  -7.625  -9.484  -8.961 -12.124  -9.369  -7.255  -8.181
  -8.369]
Ncnn Feature Map:       [ -4.756  -7.368  -7.19  -10.055  -9.993 -11.966  -8.827  -8.851  -8.068
  -9.931]
==================================
Layer Name: block_6_expand, Layer Shape: keras->(1, 28, 28, 192) ncnn->(192, 28, 28)
Max:    keras->27.534 ncnn->27.847      Min: keras->-29.319 ncnn->-29.360
Mean:   keras->-2.437 ncnn->-2.636      Var: keras->5.894 ncnn->5.949
Cosine Similarity: 0.05649
Keras Feature Map:      [ 0.115 -0.922 -1.988  0.512 -1.956  2.465  5.456 -0.95   1.962  0.186]
Ncnn Feature Map:       [-0.57  -2.647 -1.497  0.991 -1.928  3.052  5.133 -1.116  0.114  0.706]
==================================
Layer Name: block_6_expand_BN, Layer Shape: keras->(1, 28, 28, 192) ncnn->(192, 28, 28)
Max:    keras->14.883 ncnn->14.829      Min: keras->-13.867 ncnn->-14.911
Mean:   keras->-1.003 ncnn->-1.085      Var: keras->2.569 ncnn->2.582
Cosine Similarity: 0.04743
Keras Feature Map:      [-0.428 -0.755 -1.091 -0.302 -1.081  0.314  1.257 -0.763  0.155 -0.405]
Ncnn Feature Map:       [-0.644 -1.299 -0.936 -0.151 -1.072  0.499  1.155 -0.816 -0.428 -0.241]
==================================
Layer Name: block_6_expand_relu, Layer Shape: keras->(1, 28, 28, 192) ncnn->(192, 28, 28)
Max:    keras->6.000 ncnn->6.000        Min: keras->0.000 ncnn->0.000
Mean:   keras->0.548 ncnn->0.527        Var: keras->1.046 ncnn->1.032
Cosine Similarity: 0.06217
Keras Feature Map:      [0.    0.    0.    0.    0.    0.314 1.257 0.    0.155 0.   ]
Ncnn Feature Map:       [0.    0.    0.    0.    0.    0.499 1.155 0.    0.    0.   ]
==================================
Layer Name: block_6_depthwise, Layer Shape: keras->(1, 14, 14, 192) ncnn->(192, 14, 14)
Max:    keras->40.095 ncnn->38.838      Min: keras->-38.453 ncnn->-39.040
Mean:   keras->-0.898 ncnn->-0.853      Var: keras->6.489 ncnn->6.428
Cosine Similarity: 0.02185
Keras Feature Map:      [-0.416 -0.918 -1.855 -1.381 -0.372 -2.207 -0.293 -0.022 -0.729 -0.569]
Ncnn Feature Map:       [-0.523 -0.751 -2.42  -0.899 -0.007 -0.379 -0.054 -0.086 -1.412 -0.813]
==================================
Layer Name: block_6_depthwise_BN, Layer Shape: keras->(1, 14, 14, 192) ncnn->(192, 14, 14)
Max:    keras->12.617 ncnn->11.874      Min: keras->-7.672 ncnn->-8.614
Mean:   keras->1.410 ncnn->1.419        Var: keras->1.875 ncnn->1.873
Cosine Similarity: 0.01464
Keras Feature Map:      [4.61  4.422 4.073 4.25  4.626 3.941 4.656 4.757 4.493 4.553]
Ncnn Feature Map:       [4.57  4.485 3.862 4.43  4.762 4.624 4.745 4.733 4.238 4.461]
==================================
Layer Name: block_6_depthwise_relu, Layer Shape: keras->(1, 14, 14, 192) ncnn->(192, 14, 14)
Max:    keras->6.000 ncnn->6.000        Min: keras->0.000 ncnn->0.000
Mean:   keras->1.554 ncnn->1.556        Var: keras->1.594 ncnn->1.597
Cosine Similarity: 0.01086
Keras Feature Map:      [4.61  4.422 4.073 4.25  4.626 3.941 4.656 4.757 4.493 4.553]
Ncnn Feature Map:       [4.57  4.485 3.862 4.43  4.762 4.624 4.745 4.733 4.238 4.461]
==================================
Layer Name: block_6_project, Layer Shape: keras->(1, 14, 14, 64) ncnn->(64, 14, 14)
Max:    keras->10.529 ncnn->9.875       Min: keras->-10.507 ncnn->-10.119
Mean:   keras->-0.200 ncnn->-0.222      Var: keras->3.020 ncnn->3.039
Cosine Similarity: 0.01151
Keras Feature Map:      [3.786 4.784 4.127 4.653 4.73  3.973 5.026 4.681 4.641 5.061]
Ncnn Feature Map:       [3.964 4.783 4.007 4.993 4.827 4.894 4.689 4.489 4.086 4.917]
==================================
Layer Name: block_6_project_BN, Layer Shape: keras->(1, 14, 14, 64) ncnn->(64, 14, 14)
Max:    keras->20.518 ncnn->19.984      Min: keras->-18.833 ncnn->-17.758
Mean:   keras->1.267 ncnn->1.213        Var: keras->5.515 ncnn->5.525
Cosine Similarity: 0.03295
Keras Feature Map:      [-3.201 -0.387 -2.241 -0.756 -0.541 -2.675  0.296 -0.679 -0.791  0.395]
Ncnn Feature Map:       [-2.7   -0.391 -2.579  0.203 -0.265 -0.076 -0.657 -1.219 -2.355 -0.013]
==================================
Layer Name: block_7_expand, Layer Shape: keras->(1, 14, 14, 384) ncnn->(384, 14, 14)
Max:    keras->13.472 ncnn->13.561      Min: keras->-9.466 ncnn->-9.554
Mean:   keras->-0.308 ncnn->-0.325      Var: keras->2.678 ncnn->2.689
Cosine Similarity: 0.01915
Keras Feature Map:      [2.577 3.025 2.11  1.749 2.168 1.043 3.418 3.577 3.61  2.676]
Ncnn Feature Map:       [2.355 2.866 3.003 3.233 4.057 3.731 3.417 2.878 2.903 2.883]
==================================
Layer Name: block_7_expand_BN, Layer Shape: keras->(1, 14, 14, 384) ncnn->(384, 14, 14)
Max:    keras->8.738 ncnn->8.792        Min: keras->-7.485 ncnn->-6.848
Mean:   keras->0.466 ncnn->0.457        Var: keras->1.678 ncnn->1.677
Cosine Similarity: 0.01566
Keras Feature Map:      [1.067 1.346 0.776 0.55  0.812 0.11  1.591 1.691 1.711 1.128]
Ncnn Feature Map:       [0.928 1.247 1.333 1.476 1.99  1.787 1.591 1.254 1.27  1.258]
==================================
Layer Name: block_7_expand_relu, Layer Shape: keras->(1, 14, 14, 384) ncnn->(384, 14, 14)
Max:    keras->6.000 ncnn->6.000        Min: keras->0.000 ncnn->0.000
Mean:   keras->0.932 ncnn->0.926        Var: keras->1.056 ncnn->1.054
Cosine Similarity: 0.01193
Keras Feature Map:      [1.067 1.346 0.776 0.55  0.812 0.11  1.591 1.691 1.711 1.128]
Ncnn Feature Map:       [0.928 1.247 1.333 1.476 1.99  1.787 1.591 1.254 1.27  1.258]
==================================
Layer Name: block_7_depthwise, Layer Shape: keras->(1, 14, 14, 384) ncnn->(384, 14, 14)
Max:    keras->12.019 ncnn->12.395      Min: keras->-24.243 ncnn->-23.973
Mean:   keras->-0.371 ncnn->-0.363      Var: keras->2.204 ncnn->2.192
Cosine Similarity: 0.01606
Keras Feature Map:      [1.49  1.876 1.391 1.077 1.146 0.405 2.51  2.579 2.824 1.841]
Ncnn Feature Map:       [1.153 2.212 1.919 2.525 2.895 2.976 2.69  2.027 2.317 1.988]
==================================
Layer Name: block_7_depthwise_BN, Layer Shape: keras->(1, 14, 14, 384) ncnn->(384, 14, 14)
Max:    keras->7.803 ncnn->8.157        Min: keras->-14.968 ncnn->-15.154
Mean:   keras->-0.512 ncnn->-0.500      Var: keras->1.709 ncnn->1.697
Cosine Similarity: 0.02405
Keras Feature Map:      [ 0.59   1.114  0.457  0.031  0.123 -0.881  1.974  2.067  2.4    1.067]
Ncnn Feature Map:       [0.134 1.571 1.173 1.994 2.496 2.607 2.218 1.319 1.713 1.267]
==================================
Layer Name: block_7_depthwise_relu, Layer Shape: keras->(1, 14, 14, 384) ncnn->(384, 14, 14)
Max:    keras->6.000 ncnn->6.000        Min: keras->0.000 ncnn->0.000
Mean:   keras->0.308 ncnn->0.311        Var: keras->0.652 ncnn->0.662
Cosine Similarity: 0.04055
Keras Feature Map:      [0.59  1.114 0.457 0.031 0.123 0.    1.974 2.067 2.4   1.067]
Ncnn Feature Map:       [0.134 1.571 1.173 1.994 2.496 2.607 2.218 1.319 1.713 1.267]
==================================
Layer Name: block_7_project, Layer Shape: keras->(1, 14, 14, 64) ncnn->(64, 14, 14)
Max:    keras->5.462 ncnn->5.440        Min: keras->-3.609 ncnn->-3.684
Mean:   keras->0.124 ncnn->0.124        Var: keras->0.777 ncnn->0.783
Cosine Similarity: 0.02958
Keras Feature Map:      [-1.928 -0.348 -1.349 -0.51  -0.611 -0.767 -0.766 -0.633 -0.627 -0.654]
Ncnn Feature Map:       [-1.522 -0.188 -1.321 -0.388 -0.635 -0.528 -0.788 -0.964 -0.671 -0.508]
==================================
Layer Name: block_7_project_BN, Layer Shape: keras->(1, 14, 14, 64) ncnn->(64, 14, 14)
Max:    keras->21.777 ncnn->21.821      Min: keras->-13.385 ncnn->-12.442
Mean:   keras->0.297 ncnn->0.295        Var: keras->2.648 ncnn->2.679
Cosine Similarity: 0.03446
Keras Feature Map:      [-3.464  0.076 -2.168 -0.288 -0.514 -0.863 -0.86  -0.564 -0.549 -0.611]
Ncnn Feature Map:       [-2.556  0.435 -2.105 -0.013 -0.566 -0.328 -0.91  -1.305 -0.648 -0.282]
==================================
Layer Name: block_8_expand, Layer Shape: keras->(1, 14, 14, 384) ncnn->(384, 14, 14)
Max:    keras->12.334 ncnn->12.333      Min: keras->-12.568 ncnn->-13.217
Mean:   keras->-0.452 ncnn->-0.465      Var: keras->2.975 ncnn->2.985
Cosine Similarity: 0.01791
Keras Feature Map:      [1.64  3.753 3.818 4.587 3.787 4.602 4.848 5.796 6.186 5.533]
Ncnn Feature Map:       [1.894 4.114 4.329 5.281 4.921 6.031 5.819 5.652 5.543 5.082]
==================================
Layer Name: block_8_expand_BN, Layer Shape: keras->(1, 14, 14, 384) ncnn->(384, 14, 14)
Max:    keras->5.795 ncnn->5.741        Min: keras->-7.512 ncnn->-7.801
Mean:   keras->0.307 ncnn->0.300        Var: keras->1.631 ncnn->1.632
Cosine Similarity: 0.01598
Keras Feature Map:      [1.459 2.511 2.543 2.926 2.528 2.933 3.055 3.527 3.721 3.396]
Ncnn Feature Map:       [1.586 2.69  2.797 3.271 3.092 3.644 3.539 3.455 3.401 3.172]
==================================
Layer Name: block_8_expand_relu, Layer Shape: keras->(1, 14, 14, 384) ncnn->(384, 14, 14)
Max:    keras->5.795 ncnn->5.741        Min: keras->0.000 ncnn->0.000
Mean:   keras->0.812 ncnn->0.809        Var: keras->0.945 ncnn->0.946
Cosine Similarity: 0.01342
Keras Feature Map:      [1.459 2.511 2.543 2.926 2.528 2.933 3.055 3.527 3.721 3.396]
Ncnn Feature Map:       [1.586 2.69  2.797 3.271 3.092 3.644 3.539 3.455 3.401 3.172]
==================================
Layer Name: block_8_depthwise, Layer Shape: keras->(1, 14, 14, 384) ncnn->(384, 14, 14)
Max:    keras->11.430 ncnn->11.796      Min: keras->-21.831 ncnn->-21.273
Mean:   keras->-0.151 ncnn->-0.153      Var: keras->1.724 ncnn->1.713
Cosine Similarity: 0.02073
Keras Feature Map:      [0.373 0.859 2.556 1.83  3.022 2.278 2.984 2.654 2.707 3.694]
Ncnn Feature Map:       [0.432 1.037 2.703 2.254 3.353 2.434 3.13  2.996 2.721 3.374]
==================================
Layer Name: block_8_depthwise_BN, Layer Shape: keras->(1, 14, 14, 384) ncnn->(384, 14, 14)
Max:    keras->7.256 ncnn->8.148        Min: keras->-19.645 ncnn->-19.041
Mean:   keras->-0.565 ncnn->-0.566      Var: keras->1.667 ncnn->1.663
Cosine Similarity: 0.01991
Keras Feature Map:      [-0.777 -0.232  1.668  0.855  2.19   1.357  2.147  1.778  1.837  2.943]
Ncnn Feature Map:       [-0.71  -0.033  1.832  1.33   2.561  1.531  2.311  2.161  1.853  2.584]
==================================
Layer Name: block_8_depthwise_relu, Layer Shape: keras->(1, 14, 14, 384) ncnn->(384, 14, 14)
Max:    keras->6.000 ncnn->6.000        Min: keras->0.000 ncnn->0.000
Mean:   keras->0.322 ncnn->0.321        Var: keras->0.651 ncnn->0.652
Cosine Similarity: 0.03880
Keras Feature Map:      [0.    0.    1.668 0.855 2.19  1.357 2.147 1.778 1.837 2.943]
Ncnn Feature Map:       [0.    0.    1.832 1.33  2.561 1.531 2.311 2.161 1.853 2.584]
==================================
Layer Name: block_8_project, Layer Shape: keras->(1, 14, 14, 64) ncnn->(64, 14, 14)
Max:    keras->2.820 ncnn->2.759        Min: keras->-2.561 ncnn->-2.695
Mean:   keras->0.030 ncnn->0.035        Var: keras->0.735 ncnn->0.740
Cosine Similarity: 0.03164
Keras Feature Map:      [ 0.565  0.13  -0.182 -0.124 -0.243 -0.003 -0.396 -0.423 -0.247 -0.357]
Ncnn Feature Map:       [ 0.687 -0.05  -0.343 -0.311 -0.216 -0.16  -0.463 -0.44  -0.191 -0.169]
==================================
Layer Name: block_8_project_BN, Layer Shape: keras->(1, 14, 14, 64) ncnn->(64, 14, 14)
Max:    keras->13.704 ncnn->13.713      Min: keras->-9.934 ncnn->-9.665
Mean:   keras->0.198 ncnn->0.215        Var: keras->2.507 ncnn->2.532
Cosine Similarity: 0.05135
Keras Feature Map:      [2.503 1.421 0.645 0.787 0.493 1.091 0.11  0.044 0.483 0.208]
Ncnn Feature Map:       [ 2.809  0.971  0.242  0.322  0.558  0.698 -0.056  0.002  0.622  0.677]
==================================
Layer Name: block_9_expand, Layer Shape: keras->(1, 14, 14, 384) ncnn->(384, 14, 14)
Max:    keras->11.597 ncnn->10.968      Min: keras->-12.175 ncnn->-11.740
Mean:   keras->-0.243 ncnn->-0.238      Var: keras->2.822 ncnn->2.827
Cosine Similarity: 0.02385
Keras Feature Map:      [-0.042 -3.341 -2.896 -2.808 -3.98  -3.194 -3.474 -3.645 -3.661 -3.365]
Ncnn Feature Map:       [ 0.387 -3.378 -2.938 -2.865 -4.127 -3.789 -3.97  -3.943 -3.79  -3.621]
==================================
Layer Name: block_9_expand_BN, Layer Shape: keras->(1, 14, 14, 384) ncnn->(384, 14, 14)
Max:    keras->7.167 ncnn->7.357        Min: keras->-7.251 ncnn->-7.125
Mean:   keras->0.067 ncnn->0.072        Var: keras->1.621 ncnn->1.619
Cosine Similarity: 0.01928
Keras Feature Map:      [ 0.801 -0.406 -0.244 -0.211 -0.64  -0.353 -0.455 -0.517 -0.523 -0.415]
Ncnn Feature Map:       [ 0.958 -0.42  -0.259 -0.232 -0.694 -0.57  -0.636 -0.626 -0.57  -0.509]
==================================
Layer Name: block_9_expand_relu, Layer Shape: keras->(1, 14, 14, 384) ncnn->(384, 14, 14)
Max:    keras->6.000 ncnn->6.000        Min: keras->0.000 ncnn->0.000
Mean:   keras->0.681 ncnn->0.681        Var: keras->0.881 ncnn->0.884
Cosine Similarity: 0.01699
Keras Feature Map:      [0.801 0.    0.    0.    0.    0.    0.    0.    0.    0.   ]
Ncnn Feature Map:       [0.958 0.    0.    0.    0.    0.    0.    0.    0.    0.   ]
==================================
Layer Name: block_9_depthwise, Layer Shape: keras->(1, 14, 14, 384) ncnn->(384, 14, 14)
Max:    keras->9.783 ncnn->9.792        Min: keras->-13.228 ncnn->-13.372
Mean:   keras->-0.079 ncnn->-0.085      Var: keras->1.640 ncnn->1.638
Cosine Similarity: 0.02144
Keras Feature Map:      [-0.629  0.909  0.     0.     0.     0.     0.     0.     0.     0.   ]
Ncnn Feature Map:       [-0.725  1.06   0.     0.     0.     0.     0.     0.     0.     0.   ]
==================================
Layer Name: block_9_depthwise_BN, Layer Shape: keras->(1, 14, 14, 384) ncnn->(384, 14, 14)
Max:    keras->6.711 ncnn->6.764        Min: keras->-12.337 ncnn->-12.113
Mean:   keras->-0.554 ncnn->-0.558      Var: keras->1.598 ncnn->1.594
Cosine Similarity: 0.02125
Keras Feature Map:      [-1.569  0.701 -0.64  -0.64  -0.64  -0.64  -0.64  -0.64  -0.64  -0.64 ]
Ncnn Feature Map:       [-1.71   0.924 -0.64  -0.64  -0.64  -0.64  -0.64  -0.64  -0.64  -0.64 ]
==================================
Layer Name: block_9_depthwise_relu, Layer Shape: keras->(1, 14, 14, 384) ncnn->(384, 14, 14)
Max:    keras->6.000 ncnn->6.000        Min: keras->0.000 ncnn->0.000
Mean:   keras->0.337 ncnn->0.335        Var: keras->0.654 ncnn->0.651
Cosine Similarity: 0.03853
Keras Feature Map:      [0.    0.701 0.    0.    0.    0.    0.    0.    0.    0.   ]
Ncnn Feature Map:       [0.    0.924 0.    0.    0.    0.    0.    0.    0.    0.   ]
==================================
Layer Name: block_9_project, Layer Shape: keras->(1, 14, 14, 64) ncnn->(64, 14, 14)
Max:    keras->2.514 ncnn->2.342        Min: keras->-3.037 ncnn->-2.981
Mean:   keras->0.022 ncnn->0.034        Var: keras->0.711 ncnn->0.705
Cosine Similarity: 0.02985
Keras Feature Map:      [0.233 0.085 0.374 0.142 0.209 0.319 0.144 0.266 0.432 0.461]
Ncnn Feature Map:       [0.467 0.232 0.298 0.022 0.2   0.198 0.338 0.446 0.492 0.429]
==================================
Layer Name: block_9_project_BN, Layer Shape: keras->(1, 14, 14, 64) ncnn->(64, 14, 14)
Max:    keras->12.656 ncnn->11.602      Min: keras->-11.266 ncnn->-10.314
Mean:   keras->0.072 ncnn->0.106        Var: keras->2.108 ncnn->2.070
Cosine Similarity: 0.06969
Keras Feature Map:      [ 0.311 -0.079  0.683  0.073  0.248  0.538  0.078  0.399  0.836  0.912]
Ncnn Feature Map:       [ 0.928  0.309  0.484 -0.243  0.226  0.22   0.59   0.873  0.994  0.828]
==================================
Layer Name: block_10_expand, Layer Shape: keras->(1, 14, 14, 384) ncnn->(384, 14, 14)
Max:    keras->16.300 ncnn->15.026      Min: keras->-21.758 ncnn->-23.505
Mean:   keras->-1.663 ncnn->-1.714      Var: keras->3.974 ncnn->3.979
Cosine Similarity: 0.02627
Keras Feature Map:      [-1.345 -6.202 -8.325 -6.902 -5.135 -3.322 -6.181 -7.831 -8.231 -7.772]
Ncnn Feature Map:       [-1.482 -6.345 -7.881 -7.43  -7.862 -7.028 -8.388 -7.437 -7.354 -6.964]
==================================
Layer Name: block_10_expand_BN, Layer Shape: keras->(1, 14, 14, 384) ncnn->(384, 14, 14)
Max:    keras->6.773 ncnn->6.639        Min: keras->-11.529 ncnn->-11.059
Mean:   keras->-0.248 ncnn->-0.267      Var: keras->2.017 ncnn->2.017
Cosine Similarity: 0.01986
Keras Feature Map:      [ 0.182 -1.719 -2.549 -1.992 -1.301 -0.592 -1.71  -2.356 -2.512 -2.333]
Ncnn Feature Map:       [ 0.128 -1.774 -2.375 -2.199 -2.368 -2.042 -2.574 -2.202 -2.169 -2.016]
==================================
Layer Name: block_10_expand_relu, Layer Shape: keras->(1, 14, 14, 384) ncnn->(384, 14, 14)
Max:    keras->6.000 ncnn->6.000        Min: keras->0.000 ncnn->0.000
Mean:   keras->0.669 ncnn->0.657        Var: keras->0.930 ncnn->0.922
Cosine Similarity: 0.02062
Keras Feature Map:      [0.182 0.    0.    0.    0.    0.    0.    0.    0.    0.   ]
Ncnn Feature Map:       [0.128 0.    0.    0.    0.    0.    0.    0.    0.    0.   ]
==================================
Layer Name: block_10_depthwise, Layer Shape: keras->(1, 14, 14, 384) ncnn->(384, 14, 14)
Max:    keras->15.367 ncnn->14.483      Min: keras->-21.400 ncnn->-21.407
Mean:   keras->-0.371 ncnn->-0.368      Var: keras->2.764 ncnn->2.690
Cosine Similarity: 0.02607
Keras Feature Map:      [-0.66  -1.362 -1.169 -1.477 -1.851 -1.77  -1.729 -1.507 -1.188 -1.27 ]
Ncnn Feature Map:       [-0.358 -0.995 -0.627 -0.978 -1.189 -0.975 -1.105 -0.938 -1.174 -1.334]
==================================
Layer Name: block_10_depthwise_BN, Layer Shape: keras->(1, 14, 14, 384) ncnn->(384, 14, 14)
Max:    keras->6.833 ncnn->6.501        Min: keras->-10.710 ncnn->-10.714
Mean:   keras->0.400 ncnn->0.401        Var: keras->1.584 ncnn->1.575
Cosine Similarity: 0.01577
Keras Feature Map:      [1.969 1.802 1.848 1.775 1.686 1.705 1.715 1.768 1.843 1.824]
Ncnn Feature Map:       [2.041 1.89  1.977 1.894 1.843 1.894 1.863 1.903 1.847 1.809]
==================================
Layer Name: block_10_depthwise_relu, Layer Shape: keras->(1, 14, 14, 384) ncnn->(384, 14, 14)
Max:    keras->6.000 ncnn->6.000        Min: keras->0.000 ncnn->0.000
Mean:   keras->0.817 ncnn->0.814        Var: keras->1.050 ncnn->1.046
Cosine Similarity: 0.01097
Keras Feature Map:      [1.969 1.802 1.848 1.775 1.686 1.705 1.715 1.768 1.843 1.824]
Ncnn Feature Map:       [2.041 1.89  1.977 1.894 1.843 1.894 1.863 1.903 1.847 1.809]
==================================
Layer Name: block_10_project, Layer Shape: keras->(1, 14, 14, 96) ncnn->(96, 14, 14)
Max:    keras->5.723 ncnn->5.220        Min: keras->-7.205 ncnn->-6.375
Mean:   keras->-0.020 ncnn->-0.018      Var: keras->1.688 ncnn->1.666
Cosine Similarity: 0.01476
Keras Feature Map:      [-0.022 -0.693  0.182 -0.547 -0.38  -0.472 -0.421 -0.014  0.023  0.041]
Ncnn Feature Map:       [-0.344 -1.077  0.019 -0.494  0.02  -0.266 -0.183 -0.127 -0.282 -0.151]
==================================
Layer Name: block_10_project_BN, Layer Shape: keras->(1, 14, 14, 96) ncnn->(96, 14, 14)
Max:    keras->10.847 ncnn->9.661       Min: keras->-11.899 ncnn->-11.341
Mean:   keras->-0.005 ncnn->-0.005      Var: keras->2.277 ncnn->2.223
Cosine Similarity: 0.05331
Keras Feature Map:      [1.82  0.006 2.372 0.4   0.853 0.604 0.742 1.842 1.941 1.989]
Ncnn Feature Map:       [ 0.95  -1.031  1.931  0.544  1.934  1.16   1.384  1.537  1.116  1.47 ]
==================================
Layer Name: block_11_expand, Layer Shape: keras->(1, 14, 14, 576) ncnn->(576, 14, 14)
Max:    keras->5.664 ncnn->5.166        Min: keras->-5.562 ncnn->-5.832
Mean:   keras->-0.070 ncnn->-0.066      Var: keras->1.098 ncnn->1.074
Cosine Similarity: 0.04796
Keras Feature Map:      [-0.579 -0.     0.99   0.351  0.104 -0.747 -0.596 -0.521 -0.44  -0.525]
Ncnn Feature Map:       [-0.983 -0.959  0.575  0.369 -0.346 -0.808 -0.898 -0.698 -0.384 -0.309]
==================================
Layer Name: block_11_expand_BN, Layer Shape: keras->(1, 14, 14, 576) ncnn->(576, 14, 14)
Max:    keras->6.095 ncnn->6.037        Min: keras->-4.157 ncnn->-4.159
Mean:   keras->0.007 ncnn->0.011        Var: keras->1.113 ncnn->1.101
Cosine Similarity: 0.02260
Keras Feature Map:      [-0.58  -0.278  0.24  -0.094 -0.223 -0.668 -0.589 -0.549 -0.507 -0.551]
Ncnn Feature Map:       [-0.791 -0.778  0.023 -0.085 -0.458 -0.699 -0.746 -0.642 -0.478 -0.439]
==================================
Layer Name: block_11_expand_relu, Layer Shape: keras->(1, 14, 14, 576) ncnn->(576, 14, 14)
Max:    keras->6.000 ncnn->6.000        Min: keras->0.000 ncnn->0.000
Mean:   keras->0.466 ncnn->0.463        Var: keras->0.613 ncnn->0.608
Cosine Similarity: 0.01939
Keras Feature Map:      [0.   0.   0.24 0.   0.   0.   0.   0.   0.   0.  ]
Ncnn Feature Map:       [0.    0.    0.023 0.    0.    0.    0.    0.    0.    0.   ]
==================================
Layer Name: block_11_depthwise, Layer Shape: keras->(1, 14, 14, 576) ncnn->(576, 14, 14)
Max:    keras->8.136 ncnn->8.050        Min: keras->-10.333 ncnn->-10.389
Mean:   keras->-0.152 ncnn->-0.153      Var: keras->1.228 ncnn->1.208
Cosine Similarity: 0.02442
Keras Feature Map:      [0.    0.076 0.353 0.03  0.085 0.105 0.12  0.03  0.    0.   ]
Ncnn Feature Map:       [0.    0.007 0.034 0.    0.    0.    0.    0.    0.    0.   ]
==================================
Layer Name: block_11_depthwise_BN, Layer Shape: keras->(1, 14, 14, 576) ncnn->(576, 14, 14)
Max:    keras->7.134 ncnn->7.134        Min: keras->-8.407 ncnn->-8.024
Mean:   keras->-0.498 ncnn->-0.495      Var: keras->1.381 ncnn->1.370
Cosine Similarity: 0.01304
Keras Feature Map:      [-1.502 -1.439 -1.212 -1.477 -1.432 -1.415 -1.403 -1.477 -1.502 -1.502]
Ncnn Feature Map:       [-1.502 -1.496 -1.474 -1.502 -1.502 -1.502 -1.502 -1.502 -1.502 -1.502]
==================================
Layer Name: block_11_depthwise_relu, Layer Shape: keras->(1, 14, 14, 576) ncnn->(576, 14, 14)
Max:    keras->6.000 ncnn->6.000        Min: keras->0.000 ncnn->0.000
Mean:   keras->0.323 ncnn->0.321        Var: keras->0.618 ncnn->0.616
Cosine Similarity: 0.01276
Keras Feature Map:      [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
Ncnn Feature Map:       [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
==================================
Layer Name: block_11_project, Layer Shape: keras->(1, 14, 14, 96) ncnn->(96, 14, 14)
Max:    keras->2.998 ncnn->2.962        Min: keras->-3.332 ncnn->-3.569
Mean:   keras->-0.099 ncnn->-0.099      Var: keras->0.796 ncnn->0.796
Cosine Similarity: 0.01123
Keras Feature Map:      [-0.272 -0.54  -0.488 -0.214 -0.225 -0.357 -0.189 -0.139 -0.188 -0.143]
Ncnn Feature Map:       [-0.311 -0.504 -0.181  0.117 -0.093 -0.174 -0.222 -0.192 -0.108 -0.125]
==================================
Layer Name: block_11_project_BN, Layer Shape: keras->(1, 14, 14, 96) ncnn->(96, 14, 14)
Max:    keras->10.008 ncnn->9.777       Min: keras->-7.199 ncnn->-7.385
Mean:   keras->-0.012 ncnn->-0.015      Var: keras->1.215 ncnn->1.185
Cosine Similarity: 0.08616
Keras Feature Map:      [-0.495 -1.679 -1.446 -0.238 -0.287 -0.87  -0.128  0.093 -0.123  0.075]
Ncnn Feature Map:       [-0.665 -1.519 -0.094  1.224  0.295 -0.062 -0.272 -0.142  0.232  0.154]
==================================
Layer Name: block_12_expand, Layer Shape: keras->(1, 14, 14, 576) ncnn->(576, 14, 14)
Max:    keras->7.093 ncnn->7.374        Min: keras->-9.930 ncnn->-10.430
Mean:   keras->0.034 ncnn->0.050        Var: keras->1.244 ncnn->1.221
Cosine Similarity: 0.05010
Keras Feature Map:      [-0.691 -1.597 -0.523 -0.015 -0.383 -0.755 -0.792 -0.97  -0.892 -0.939]
Ncnn Feature Map:       [-0.836 -0.948 -0.041 -0.029 -0.451 -1.382 -1.209 -0.975 -0.801 -1.063]
==================================
Layer Name: block_12_expand_BN, Layer Shape: keras->(1, 14, 14, 576) ncnn->(576, 14, 14)
Max:    keras->4.634 ncnn->4.517        Min: keras->-8.392 ncnn->-7.488
Mean:   keras->-0.074 ncnn->-0.064      Var: keras->1.063 ncnn->1.044
Cosine Similarity: 0.02250
Keras Feature Map:      [ 0.427 -0.065  0.517  0.793  0.593  0.392  0.372  0.275  0.317  0.292]
Ncnn Feature Map:       [0.348 0.287 0.779 0.785 0.556 0.052 0.146 0.272 0.367 0.225]
==================================
Layer Name: block_12_expand_relu, Layer Shape: keras->(1, 14, 14, 576) ncnn->(576, 14, 14)
Max:    keras->4.634 ncnn->4.517        Min: keras->0.000 ncnn->0.000
Mean:   keras->0.406 ncnn->0.403        Var: keras->0.583 ncnn->0.576
Cosine Similarity: 0.01832
Keras Feature Map:      [0.427 0.    0.517 0.793 0.593 0.392 0.372 0.275 0.317 0.292]
Ncnn Feature Map:       [0.348 0.287 0.779 0.785 0.556 0.052 0.146 0.272 0.367 0.225]
==================================
Layer Name: block_12_depthwise, Layer Shape: keras->(1, 14, 14, 576) ncnn->(576, 14, 14)
Max:    keras->7.391 ncnn->8.197        Min: keras->-15.368 ncnn->-15.477
Mean:   keras->-0.023 ncnn->-0.021      Var: keras->1.286 ncnn->1.272
Cosine Similarity: 0.01765
Keras Feature Map:      [-0.818 -0.584 -0.957 -1.765 -2.205 -1.959 -1.4   -1.133 -1.047 -1.451]
Ncnn Feature Map:       [-0.832 -0.913 -1.257 -1.77  -1.822 -1.377 -0.796 -0.742 -1.006 -1.4  ]
==================================
Layer Name: block_12_depthwise_BN, Layer Shape: keras->(1, 14, 14, 576) ncnn->(576, 14, 14)
Max:    keras->4.954 ncnn->5.237        Min: keras->-8.463 ncnn->-8.457
Mean:   keras->-0.515 ncnn->-0.512      Var: keras->1.435 ncnn->1.431
Cosine Similarity: 0.00803
Keras Feature Map:      [-1.577 -1.39  -1.687 -2.332 -2.683 -2.487 -2.041 -1.828 -1.759 -2.081]
Ncnn Feature Map:       [-1.588 -1.653 -1.927 -2.336 -2.377 -2.022 -1.559 -1.516 -1.726 -2.041]
==================================
Layer Name: block_12_depthwise_relu, Layer Shape: keras->(1, 14, 14, 576) ncnn->(576, 14, 14)
Max:    keras->4.954 ncnn->5.237        Min: keras->0.000 ncnn->0.000
Mean:   keras->0.356 ncnn->0.355        Var: keras->0.634 ncnn->0.634
Cosine Similarity: 0.00800
Keras Feature Map:      [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
Ncnn Feature Map:       [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
==================================
Layer Name: block_12_project, Layer Shape: keras->(1, 14, 14, 96) ncnn->(96, 14, 14)
Max:    keras->2.589 ncnn->2.678        Min: keras->-2.596 ncnn->-2.473
Mean:   keras->-0.032 ncnn->-0.032      Var: keras->0.819 ncnn->0.818
Cosine Similarity: 0.00723
Keras Feature Map:      [0.632 0.714 1.078 1.251 1.204 1.366 1.292 1.049 1.024 1.114]
Ncnn Feature Map:       [0.609 0.673 1.179 1.206 1.118 1.021 1.251 1.138 1.117 1.071]
==================================
Layer Name: block_12_project_BN, Layer Shape: keras->(1, 14, 14, 96) ncnn->(96, 14, 14)
Max:    keras->11.575 ncnn->12.236      Min: keras->-10.420 ncnn->-12.280
Mean:   keras->-0.009 ncnn->-0.008      Var: keras->1.412 ncnn->1.408
Cosine Similarity: 0.09158
Keras Feature Map:      [-2.567 -1.992  0.533  1.734  1.409  2.533  2.016  0.33   0.157  0.785]
Ncnn Feature Map:       [-2.724 -2.278  1.237  1.421  0.813  0.137  1.734  0.946  0.803  0.485]
==================================
Layer Name: block_13_expand, Layer Shape: keras->(1, 14, 14, 576) ncnn->(576, 14, 14)
Max:    keras->13.085 ncnn->14.786      Min: keras->-10.499 ncnn->-10.127
Mean:   keras->0.245 ncnn->0.303        Var: keras->2.026 ncnn->1.989
Cosine Similarity: 0.04355
Keras Feature Map:      [ 2.562  0.332 -1.759 -0.86   0.08  -0.055  0.257  0.051 -0.02  -0.094]
Ncnn Feature Map:       [ 2.442 -0.091 -1.685 -1.49   0.26  -0.271 -0.165  0.482  0.171 -0.477]
==================================
Layer Name: block_13_expand_BN, Layer Shape: keras->(1, 14, 14, 576) ncnn->(576, 14, 14)
Max:    keras->6.427 ncnn->6.364        Min: keras->-5.845 ncnn->-5.737
Mean:   keras->-0.794 ncnn->-0.767      Var: keras->1.157 ncnn->1.135
Cosine Similarity: 0.01865
Keras Feature Map:      [-0.684 -1.667 -2.589 -2.193 -1.779 -1.838 -1.7   -1.791 -1.823 -1.855]
Ncnn Feature Map:       [-0.737 -1.854 -2.557 -2.471 -1.699 -1.933 -1.886 -1.601 -1.738 -2.024]
==================================
Layer Name: block_13_expand_relu, Layer Shape: keras->(1, 14, 14, 576) ncnn->(576, 14, 14)
Max:    keras->6.000 ncnn->6.000        Min: keras->0.000 ncnn->0.000
Mean:   keras->0.182 ncnn->0.181        Var: keras->0.440 ncnn->0.441
Cosine Similarity: 0.02868
Keras Feature Map:      [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
Ncnn Feature Map:       [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
==================================
Layer Name: block_13_depthwise, Layer Shape: keras->(1, 7, 7, 576) ncnn->(576, 7, 7)
Max:    keras->8.863 ncnn->9.132        Min: keras->-13.892 ncnn->-13.369
Mean:   keras->-0.300 ncnn->-0.290      Var: keras->1.482 ncnn->1.472
Cosine Similarity: 0.01400
Keras Feature Map:      [0.    0.    0.    0.    0.    0.    0.578]
Ncnn Feature Map:       [0.    0.    0.    0.    0.    0.    1.191]
==================================
Layer Name: block_13_depthwise_BN, Layer Shape: keras->(1, 7, 7, 576) ncnn->(576, 7, 7)
Max:    keras->4.859 ncnn->4.944        Min: keras->-3.475 ncnn->-3.424
Mean:   keras->0.829 ncnn->0.833        Var: keras->1.350 ncnn->1.350
Cosine Similarity: 0.00214
Keras Feature Map:      [-0.036 -0.036 -0.036 -0.036 -0.036 -0.036  0.303]
Ncnn Feature Map:       [-0.036 -0.036 -0.036 -0.036 -0.036 -0.036  0.662]
==================================
Layer Name: block_13_depthwise_relu, Layer Shape: keras->(1, 7, 7, 576) ncnn->(576, 7, 7)
Max:    keras->4.859 ncnn->4.944        Min: keras->0.000 ncnn->0.000
Mean:   keras->0.898 ncnn->0.901        Var: keras->1.273 ncnn->1.275
Cosine Similarity: 0.00173
Keras Feature Map:      [0.    0.    0.    0.    0.    0.    0.303]
Ncnn Feature Map:       [0.    0.    0.    0.    0.    0.    0.662]
==================================
Layer Name: block_13_project, Layer Shape: keras->(1, 7, 7, 160) ncnn->(160, 7, 7)
Max:    keras->6.744 ncnn->6.694        Min: keras->-5.335 ncnn->-5.323
Mean:   keras->0.210 ncnn->0.217        Var: keras->2.102 ncnn->2.104
Cosine Similarity: 0.00195
Keras Feature Map:      [4.588 4.455 3.943 4.061 4.358 4.627 4.937]
Ncnn Feature Map:       [4.498 4.585 4.086 4.033 4.362 4.669 4.912]
==================================
Layer Name: block_13_project_BN, Layer Shape: keras->(1, 7, 7, 160) ncnn->(160, 7, 7)
Max:    keras->8.224 ncnn->8.752        Min: keras->-9.365 ncnn->-9.951
Mean:   keras->0.153 ncnn->0.177        Var: keras->1.646 ncnn->1.643
Cosine Similarity: 0.03243
Keras Feature Map:      [ 0.591  0.135 -1.623 -1.216 -0.198  0.726  1.791]
Ncnn Feature Map:       [ 0.283  0.582 -1.132 -1.313 -0.182  0.871  1.705]
==================================
Layer Name: block_14_expand, Layer Shape: keras->(1, 7, 7, 960) ncnn->(960, 7, 7)
Max:    keras->6.425 ncnn->6.087        Min: keras->-7.111 ncnn->-8.011
Mean:   keras->-0.027 ncnn->-0.021      Var: keras->0.976 ncnn->0.975
Cosine Similarity: 0.02495
Keras Feature Map:      [ 0.199 -0.161  0.397  0.224 -0.205 -0.118  0.584]
Ncnn Feature Map:       [ 0.444 -0.001  0.212  0.192 -0.243  0.022  0.518]
==================================
Layer Name: block_14_expand_BN, Layer Shape: keras->(1, 7, 7, 960) ncnn->(960, 7, 7)
Max:    keras->5.887 ncnn->5.496        Min: keras->-7.565 ncnn->-7.595
Mean:   keras->0.329 ncnn->0.333        Var: keras->1.017 ncnn->1.009
Cosine Similarity: 0.01295
Keras Feature Map:      [0.772 0.498 0.923 0.791 0.465 0.531 1.064]
Ncnn Feature Map:       [0.959 0.62  0.782 0.767 0.436 0.637 1.014]
==================================
Layer Name: block_14_expand_relu, Layer Shape: keras->(1, 7, 7, 960) ncnn->(960, 7, 7)
Max:    keras->5.887 ncnn->5.496        Min: keras->0.000 ncnn->0.000
Mean:   keras->0.602 ncnn->0.601        Var: keras->0.660 ncnn->0.658
Cosine Similarity: 0.01058
Keras Feature Map:      [0.772 0.498 0.923 0.791 0.465 0.531 1.064]
Ncnn Feature Map:       [0.959 0.62  0.782 0.767 0.436 0.637 1.014]
==================================
Layer Name: block_14_depthwise, Layer Shape: keras->(1, 7, 7, 960) ncnn->(960, 7, 7)
Max:    keras->10.581 ncnn->10.334      Min: keras->-10.765 ncnn->-10.391
Mean:   keras->-0.198 ncnn->-0.195      Var: keras->1.392 ncnn->1.385
Cosine Similarity: 0.01359
Keras Feature Map:      [-1.25  -1.75  -1.459 -1.737 -1.788 -2.079 -1.343]
Ncnn Feature Map:       [-1.629 -2.041 -1.597 -1.63  -1.739 -1.856 -1.238]
==================================
Layer Name: block_14_depthwise_BN, Layer Shape: keras->(1, 7, 7, 960) ncnn->(960, 7, 7)
Max:    keras->6.055 ncnn->5.664        Min: keras->-9.987 ncnn->-10.633
Mean:   keras->-0.560 ncnn->-0.557      Var: keras->1.342 ncnn->1.339
Cosine Similarity: 0.00914
Keras Feature Map:      [-1.522 -2.013 -1.727 -2.    -2.05  -2.336 -1.613]
Ncnn Feature Map:       [-1.893 -2.299 -1.863 -1.895 -2.002 -2.117 -1.51 ]
==================================
Layer Name: block_14_depthwise_relu, Layer Shape: keras->(1, 7, 7, 960) ncnn->(960, 7, 7)
Max:    keras->6.000 ncnn->5.664        Min: keras->0.000 ncnn->0.000
Mean:   keras->0.257 ncnn->0.256        Var: keras->0.528 ncnn->0.528
Cosine Similarity: 0.01076
Keras Feature Map:      [0. 0. 0. 0. 0. 0. 0.]
Ncnn Feature Map:       [0. 0. 0. 0. 0. 0. 0.]
==================================
Layer Name: block_14_project, Layer Shape: keras->(1, 7, 7, 160) ncnn->(160, 7, 7)
Max:    keras->3.288 ncnn->3.119        Min: keras->-3.165 ncnn->-3.122
Mean:   keras->-0.130 ncnn->-0.130      Var: keras->0.773 ncnn->0.778
Cosine Similarity: 0.00886
Keras Feature Map:      [-0.939 -0.588 -0.616 -0.618 -0.723 -0.444 -0.923]
Ncnn Feature Map:       [-0.979 -0.728 -0.729 -0.742 -0.743 -0.316 -0.885]
==================================
Layer Name: block_14_project_BN, Layer Shape: keras->(1, 7, 7, 160) ncnn->(160, 7, 7)
Max:    keras->7.290 ncnn->7.449        Min: keras->-7.493 ncnn->-8.138
Mean:   keras->-0.199 ncnn->-0.196      Var: keras->1.404 ncnn->1.410
Cosine Similarity: 0.03687
Keras Feature Map:      [-0.906  0.412  0.308  0.302 -0.095  0.953 -0.846]
Ncnn Feature Map:       [-1.056 -0.113 -0.117 -0.165 -0.17   1.436 -0.704]
==================================
Layer Name: block_15_expand, Layer Shape: keras->(1, 7, 7, 960) ncnn->(960, 7, 7)
Max:    keras->11.811 ncnn->11.756      Min: keras->-13.595 ncnn->-13.084
Mean:   keras->0.114 ncnn->0.134        Var: keras->1.396 ncnn->1.389
Cosine Similarity: 0.02409
Keras Feature Map:      [ 2.029 -0.232 -0.471  0.18  -0.203 -0.422  6.361]
Ncnn Feature Map:       [ 1.872 -0.371 -0.315  0.017 -0.15  -0.868  7.039]
==================================
Layer Name: block_15_expand_BN, Layer Shape: keras->(1, 7, 7, 960) ncnn->(960, 7, 7)
Max:    keras->6.861 ncnn->7.530        Min: keras->-4.381 ncnn->-4.316
Mean:   keras->0.169 ncnn->0.178        Var: keras->1.059 ncnn->1.055
Cosine Similarity: 0.01185
Keras Feature Map:      [ 0.402 -0.668 -0.781 -0.473 -0.654 -0.758  2.452]
Ncnn Feature Map:       [ 0.327 -0.733 -0.707 -0.55  -0.629 -0.969  2.772]
==================================
Layer Name: block_15_expand_relu, Layer Shape: keras->(1, 7, 7, 960) ncnn->(960, 7, 7)
Max:    keras->6.000 ncnn->6.000        Min: keras->0.000 ncnn->0.000
Mean:   keras->0.520 ncnn->0.522        Var: keras->0.687 ncnn->0.689
Cosine Similarity: 0.00889
Keras Feature Map:      [0.402 0.    0.    0.    0.    0.    2.452]
Ncnn Feature Map:       [0.327 0.    0.    0.    0.    0.    2.772]
==================================
Layer Name: block_15_depthwise, Layer Shape: keras->(1, 7, 7, 960) ncnn->(960, 7, 7)
Max:    keras->12.972 ncnn->13.108      Min: keras->-18.998 ncnn->-19.140
Mean:   keras->0.063 ncnn->0.073        Var: keras->1.800 ncnn->1.808
Cosine Similarity: 0.00651
Keras Feature Map:      [0.271 0.071 0.    0.    0.    2.068 1.65 ]
Ncnn Feature Map:       [0.22  0.058 0.    0.    0.    2.338 1.866]
==================================
Layer Name: block_15_depthwise_BN, Layer Shape: keras->(1, 7, 7, 960) ncnn->(960, 7, 7)
Max:    keras->3.286 ncnn->3.355        Min: keras->-11.179 ncnn->-11.642
Mean:   keras->-1.051 ncnn->-1.043      Var: keras->1.640 ncnn->1.638
Cosine Similarity: 0.00330
Keras Feature Map:      [-1.535 -1.741 -1.815 -1.815 -1.815  0.324 -0.108]
Ncnn Feature Map:       [-1.587 -1.755 -1.815 -1.815 -1.815  0.604  0.116]
==================================
Layer Name: block_15_depthwise_relu, Layer Shape: keras->(1, 7, 7, 960) ncnn->(960, 7, 7)
Max:    keras->3.286 ncnn->3.355        Min: keras->0.000 ncnn->0.000
Mean:   keras->0.227 ncnn->0.227        Var: keras->0.511 ncnn->0.512
Cosine Similarity: 0.00317
Keras Feature Map:      [0.    0.    0.    0.    0.    0.324 0.   ]
Ncnn Feature Map:       [0.    0.    0.    0.    0.    0.604 0.116]
==================================
Layer Name: block_15_project, Layer Shape: keras->(1, 7, 7, 160) ncnn->(160, 7, 7)
Max:    keras->1.743 ncnn->1.804        Min: keras->-2.490 ncnn->-2.512
Mean:   keras->-0.113 ncnn->-0.115      Var: keras->0.644 ncnn->0.643
Cosine Similarity: 0.00293
Keras Feature Map:      [ 0.386  0.265  0.173  0.169  0.137  0.215 -0.162]
Ncnn Feature Map:       [ 0.396  0.251  0.249  0.223  0.161  0.199 -0.175]
==================================
Layer Name: block_15_project_BN, Layer Shape: keras->(1, 7, 7, 160) ncnn->(160, 7, 7)
Max:    keras->10.321 ncnn->10.310      Min: keras->-12.182 ncnn->-9.518
Mean:   keras->0.005 ncnn->-0.009       Var: keras->1.567 ncnn->1.576
Cosine Similarity: 0.04918
Keras Feature Map:      [ 0.56  -0.703 -1.648 -1.695 -2.027 -1.22  -5.126]
Ncnn Feature Map:       [ 0.666 -0.845 -0.864 -1.129 -1.777 -1.385 -5.263]
==================================
Layer Name: block_16_expand, Layer Shape: keras->(1, 7, 7, 960) ncnn->(960, 7, 7)
Max:    keras->9.582 ncnn->9.373        Min: keras->-10.151 ncnn->-9.758
Mean:   keras->0.631 ncnn->0.690        Var: keras->1.722 ncnn->1.724
Cosine Similarity: 0.02676
Keras Feature Map:      [-2.551 -2.488 -0.667  0.433 -0.421 -1.029 -4.008]
Ncnn Feature Map:       [-3.36  -1.991  0.609  1.127 -0.113 -0.973 -3.153]
==================================
Layer Name: block_16_expand_BN, Layer Shape: keras->(1, 7, 7, 960) ncnn->(960, 7, 7)
Max:    keras->3.719 ncnn->3.771        Min: keras->-8.312 ncnn->-7.655
Mean:   keras->-0.479 ncnn->-0.458      Var: keras->0.822 ncnn->0.815
Cosine Similarity: 0.01433
Keras Feature Map:      [-2.275 -2.251 -1.543 -1.115 -1.447 -1.683 -2.842]
Ncnn Feature Map:       [-2.59  -2.057 -1.047 -0.845 -1.327 -1.662 -2.509]
==================================
Layer Name: block_16_expand_relu, Layer Shape: keras->(1, 7, 7, 960) ncnn->(960, 7, 7)
Max:    keras->3.719 ncnn->3.771        Min: keras->0.000 ncnn->0.000
Mean:   keras->0.144 ncnn->0.146        Var: keras->0.374 ncnn->0.377
Cosine Similarity: 0.01721
Keras Feature Map:      [0. 0. 0. 0. 0. 0. 0.]
Ncnn Feature Map:       [0. 0. 0. 0. 0. 0. 0.]
==================================
Layer Name: block_16_depthwise, Layer Shape: keras->(1, 7, 7, 960) ncnn->(960, 7, 7)
Max:    keras->10.387 ncnn->10.628      Min: keras->-15.548 ncnn->-17.169
Mean:   keras->-0.143 ncnn->-0.141      Var: keras->0.926 ncnn->0.936
Cosine Similarity: 0.00726
Keras Feature Map:      [0. 0. 0. 0. 0. 0. 0.]
Ncnn Feature Map:       [0. 0. 0. 0. 0. 0. 0.]
==================================
Layer Name: block_16_depthwise_BN, Layer Shape: keras->(1, 7, 7, 960) ncnn->(960, 7, 7)
Max:    keras->9.885 ncnn->9.269        Min: keras->-7.337 ncnn->-7.337
Mean:   keras->0.312 ncnn->0.319        Var: keras->1.347 ncnn->1.340
Cosine Similarity: 0.00586
Keras Feature Map:      [0.026 0.026 0.026 0.026 0.026 0.026 0.026]
Ncnn Feature Map:       [0.026 0.026 0.026 0.026 0.026 0.026 0.026]
==================================
Layer Name: block_16_depthwise_relu, Layer Shape: keras->(1, 7, 7, 960) ncnn->(960, 7, 7)
Max:    keras->6.000 ncnn->6.000        Min: keras->0.000 ncnn->0.000
Mean:   keras->0.595 ncnn->0.598        Var: keras->1.005 ncnn->1.002
Cosine Similarity: 0.00474
Keras Feature Map:      [0.026 0.026 0.026 0.026 0.026 0.026 0.026]
Ncnn Feature Map:       [0.026 0.026 0.026 0.026 0.026 0.026 0.026]
==================================
Layer Name: block_16_project, Layer Shape: keras->(1, 7, 7, 320) ncnn->(320, 7, 7)
Max:    keras->4.314 ncnn->4.289        Min: keras->-9.636 ncnn->-10.068
Mean:   keras->-0.052 ncnn->-0.055      Var: keras->1.264 ncnn->1.273
Cosine Similarity: 0.00516
Keras Feature Map:      [2.116 1.919 1.718 1.709 1.569 1.704 1.574]
Ncnn Feature Map:       [2.171 1.93  1.725 1.586 1.361 1.633 1.503]
==================================
Layer Name: block_16_project_BN, Layer Shape: keras->(1, 7, 7, 320) ncnn->(320, 7, 7)
Max:    keras->5.908 ncnn->5.719        Min: keras->-6.183 ncnn->-5.969
Mean:   keras->-0.081 ncnn->-0.089      Var: keras->1.533 ncnn->1.553
Cosine Similarity: 0.04160
Keras Feature Map:      [1.892 1.342 0.782 0.755 0.363 0.741 0.377]
Ncnn Feature Map:       [ 2.044  1.373  0.8    0.411 -0.217  0.542  0.18 ]
==================================
Layer Name: Conv_1, Layer Shape: keras->(1, 7, 7, 1280) ncnn->(1280, 7, 7)
Max:    keras->3.925 ncnn->3.910        Min: keras->-3.279 ncnn->-3.581
Mean:   keras->0.223 ncnn->0.176        Var: keras->0.835 ncnn->0.848
Cosine Similarity: 0.03831
Keras Feature Map:      [-0.477 -0.606 -0.395 -0.253  0.22   0.925  0.685]
Ncnn Feature Map:       [-0.723 -0.434 -0.256  0.266  0.537  1.186  0.739]
==================================
Layer Name: Conv_1_bn, Layer Shape: keras->(1, 7, 7, 1280) ncnn->(1280, 7, 7)
Max:    keras->13.670 ncnn->13.600      Min: keras->-19.339 ncnn->-20.992
Mean:   keras->-3.639 ncnn->-3.855      Var: keras->3.821 ncnn->3.876
Cosine Similarity: 0.02005
Keras Feature Map:      [-6.94  -7.525 -6.567 -5.921 -3.768 -0.563 -1.651]
Ncnn Feature Map:       [-8.057 -6.742 -5.932 -3.558 -2.326  0.624 -1.408]
==================================
Layer Name: out_relu, Layer Shape: keras->(1, 7, 7, 1280) ncnn->(1280, 7, 7)
Max:    keras->6.000 ncnn->6.000        Min: keras->0.000 ncnn->0.000
Mean:   keras->0.344 ncnn->0.323        Var: keras->1.055 ncnn->1.022
Cosine Similarity: 0.06342
Keras Feature Map:      [0. 0. 0. 0. 0. 0. 0.]
Ncnn Feature Map:       [0.    0.    0.    0.    0.    0.624 0.   ]
==================================
Layer Name: Logits_Softmax, Layer Shape: keras->(1, 1000) ncnn->(1, 1, 1000)
Max:    keras->0.609 ncnn->0.639        Min: keras->0.000 ncnn->0.000
Mean:   keras->0.001 ncnn->0.001        Var: keras->0.022 ncnn->0.022
Cosine Similarity: 0.00726
Keras Feature Map:      [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
Ncnn Feature Map:       [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
Top-k:
Keras Top-k:    794:0.609, 721:0.347, 549:0.009, 443:0.003, 411:0.002, 591:0.002, 879:0.002, 893:0.001, 636:0.001, 750:0.001
ncnn Top-k:     794:0.639, 721:0.268, 549:0.009, 411:0.009, 443:0.007, 591:0.006, 879:0.004, 709:0.003, 904:0.002, 750:0.002
```
