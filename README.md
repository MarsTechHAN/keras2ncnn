# keras2ncnn

## This project is still WIP. 

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

## Ops that dont work but have done coding
- BatchNormalization

## Ops that have dine coding but not checked yet
- Dense

## Current status
- [X] Be able to convert MobilenetV2, load and excute
- [ ] Get all current ops working
- [ ] Support more ops
