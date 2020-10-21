# keras2ncnn

### If you want me to support a model, you can open an issue and attach the keras h5df file. 

---
## Usage:
```
# If you only want to convert the model
python3 keras2ncnn.py -i SOME_H5DF_FILE.h5 -o DIR_TO_SAVE_NCNN_PARAM 

# You can see the structure of the converted model and the original model(after optimization)
python3 keras2ncnn.py -i SOME_H5DF_FILE.h5 -o DIR_TO_SAVE_NCNN_PARAM --plot_graph/-p

# You can see run the resulted C file, and load with --load_debug_load/-l to see the model output comparing to keras implementation
python3 keras2ncnn.py -i SOME_H5DF_FILE.h5 -o DIR_TO_SAVE_NCNN_PARAM --debug/-d
# Compile and run SOME_H5DF_FILE.c ans use &> to save the stdout
python3 keras2ncnn.py -i SOME_H5DF_FILE.h5 -o DIR_TO_SAVE_NCNN_PARAM --load_debug_load/-l SOME_H5DF_FILE_RUN_LOG_FILE
```
---
## Supported Op
- InputLayer
- Conv2D
- DepthwiseConv2D
- Add
- ZeroPadding2D
- ReLU (No leaky clip relu support)
- LeakyReLU
- UpSampling2D
- Concatenate
- GlobalAveragePooling2D
- MaxAveragePooling2D
- AveragePooling2D
- MaxPooling2D
- BatchNormalization
- Dense (With none, linear or softmax activation)
- Activation (relu only for now)

## Ops that dont work but have done coding


## Ops that have done coding but not checked yet


## Current status
- [X] Be able to convert MobilenetV2, load and excute
- [X] Get all current ops working
- [ ] Support more ops

