# keras2ncnn

### If you want me to support a model, you can open an issue and attach the keras h5df file. 

---
## Usage:
```
# If you only want to convert the model
python3 -m keras2ncnn -i SOME_H5DF_FILE.h5 -o DIR_TO_SAVE_NCNN_PARAM 

# You can see the structure of the converted model and the original model(after optimization)
python3 -m keras2ncnn -i SOME_H5DF_FILE.h5 -o DIR_TO_SAVE_NCNN_PARAM --plot_graph/-p
```
---
## Supported Op
- InputLayer
- Conv2D 
- Conv2DTranspose 
- DepthwiseConv2D
- Add
- Multiply
- ZeroPadding2D
- ReLU
- LeakyReLU
- UpSampling2D
- Concatenate
- GlobalAveragePooling2D
- MaxAveragePooling2D
- AveragePooling2D
- MaxPooling2D
- BatchNormalization
- Dense
- Activation 

