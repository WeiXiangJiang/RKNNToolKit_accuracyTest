## build

push gcc to 3rdparty
modify `GCC_COMPILER` on `build.sh` for target platform, then execute

```
./build.sh
```

## install

connect device and push build output into `/userdata`

```
adb push install/ /userdata/
```

## run

```
adb shell
cd /userdata/install/
```

- rv1109/rv1126
```
./rknn_classification_demo model/mobilenet_v1_rv1109_rv1126.rknn data/dog_224x224.jpg
```
