# tflite-object-detection-tracking

## About

The goal of `tflite-object-detection-tracking` is to help smooth the output from a `*.tflite` model so that the UI is more friendly. The reason is that image recongition output can be finicky at times. The tflite model outputs the results in order from highest to lowest. A slight shift of the image can change the output which will trigger changes that the user doesn't expect or intend. Additionally, hand movements on mobile devices cause the bounding boxes to constantly shift dimensions.

This library works to allow smoothing of model outputs by taking into account the location of bounding boxes - with a threshold - and keeping track of these objects.

Many thanks to @drewag for coming up with the main concepts on how this works.

## Install

```bash
npm install tflite-object-detection-tracking
```

## Usage

1. Initialize the instance: `const tracker = new FrameRecognition(height, width);`
2. Add frame data from `tflite` output: `frameRecognition.addFrameData(boundingBoxes, labels, scores);`
3. Get the confident bounding box output: `const confidentObj = frameRecognition.getConfidentObject();`
4. Use the results to render UI output as needed.

## Contribution

Contributors are welcome! Please open a PR.

### Expo Users:

- To test your changes in your application, before PR & merge, you will need to modify the `metro.config` file as [described in the docs](https://docs.expo.dev/guides/monorepos/#modify-the-metro-config) to allow for sym-linking
