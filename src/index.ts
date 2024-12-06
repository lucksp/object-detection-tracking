export interface BoundingBox {
  top: number;
  left: number;
  bottom: number;
  right: number;
}
export interface RecognizedObject {
  boundingBoxes: BoundingBox[];
  boundingBox: BoundingBox;
  label: number;
  scores: number[];
  score: number;
  recognitionCount: number;
  // The nunber of frames since this was recongized
  missedCount: number;
}

/**
 * FrameRecognition arguments:
 * @param height: the height of the device
 * @param width: the width of the device
 * @param recognitionCount: optional, minimum number of items you want before being considered as a top object
 * @param score: optional, minimum threshold of box score to keep track of
 */
interface Props {
  height: number;
  width: number;
  recognitionCount?: number;
  score?: number;
}

/**
 * Class: FrameRecognition takes the moments "through time" to track each object as the data is fed from the frame processor.
 * @param trackingObjects: the record of all objects based on arbitrary numberical id as key.  The id is not related to anything from the frame processor, it's just for our own tracking.
 * @param recognizedObjectsCount: just to figure out what the next trackingObject key should be.
 */
export class FrameRecognition {
  trackingObjects: Record<string, RecognizedObject>;
  recognizedObjectsCount: number;
  height: number;
  width: number;
  recognitionCount: number;
  scoreThreshold: number;

  constructor({ height, width, recognitionCount = 20, score = 0.25 }: Props) {
    this.height = height;
    this.width = width;
    this.trackingObjects = {};
    this.recognizedObjectsCount = 0;
    this.recognitionCount = recognitionCount;
    this.scoreThreshold = score;
  }

  /**
   * getConfidentObject are the most confident about.
   * It finds the object with the highest score & count.
   * @returns RecognizedObject
   * @todo allow more than 1 output item
   */
  getConfidentObject() {
    let topObject: RecognizedObject | null = null;

    for (const recognized of Object.values(this.trackingObjects)) {
      if (recognized.recognitionCount < this.recognitionCount) {
        continue;
      }

      if (!topObject) {
        topObject = recognized;
      } else if (
        recognized.score > topObject.score ||
        (recognized.score === topObject.score &&
          recognized.recognitionCount > topObject.recognitionCount)
      ) {
        topObject = recognized;
      }
    }

    return topObject;
  }

  /**
   * addFrameData() handles the data directly from the output of the frame processor.  It looks for an existing & similar bounding box and either adds new or updates tracking object.
   *
   * Frame data from TFLite is comprised of:
   * @param boundingBoxes - from frame processor as `BoundingBox[]`.  Every 4 indexes relates to the index of the label.
   * @param labels - from frame processor as `{[x: string]: number}`.  Map the labels to the dictionary for human readible.
   * @param scores - from frame processor as `number[]`.
   */
  addFrameData(
    boundingBoxes: number[],
    labels: { [x: string]: number },
    scores: number[]
  ) {
    const missedKeys = new Set(Object.keys(this.trackingObjects));

    for (let i = 0; i < Object.keys(labels).length; i++) {
      const label = labels[i];
      const score = scores[i];

      if (score < this.scoreThreshold) {
        continue;
      }

      const boundingBox: BoundingBox = {
        left: boundingBoxes[i * 4],
        top: boundingBoxes[i * 4 + 1],
        right: boundingBoxes[i * 4 + 2],
        bottom: boundingBoxes[i * 4 + 3],
      };

      boundingBox.top =
        0.5 - ((0.5 - boundingBox.top) * this.width) / this.height;
      boundingBox.bottom =
        0.5 - ((0.5 - boundingBox.bottom) * this.width) / this.height;

      const recognizedKey = this.findTrackedObjectKey(boundingBox, label);
      if (!recognizedKey) {
        this.addNewObject(boundingBox, label, score);
      } else {
        this.updateTrackingObject(recognizedKey, boundingBox, score);
        missedKeys.delete(recognizedKey);
      }
    }

    for (const missedKey of missedKeys) {
      this.penalizeObject(missedKey);
    }
  }

  /**
   * addNewObject() adds new record to trackingObjects
   */
  addNewObject(boundingBox: BoundingBox, label: number, score: number) {
    const newId = this.recognizedObjectsCount + 1;
    this.trackingObjects[newId] = {
      label,
      boundingBox,
      boundingBoxes: [boundingBox],
      score,
      scores: [score],
      recognitionCount: 1,
      missedCount: 0,
    };
    this.recognizedObjectsCount = newId;
  }

  /**
   * updateTrackingObject() updates existing record in the trackingObjects.  It updates the averages for score, bounding box position.
   */
  updateTrackingObject(key: string, boundingBox: BoundingBox, score: number) {
    const trackedObject = this.trackingObjects[key];

    this.updateAveragedArray(trackedObject.scores, score);
    trackedObject.score = this.getAverage(trackedObject.scores);

    this.updateAveragedArray(trackedObject.boundingBoxes, boundingBox);
    trackedObject.boundingBox = this.getAverageBoundingBox(
      trackedObject.boundingBoxes
    );

    trackedObject.recognitionCount++;
  }

  /**
   * penalizeObject(): called when the object with the given key is not recognized in the latest frame data.  It adds to the missedCount so that if the threshold is reached, the entire key of trackingObject will be removed.
   */
  penalizeObject(key: string) {
    const trackedObject = this.trackingObjects[key];
    trackedObject.missedCount++;

    if (trackedObject.missedCount >= 30) {
      delete this.trackingObjects[key];
    }
  }

  /**
   * findTrackedObjectKey() looks for existing bounding box within threshold.
   * @param boundingBox
   * @param label
   * @returns key from trackingObjects
   */
  findTrackedObjectKey(boundingBox: BoundingBox, label: number): string | null {
    const threshold = 0.03;

    for (const [key, tracked] of Object.entries(this.trackingObjects)) {
      if (tracked.label !== label) continue;

      const topDelta = boundingBox.top - tracked.boundingBox.top;
      const bottomDelta = boundingBox.bottom - tracked.boundingBox.bottom;
      const leftDelta = boundingBox.left - tracked.boundingBox.left;
      const rightDelta = boundingBox.right - tracked.boundingBox.right;

      if (
        Math.abs(topDelta) <= threshold &&
        Math.abs(bottomDelta) <= threshold &&
        Math.abs(leftDelta) <= threshold &&
        Math.abs(rightDelta) <= threshold
      ) {
        return key;
      }
    }

    return null;
  }

  /**
   * updatingAveragedArray() adds a new element to the array, based on FIFO logic if over the threshold.
   */
  updateAveragedArray<T>(values: T[], newValue: T) {
    if (values.length >= 50) {
      values.shift();
    }
    values.push(newValue);
  }

  getAverage(values: number[]) {
    const sum = values.reduce((acc, value) => acc + value, 0);
    return sum / values.length;
  }

  getAverageBoundingBox(boxes: BoundingBox[]) {
    const total = boxes.reduce(
      (acc, box) => {
        acc.top += box.top;
        acc.left += box.left;
        acc.bottom += box.bottom;
        acc.right += box.right;
        return acc;
      },
      { top: 0, left: 0, bottom: 0, right: 0 }
    );

    const count = boxes.length;

    return {
      top: total.top / count,
      left: total.left / count,
      bottom: total.bottom / count,
      right: total.right / count,
    };
  }

  resetAll() {
    this.trackingObjects = {};
    this.recognizedObjectsCount = 0;
  }
}
