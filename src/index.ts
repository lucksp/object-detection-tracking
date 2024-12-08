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
 * @param checkCount: optional, how many items to track for each tracked object
 */
interface Props {
  height: number;
  width: number;
  recognitionCount?: number;
  score?: number;
  checkCount?: number;
}

/**
 * Class: FrameRecognition takes the moments "through time" to track each object as the data is fed from the frame processor.
 * @param trackingObjects: the record of all objects based on arbitrary numberical id as key.  The id is not related to anything from the frame processor, it's just for our own tracking.
 * @param recognizedObjectsCount: just to figure out what the next trackingObject key should be.
 */
export class FrameRecognition {
  private static readonly MAX_HISTORY_SIZE = 50;
  private static readonly MAX_MISSED_COUNT = 30;
  private static readonly TRACKING_THRESHOLD = 0.5;

  private trackingObjects: Map<number, RecognizedObject>;
  private recognizedObjectsCount: number;
  private readonly height: number;
  private readonly width: number;
  private readonly recognitionCount: number;
  private readonly scoreThreshold: number;
  private readonly checkCount: number;

  constructor({
    height,
    width,
    recognitionCount = 20,
    score = 0.25,
    checkCount = 50,
  }: Props) {
    this.height = height;
    this.width = width;
    this.trackingObjects = new Map();
    this.recognizedObjectsCount = 0;
    this.recognitionCount = recognitionCount;
    this.scoreThreshold = score;
    this.checkCount = checkCount;
  }

  // Memoize and optimize confident object retrieval
  private _cachedConfidentObject: RecognizedObject | null = null;
  private _lastConfidentObjectCheck = 0;

  /**
   * getConfidentObject are the most confident about.
   * It finds the object with the highest score & count.
   * @returns RecognizedObject
   * @todo allow more than 1 output item
   */
  getConfidentObject(): RecognizedObject | null {
    const currentTime = Date.now();
    if (
      this._cachedConfidentObject &&
      currentTime - this._lastConfidentObjectCheck < this.checkCount
    ) {
      return this._cachedConfidentObject;
    }

    let topObject: RecognizedObject | null = null;

    for (const recognized of this.trackingObjects.values()) {
      if (recognized.recognitionCount < this.recognitionCount) {
        continue;
      }

      if (
        !topObject ||
        recognized.score > topObject.score ||
        (recognized.score === topObject.score &&
          recognized.recognitionCount > topObject.recognitionCount)
      ) {
        topObject = recognized;
      }
    }

    this._cachedConfidentObject = topObject;
    this._lastConfidentObjectCheck = currentTime;
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
    const missedKeys = new Set(this.trackingObjects.keys());

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

      // Optimize bounding box transformation
      boundingBox.top =
        0.5 - ((0.5 - boundingBox.top) * this.width) / this.height;
      boundingBox.bottom =
        0.5 - ((0.5 - boundingBox.bottom) * this.width) / this.height;

      // Convert to number key to match Map type
      const recognizedKey = this.findTrackedObjectKey(boundingBox, label);
      if (recognizedKey === null) {
        this.addNewObject(boundingBox, label, score);
      } else {
        this.updateTrackingObject(recognizedKey, boundingBox, score);
        missedKeys.delete(recognizedKey);
      }
    }

    // Optimize penalty application
    for (const missedKey of missedKeys) {
      this.penalizeObject(missedKey);
    }

    // Invalidate cached confident object
    this._cachedConfidentObject = null;
  }

  /**
   * addNewObject() adds new record to trackingObjects
   */
  private addNewObject(boundingBox: BoundingBox, label: number, score: number) {
    // Increment first to start with 1 as the first key
    this.recognizedObjectsCount++;
    this.trackingObjects.set(this.recognizedObjectsCount, {
      label,
      boundingBox,
      boundingBoxes: [boundingBox],
      score,
      scores: [score],
      recognitionCount: 1,
      missedCount: 0,
    });
  }

  /**
   * updateTrackingObject() updates existing record in the trackingObjects.  It updates the averages for score, bounding box position.
   */
  private updateTrackingObject(
    key: number,
    boundingBox: BoundingBox,
    score: number
  ) {
    const trackedObject = this.trackingObjects.get(key);
    if (!trackedObject) return;

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
  private penalizeObject(key: number) {
    const trackedObject = this.trackingObjects.get(key);
    if (!trackedObject) return;

    trackedObject.missedCount++;

    if (trackedObject.missedCount >= FrameRecognition.MAX_MISSED_COUNT) {
      this.trackingObjects.delete(key);
    }
  }

  /**
   * findTrackedObjectKey() looks for existing bounding box within threshold.
   * @param boundingBox
   * @param label
   * @returns key from trackingObjects
   */
  private findTrackedObjectKey(
    boundingBox: BoundingBox,
    label: number
  ): number | null {
    for (const [key, tracked] of this.trackingObjects.entries()) {
      if (tracked.label !== label) continue;

      const isClose =
        Math.abs(boundingBox.top - tracked.boundingBox.top) <=
          FrameRecognition.TRACKING_THRESHOLD &&
        Math.abs(boundingBox.bottom - tracked.boundingBox.bottom) <=
          FrameRecognition.TRACKING_THRESHOLD &&
        Math.abs(boundingBox.left - tracked.boundingBox.left) <=
          FrameRecognition.TRACKING_THRESHOLD &&
        Math.abs(boundingBox.right - tracked.boundingBox.right) <=
          FrameRecognition.TRACKING_THRESHOLD;

      if (isClose) return key;
    }

    return null;
  }

  /**
   * updatingAveragedArray() adds a new element to the array, based on FIFO logic if over the threshold.
   */
  private updateAveragedArray<T>(values: T[], newValue: T): void {
    if (values.length >= FrameRecognition.MAX_HISTORY_SIZE) {
      values.shift();
    }
    values.push(newValue);
  }

  private getAverage(values: number[]): number {
    return values.reduce((acc, value) => acc + value, 0) / values.length;
  }

  private getAverageBoundingBox(boxes: BoundingBox[]): BoundingBox {
    const count = boxes.length;
    const total = boxes.reduce(
      (acc, box) => ({
        top: acc.top + box.top,
        left: acc.left + box.left,
        bottom: acc.bottom + box.bottom,
        right: acc.right + box.right,
      }),
      { top: 0, left: 0, bottom: 0, right: 0 }
    );

    return {
      top: total.top / count,
      left: total.left / count,
      bottom: total.bottom / count,
      right: total.right / count,
    };
  }

  resetAll(): void {
    this.trackingObjects.clear();
    this.recognizedObjectsCount = 0;
    this._cachedConfidentObject = null;
  }
}
