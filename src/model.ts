// import * as tfconv from '@tensorflow/tfjs-converter';
// import * as tf from '@tensorflow/tfjs-core';
import * as tf from '@tensorflow/tfjs';

(window as any).tf = tf;


const DEFAULT_MODLE_URL = 'https://0.0.0.0:8081/bilibili/model.json';
const SELF_TRAINED_URL = 'https://0.0.0.0:8081/self_trained_01/weights_manifest.json';

export async function load(
  {
    maxFaces = 10,
    inputWidth = 224,
    inputHeight = 224,
    iouThreshold = 0.3,
    scoreThreshold = 0.75
  } = {},
  modelUrl = DEFAULT_MODLE_URL,
): Promise<any> {

  // modelUrl = SELF_TRAINED_URL;
  // const tfmodel = await tfconv.loadGraphModel(modelUrl);

  
  const tfmodel = await tf.loadGraphModel(modelUrl);
  // const tfmodel: any =  await tfjs.loadGraphModel(modelUrl);
  

  console.log('load tf graph model');
  const model = new ImageScoreModel(
    tfmodel, inputWidth, inputHeight, maxFaces, iouThreshold,
      scoreThreshold);
  return model;
}


/*
 * The object describing a face.
 */
export interface NormalizedFace {
  /** The upper left-hand corner of the face. */
  topLeft: [number, number]|tf.Tensor1D;
  /** The lower right-hand corner of the face. */
  bottomRight: [number, number]|tf.Tensor1D;
  /** Facial landmark coordinates. */
  landmarks?: number[][]|tf.Tensor2D;
  /** Probability of the face detection. */
  probability?: number|tf.Tensor1D;
}



// Blazeface scatters anchor points throughout the input image and for each
// point predicts the probability that it lies within a face. `ANCHORS_CONFIG`
// is a fixed configuration that determines where the anchor points are
// scattered.
const ANCHORS_CONFIG = {
  'strides': [8, 16],
  'anchors': [2, 6]
};

// `NUM_LANDMARKS` is a fixed property of the model.
const NUM_LANDMARKS = 6;

function generateAnchors(
    width: number, height: number,
    outputSpec: {strides: [number, number], anchors: [number, number]}):
    number[][] {
  const anchors = [];
  for (let i = 0; i < outputSpec.strides.length; i++) {
    const stride = outputSpec.strides[i];
    const gridRows = Math.floor((height + stride - 1) / stride);
    const gridCols = Math.floor((width + stride - 1) / stride);
    const anchorsNum = outputSpec.anchors[i];

    for (let gridY = 0; gridY < gridRows; gridY++) {
      const anchorY = stride * (gridY + 0.5);

      for (let gridX = 0; gridX < gridCols; gridX++) {
        const anchorX = stride * (gridX + 0.5);
        for (let n = 0; n < anchorsNum; n++) {
          anchors.push([anchorX, anchorY]);
        }
      }
    }
  }

  return anchors;
}



function getInputTensorDimensions(input: tf.Tensor3D|ImageData|HTMLVideoElement|
                                  HTMLImageElement|
                                  HTMLCanvasElement): [number, number] {
  return input instanceof tf.Tensor ? [input.shape[0], input.shape[1]] :
                                      [input.height, input.width];
}

function flipFaceHorizontal(
    face: NormalizedFace, imageWidth: number): NormalizedFace {
  let flippedTopLeft: [number, number]|tf.Tensor1D,
      flippedBottomRight: [number, number]|tf.Tensor1D,
      flippedLandmarks: number[][]|tf.Tensor2D;

  if (face.topLeft instanceof tf.Tensor &&
      face.bottomRight instanceof tf.Tensor) {
    const [topLeft, bottomRight] = tf.tidy(() => {
      return [
        tf.concat([
          tf.sub(imageWidth - 1, (face.topLeft as tf.Tensor).slice(0, 1)),
          (face.topLeft as tf.Tensor).slice(1, 1)
        ]) as tf.Tensor1D,
        tf.concat([
          tf.sub(imageWidth - 1, (face.bottomRight as tf.Tensor).slice(0, 1)),
          (face.bottomRight as tf.Tensor).slice(1, 1)
        ]) as tf.Tensor1D
      ];
    });

    flippedTopLeft = topLeft;
    flippedBottomRight = bottomRight;

    if (face.landmarks != null) {
      flippedLandmarks = tf.tidy(() => {
        const a: tf.Tensor2D =
            tf.sub(tf.tensor1d([imageWidth - 1, 0]), face.landmarks as any);
        const b = tf.tensor1d([1, -1]);
        const product: tf.Tensor2D = tf.mul(a, b);
        return product;
      });
    }
  } else {
    const [topLeftX, topLeftY] = face.topLeft as [number, number];
    const [bottomRightX, bottomRightY] = face.bottomRight as [number, number];

    flippedTopLeft = [imageWidth - 1 - topLeftX, topLeftY];
    flippedBottomRight = [imageWidth - 1 - bottomRightX, bottomRightY];

    if (face.landmarks != null) {
      flippedLandmarks =
          (face.landmarks as number[][]).map((coord: [number, number]) => ([
                                               imageWidth - 1 - coord[0],
                                               coord[1]
                                             ]));
    }
  }

  const flippedFace: NormalizedFace = {
    topLeft: flippedTopLeft,
    bottomRight: flippedBottomRight
  };

  if (flippedLandmarks != null) {
    flippedFace.landmarks = flippedLandmarks;
  }

  if (face.probability != null) {
    flippedFace.probability = face.probability instanceof tf.Tensor ?
        face.probability.clone() :
        face.probability;
  }

  return flippedFace;
}



export class ImageScoreModel {
  private ImageScoreModel: tf.GraphModel;
  private width: number;
  private height: number;
  private maxFaces: number;
  private anchors: tf.Tensor2D;
  private anchorsData: number[][];
  private inputSize: tf.Tensor1D;
  private inputSizeData: [number, number];
  private iouThreshold: number;
  private scoreThreshold: number;

  constructor(
      model: tf.GraphModel, width: number, height: number, maxFaces: number,
      iouThreshold: number, scoreThreshold: number) {
    this.ImageScoreModel = model;
    this.width = width;
    this.height = height;
    this.maxFaces = maxFaces;

    this.inputSizeData = [width, height];
    this.inputSize = tf.tensor1d([width, height]);

    this.iouThreshold = iouThreshold;
    this.scoreThreshold = scoreThreshold;
  }

  


  async estimateScore(
    input: tf.Tensor3D|ImageData|HTMLVideoElement|HTMLImageElement| HTMLCanvasElement,
    returnTensors = false, flipHorizontal = false,
    annotateBoxes = true
  ) {

    const [, width] = getInputTensorDimensions(input);
    const image: tf.Tensor4D = tf.tidy(() => {
      if (!(input instanceof tf.Tensor)) {
        input = tf.browser.fromPixels(input);
      }
      return (input as tf.Tensor).toFloat().expandDims(0);
    });
    
    // const [prediction] = 
    const score = tf.tidy(() => {
      
      // 图片二值化
      const resizedImage = image.resizeBilinear([this.width, this.height]);
      
      const normalizedImage = tf.mul(tf.sub(resizedImage.div(255), 0.5), 2);

      const batchedPrediction: any = this.ImageScoreModel.predict(normalizedImage);
      
      // batchedPrediction.print();
      const prediction = (batchedPrediction as tf.Tensor3D).squeeze();

      prediction.print();

      
      return prediction.as1D().dataSync();
    });

    if (score[0]) {
      return score[0];
    }
    
    return 0;

  }


}



