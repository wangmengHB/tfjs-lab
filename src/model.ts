// import * as tfconv from '@tensorflow/tfjs-converter';
// import * as tf from '@tensorflow/tfjs-core';
import * as tf from '@tensorflow/tfjs';
import { imgToTensor } from './imageUtilities';


(window as any).tf = tf;


const DEFAULT_MODLE_URL = 'https://0.0.0.0:8081/bilibili/model.json';
const SELF_TRAINED_URL = 'https://0.0.0.0:8081/self_trained_01/weights_manifest.json';

export async function load(
  {
    inputWidth = 224,
    inputHeight = 224,
  } = {},
  modelUrl = DEFAULT_MODLE_URL,
): Promise<any> {

  // modelUrl = SELF_TRAINED_URL;
  // const tfmodel = await tfconv.loadGraphModel(modelUrl);

  const tfmodel = await tf.loadGraphModel(modelUrl);
  // const tfmodel: any =  await tfjs.loadGraphModel(modelUrl);
  
  console.log('model loaded!!');
  const model = new ImageScoreModel(
    tfmodel, inputWidth, inputHeight);
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

const IMAGE_SIZE = 224;

export class ImageScoreModel {
  private scoreModel: tf.GraphModel;
  private width: number;
  private height: number;
  

  constructor(
      model: tf.GraphModel, 
      width: number, 
      height: number, 
  ) {
    this.scoreModel = model;
    this.width = width;
    this.height = height;
  }


  async estimateScore2(
    input: tf.Tensor3D|ImageData|HTMLVideoElement|HTMLImageElement| HTMLCanvasElement,
  ) {
    const imageResize = [IMAGE_SIZE, IMAGE_SIZE];
    const processedImg = imgToTensor(input, imageResize);
    const batchedPrediction: any = this.scoreModel.predict(processedImg);
    const prediction = (batchedPrediction as tf.Tensor3D).squeeze();
    prediction.print();
    console.log(prediction.as1D().dataSync());
    return prediction.as1D().dataSync();
  }

  
  async estimateScore(
    input: tf.Tensor3D|ImageData|HTMLVideoElement|HTMLImageElement| HTMLCanvasElement,
  ) {

    const image: tf.Tensor4D = tf.tidy(() => {
      if (!(input instanceof tf.Tensor)) {
        input = tf.browser.fromPixels(input);
      }
      return (input as tf.Tensor).toFloat().expandDims(0);
    });
    
    const score = tf.tidy(() => {   
      // 图片二值化
      const resizedImage = image.resizeBilinear([this.width, this.height]);
      const normalizedImage = tf.mul(tf.sub(resizedImage.div(255), 0.5), 2);
      const batchedPrediction: any = this.scoreModel.predict(normalizedImage);
      // batchedPrediction.print();
      const prediction = (batchedPrediction as tf.Tensor3D).squeeze();
      prediction.print();  
      return prediction.as1D().dataSync();
    });

    return score;

  }


}



