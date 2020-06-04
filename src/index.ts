// import * as tf from '@tensorflow/tfjs-core';
// import * as tfjsWasm from '@tensorflow/tfjs-backend-wasm';
// import * as tfconv from '@tensorflow/tfjs-converter';
import * as imagescore from './model';
import "core-js";
import * as tf from '@tensorflow/tfjs';

// tfjsWasm.setWasmPath('https://0.0.0.0:8081/tfjs-backend-wasm.wasm');



let model: any, ctx, videoWidth, videoHeight, video: any, canvas;

const state = {
  backend: 'webgl'
};



async function setupCamera() {
  video = document.getElementById('video');

  const stream = await navigator.mediaDevices.getUserMedia({
    'audio': false,
    'video': { facingMode: 'user' },
  });
  video.srcObject = stream;

  return new Promise((resolve) => {
    video.onloadedmetadata = () => {
      resolve(video);
    };
  });
}

const renderPrediction = async () => {
  

  console.time('predict');
  const score = await model.estimateScore2(video);
  const output = document.getElementById('output');
  output.innerText = score;
  console.timeEnd('predict');

  
  requestAnimationFrame(renderPrediction);
};

const setupPage = async () => {
  await tf.setBackend(state.backend);
  await setupCamera();
  video.play();

  videoWidth = video.videoWidth;
  videoHeight = video.videoHeight;
  video.width = videoWidth;
  video.height = videoHeight;

  model = await imagescore.load();

  renderPrediction();
};

setupPage();

const gui = new (window as any).dat.GUI();
gui.add(state, 'backend', ['wasm', 'webgl', 'cpu']).onChange(async (backend: string) => {
  await tf.setBackend(backend);
});
