import * as tfDefault from '@tensorflow/tfjs';
import { LayersModel } from '@tensorflow/tfjs';
import { ModelArtifacts } from './defines';

export async function exportModel(model: LayersModel) {
  const data: ModelArtifacts = {
    modelTopology: {},
    weightSpecs: [],
    weightData: '',
  };
  await model.save({
    save: async ({ modelTopology, weightData, weightSpecs }) => {
      Object.assign(data, {
        modelTopology,
        weightSpecs,
        weightData: arrayBufferToBase64(weightData as ArrayBuffer),
      });
      return {
        modelArtifactsInfo: {
          dateSaved: new Date(),
          modelTopologyType: 'JSON',
        },
      };
    },
  });
  return data;
}

export async function importModel(data: ModelArtifacts, tf = tfDefault) {
  const model = await tf.loadLayersModel({
    load: async () => {
      const { weightData, ...otherProps } = data;
      return {
        ...otherProps,
        weightData: base64ToArrayBuffer(weightData),
      };
    },
  });
  return model;
}

// 将对象转为格式化的字符串，数字自动取3位处理
export function formatParams(p: { [key: string]: number | string }) {
  return Object.keys(p)
    .map((k) => {
      const v = p[k];
      let str = `${k}: `;
      if (typeof v === 'string') {
        str += v;
      } else {
        if (v >= 0) {
          str += '+';
        }
        if (Math.round(v) === v) {
          str += v;
        } else {
          str += v.toFixed(3);
        }
      }
      return str;
    })
    .join(', ');
}

function arrayBufferToBase64(data: ArrayBuffer) {
  return btoa(String.fromCharCode(...new Uint8Array(data)));
}

function base64ToArrayBuffer(base64: string) {
  const binaryString = atob(base64);
  const len = binaryString.length;
  const bytes = new Uint8Array(len);

  for (let i = 0; i < len; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }

  return bytes.buffer;
}
