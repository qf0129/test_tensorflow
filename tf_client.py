import tensorflow as tf
import requests
import json
import os


class TFClient(object):

    API_URL = 'http://127.0.0.1:8501/v1/models/fashion_model:predict'
    LABEL_LIST = ["T恤/上衣", "裤子", "套头衫", "连衣裙", "外套", "凉鞋", "衬衫", "运动鞋", "包", "短靴"]

    @classmethod
    def verify_image(cls, img_path):
        img = cls._load_image_array(img_path)
        if img is not None:
            ret_json = cls._post(json.dumps({"instances": img.tolist()}))
            if ret_json and 'predictions' in ret_json:
                predictions = ret_json['predictions']
                return cls._fill_ret(predictions)
            else:
                print('error: invalid ret. ret_json: %s' % ret_json)
        else:
            print('error: invalid img.')
        return None

    @classmethod
    def _fill_ret(cls, predictions):
        ret = []
        if predictions:
            for i, p in enumerate(predictions[0]):
                ret.append([
                    cls.LABEL_LIST[i],
                    p
                ])
        ret = sorted(ret, key=lambda x: x[1], reverse=True)
        return ret

    @classmethod
    def _load_image_array(cls, img_path):
        if os.path.exists(img_path):
            img = tf.io.read_file(img_path)
            if img_path.endswith(".png"):
                img = tf.image.decode_png(img)
            else:
                img = tf.image.decode_jpeg(img)
                img = tf.image.rgb_to_grayscale(img)
            img = tf.image.resize(img, [28, 28])
            img /= 255.0
            img = 1 - img
            img = img.numpy()
            return img
        return None

    @classmethod
    def _post(cls, data):
        ret = requests.post(cls.API_URL, data=data, headers={"content-type": "application/json"})
        if ret.status_code == 200 and ret.json():
            return ret.json()
        else:
            print('error: post error. code: %s, ret: %s' % (ret.status_code, ret.text))
        return None


# if __name__ == "__main__":
#     ret = TFClient.verify_image("/home/ubuntu/work/test_tf_server/imgs/bbb.jpg")
#     print(ret)
