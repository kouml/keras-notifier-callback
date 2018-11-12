import requests
import json
from datetime import datetime

from keras.callbacks import Callback


class SlackNotifier(Callback):
    def __init__(self, url, every_log_steps=5, include_time=True):
        self.headers = {'Content-Type': 'application/json'}
        self.count = 0
        self.url = url
        self.every_log_steps = every_log_steps
        self.include_time = include_time

    def _jsonify(self, logs):
        log_str = "epochs:{}\n".format(self.count)
        if self.include_time:
            current_time = datetime.now().strftime('%Y/%m/%d-%H:%M:%S')
            log_str += '({}): '.format(current_time)
        for k, v in sorted(logs.items()):
            kv = "{}:{:.3f} ".format(k, v)
            log_str += kv
        return json.dumps({
            'text': log_str
        })

    def _post_message(self, content):
        try:
            res = requests.post(self.url,
                                data=content, headers=self.headers)
        except Exception:
            print('Error (%s): Can\'t send to slack' % (res))
        if not res.ok:
            print('Error status(%s): Can\'t send to slack' % (res))

    def on_epoch_end(self, epoch, logs=None):
        if self.count % self.every_log_steps == 0:
            content = self._jsonify(logs)
            self._post_message(content)
        self.count += 1

    def on_train_begin(self, logs=None):
        content = json.dumps({'text': '-'*20 + 'begin' + '-'*20})
        self._post_message(content)

    def on_train_end(self, logs=None):
        content = json.dumps({'text': '-'*20 + 'end' + '-'*20})
        self._post_message(content)


if __name__ == "__main__":
    import os

    from keras import Sequential
    from keras.layers import Dense, Activation
    import numpy as np

    model = Sequential()
    model.add(Dense(3, input_shape=(3, )))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    inp = np.array([[2, 3, 4], [3, 4, 5]])
    out = np.array([[1, 0, 0], [0, 0, 1]])

    model.fit(inp, out, epochs=30,
              callbacks=[SlackNotifier(os.getenv("SLACK_URL"), every_log_steps=10, include_time=True)])
