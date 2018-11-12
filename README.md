# keras-notifier-callback
This is a simple keras notifier callback.  
It is only available to the slack now.

## usage:
1. add your slack url in enviroment variable  
`export SLACK_URL="http:hooks.slack.com/services/xxxxxxx/yyyyyyyyyy"`

2. just append `SlackNotifier` in your callbacks


```python
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
```

and, you can get the result.

<img width="318" alt="result" src="https://user-images.githubusercontent.com/23252616/48365418-37593c00-e6ee-11e8-96bb-df8607cb6c04.png">
