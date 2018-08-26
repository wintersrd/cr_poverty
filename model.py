import collections


def build_model(train_x, train_y):
    train_y = encode_labels(train_y)
    model = Sequential()
    model.add(Dense(256, input_dim=len(train_x.columns), activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(4, activation="softmax"))
    model.compile(
        # optimizer=keras.optimizers.RMSprop(lr=1e-2, decay=1e-8),
        optimizer=keras.optimizers.Adagrad(
            lr=0.001, beta_1=0.9, beta_2=0.999),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.fit(train_x.as_matrix(), train_y.as_matrix(),
              epochs=100, verbose=1, batch_size=32)
    return model


def encode_labels(label_list):
    res = []
    vals = [1, 2, 3, 4]
    for i in label_list:
        ret = {i: 1}
        for x in [x for x in vals if x != i]:
            ret[x] = 0
        res.append(ret)
    return pd.DataFrame(res, columns=vals)


def model_eval(model, val_data, val_labels):
    scores = model.predict(model_val.as_matrix())
    print('Expected: ' + str(dict(collections.Counter(model_val_label.tolist()))))
