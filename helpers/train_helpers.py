
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

from tensorflow import keras

BATCH_SIZE = 4096
EPOCHS = 200 

def plot_model(history, model, fold, save = False):

    l_name = list(history.history.keys())[0]
    vl_name = list(history.history.keys())[2]
    a_name = list(history.history.keys())[1]
    al_name = list(history.history.keys())[3]

    loss, val_loss = history.history[l_name], history.history[vl_name]
    auc, val_auc = history.history[a_name], history.history[al_name]
    epochs = range(1, len(loss) + 1)

    plt.figure(figsize=(15, 8))
    plt.plot(
        epochs,
        loss,
        color="tab:blue",
        linestyle="-",
        linewidth=2,
        marker="*",
        label="Training loss",
    )
    plt.plot(
        epochs,
        val_loss,
        color="tab:orange",
        linestyle="-",
        marker="o",
        label="Validation loss",
    )
    plt.title(f"Training and validation loss, {model.name}, fold {fold}", fontsize=16)
    plt.xlabel("Epoch", fontsize=16)
    plt.ylabel("Loss", fontsize=16)
    plt.legend(loc="upper right", fontsize="x-large")
    plt.tick_params(labelsize=14)
    if save:
        plt.savefig(f'Loss_plots/loss_{model.name}_fold{fold}', bbox_inches='tight')
    plt.show()
    plt.clf()

    plt.figure(figsize=(15, 8))
    plt.plot(
        epochs,
        auc,
        color="tab:blue",
        linestyle="-",
        linewidth=2,
        marker="*",
        label="Training auc",
    )
    plt.plot(
        epochs,
        val_auc,
        color="tab:orange",
        linestyle="-",
        marker="o",
        label="Validation auc",
    )
    plt.title(f"Training and validation auc, {model.name}, fold {fold}", fontsize=16)
    plt.xlabel("Epoch", fontsize=16)
    plt.ylabel("Auc", fontsize=16)
    plt.legend(loc="upper left", fontsize="x-large")
    plt.tick_params(labelsize=14)
    if save:
        plt.savefig(f'Loss_plots/auc_{model.name}_fold_{fold}', bbox_inches='tight')
    plt.show()
    plt.clf()


def train_model(train, labels, model_in, n_folds=5):

    early_stopping = keras.callbacks.EarlyStopping(
                    patience=20, monitor="val_loss", restore_best_weights=True, verbose = 1
                )
    learn_reducer = keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', factor = 0.7, patience = 5, verbose = 1)
    kf = KFold(n_folds)
    store = []

    model_in.summary()

    for fold, (train_idx, val_idx) in enumerate(
        kf.split(train)
    ):
        
        print(f"Fitting fold {fold} for {model_in.name}...")
        model = keras.models.clone_model(model_in)
        model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=[keras.metrics.AUC()]
        )

        X_train, X_val = train.iloc[train_idx], train.iloc[val_idx]
        y_train, y_val = labels.iloc[train_idx], labels.iloc[val_idx]

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            verbose=1,
            batch_size=BATCH_SIZE,
            callbacks=[
                early_stopping, learn_reducer
            ],
        )
        auc = roc_auc_score(y_val, model.predict(X_val).squeeze())
        print(f"The val auc for fold {fold}, {model_in.name} is {auc}")

        plot_model(history, model, fold)

        store.append(auc)
            
    result = sum(store) / n_folds 
    return result
