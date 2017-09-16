import numpy as np
import pandas as pd
from keras.backend import set_image_dim_ordering
from keras.layers import Dense, Dropout
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from sklearn.utils import shuffle

set_image_dim_ordering('tf')
np.random.seed(42)

# FILEPATHS
filepath_combats = "data/combats.csv"
filepath_stats = "data/pokemon.csv"


def normalize_dataset(dataset):
    result = dataset.copy()
    for feature_name in dataset.columns:
        max_value = dataset[feature_name].max()
        min_value = dataset[feature_name].min()
        result[feature_name] = (dataset[feature_name] - min_value) / (max_value - min_value)

    return result


def load_and_format_stats():
    stats = pd.read_csv(filepath_stats, index_col='#')

    # Set Type columns to 'category':
    stats["Type 1"] = stats["Type 1"].astype('category')
    stats["Type 2"] = stats["Type 2"].astype('category')
    stats["Legendary"] = stats["Legendary"].astype('category')

    # Add new columns with categorical data:
    stats["Type_1_id"] = stats["Type 1"].cat.codes
    stats["Type_2_id"] = stats["Type 2"].cat.codes
    stats["Legendary_id"] = stats["Legendary"].cat.codes

    # Delete all the useless columns:
    del stats['Type 1']
    del stats['Type 2']
    del stats['Legendary']
    del stats['Name']

    # del stats['HP']
    # del stats['Attack']
    # del stats['Defense']
    # del stats['Sp. Atk']
    # del stats['Sp. Def']
    # del stats['Speed']
    # del stats['Generation']
    # del stats['Legendary_id']

    # Normalize stats
    stats = normalize_dataset(stats)

    # Write CSV with formated dataset
    #stats.to_csv('data/formated_stats.csv')

    return stats


def load_and_format_combats():

    # Load combats dataset:
    names = ['Pokemon_A', 'Pokemon_B', 'Winner']
    combats = pd.read_csv(filepath_combats, names=names)

    # Shuffle combats
    combats = shuffle(combats)

    return combats


def format_net_input_output():
    combats = load_and_format_combats()
    stats = load_and_format_stats()

    # Total number of combats:
    total_combats = combats.count()[0]

    X = []
    Y = []
    for i in range(0, 40000):
        pok_A_id, pok_B_id, winner_id = combats[combats.index == i].values[0].tolist()
        stats_A = stats[stats.index == pok_A_id].values[0].tolist()
        stats_B = stats[stats.index == pok_B_id].values[0].tolist()
        X.append(stats_A + stats_B)
        winner = 0 if winner_id == pok_A_id else 1
        Y.append([winner])

    X_test = []
    Y_test = []
    for i in range(40000, total_combats):
        pok_A_id, pok_B_id, winner_id = combats[combats.index == i].values[0].tolist()
        stats_A = stats[stats.index == pok_A_id].values[0].tolist()
        stats_B = stats[stats.index == pok_B_id].values[0].tolist()
        X_test.append(stats_A + stats_B)
        winner = 0 if winner_id == pok_A_id else 1
        Y_test.append([winner])

    return X, Y, X_test, Y_test


# SolosisNET:
def create_model(input_dim):
    print('Creating model...')

    # Convert every char to ---> ascii code
    neuronas_capa_P =  ord('P')  # 80
    neuronas_capa_O =  ord('O')  # 79
    neuronas_capa_K =  ord('K')  # 75
    neuronas_capa_E =  ord('E')  # 69
    neuronas_capa_M =  ord('M')  # 77
    neuronas_capa_O2 = ord('O')  # 79
    neuronas_capa_N =  ord('N')  # 78
    neuronas_capa_ex = ord('!')  # 33
    neuronas_capa_end = 10

    # Drop probability
    drop_prob = 0.1

    model = Sequential()

    model.add(Dense(neuronas_capa_P, input_dim=input_dim, activation='relu'))
    #model.add(Dropout(drop_prob))
    model.add(Dense(neuronas_capa_O, activation='relu'))
    model.add(Dropout(drop_prob))
    model.add(Dense(neuronas_capa_K, activation='relu'))
    #model.add(Dropout(drop_prob))
    model.add(Dense(neuronas_capa_E, activation='relu'))
    model.add(Dropout(drop_prob))
    model.add(Dense(neuronas_capa_M, activation='relu'))
    #model.add(Dropout(drop_prob))
    model.add(Dense(neuronas_capa_O2, activation='relu'))
    model.add(Dropout(drop_prob))
    model.add(Dense(neuronas_capa_N, activation='relu'))
    #model.add(Dropout(drop_prob))
    model.add(Dense(neuronas_capa_ex, activation='relu'))
    model.add(Dropout(drop_prob))
    model.add(Dense(neuronas_capa_end, activation='relu'))
    model.add(Dropout(drop_prob))

    model.add(Dense(1, activation='sigmoid', name='prediction'))

    model.compile(loss='binary_crossentropy',
                  optimizer='nadam',
                  metrics=['accuracy'])

    print(model.summary())
    return model


# CALLBACKS:
early_stopper = EarlyStopping(monitor='val_loss', patience=100, verbose=0)
checkpointer = ModelCheckpoint(filepath="solosisnet.h5",
                               verbose=1,
                               save_best_only=True,
                               monitor='val_loss')

callbacks_list = [checkpointer, early_stopper]


# Main Program
if __name__ == '__main__':

    print('''
    ░░░█▀▀▄░░░░░░░░░░░░░▄▀▀█░░░░░░░
    ░░░█░░░░▀▄░▄▄▄▄▄░▄▀░░░█░░░░░░░░
    ░░░░▀▄░░░▀░░░░░░░▀░░░▄▀░░░░░░░░
    ░░░░░░▌░▄▄░░░░░░░▄▄░▐▀▀░░░░░░░░
    ░░░░░▐░░█▄░░░░░░░▄█░░▌▄▄▀▀▀▀█░░
    ░░░░░▌▄▄▀▀░░░▄░░░▀▀▄▄▐░░░░░░█░░
    ░░▄▀▀▐▀▀░░░▄▄▄▄▄░░░▀▀▌▄▄▄░░░█░░
    ░░█░░░▀▄░░░█░░░█░░░▄▀░░░░█▀▀▀░░
    ░░░▀▄░░▀░░░░▀▀▀░░░░▀░░░▄█▀░░░░░
    ░░░░░█░░░░░░░░░░░░░░░▄▀▄░▀▄░░░░
    ░░░░░█░░░░░░░░░░░░░▄▀█░░█░░█░░░
    ░░░░░█░░░░░░░░░░░░░░░█▄█░░▄▀░░░
    ░░░░░█░░░░░░░░░░░░░░░████▀░░░░░
    ░░░░░▀▄▄▄▀▀▄▄▄▀▀▄▄▄▄▄█▀░░░░░░░░
    ''')

    print('\n>>> SOLOSISNET! ROLL OUT!! <<<\n')

    print('Formatting Data...')
    X, Y, X_test, Y_test = format_net_input_output()

    input_dim = len(X[0])
    input_batch_size = len(X[1])

    import_model = False
    if import_model:
        print('Loading model...')
        solosisnet = load_model("solosisnet.h5")
    else:
        solosisnet = create_model(input_dim)

    print('Training Net...')

    # TRAIN MODEL V1
    solosisnet.fit(X, Y, epochs=1000, batch_size=input_batch_size,
                   validation_split=0.2, callbacks=callbacks_list,
                   verbose=1, shuffle=True)

    # TRAIN MODEL V2:
    # epochs = 1000
    # for epoch in range(epochs):
    #     solosisnet.fit(X, Y, epochs=1, batch_size=input_batch_size,
    #                    callbacks=callbacks_list, verbose=0, shuffle=True)
    #     solosisnet.load_weights("solosisnet.h5")

    # TRAIN MODEL V3:
    # epochs = 1000
    # for epoch in range(epochs):
    #     solosisnet.fit(X, Y, epochs=1, callbacks=callbacks_list, verbose=1,
    #                    shuffle=True, batch_size=input_batch_size,
    #                    validation_split=0.33)
    #     solosisnet.load_weights("solosisnet.h5")

    print('Evaluate model...')
    score = solosisnet.evaluate(X_test, Y_test)
    print('\t test -- score:%f accuracy:%f' % (score[0], score[1]))

    print('Saving model...')
    solosisnet.save("solosisnet.h5")

    print('pikaaaa')
