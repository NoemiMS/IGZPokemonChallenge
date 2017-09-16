import pandas as pd
from keras.models import load_model

# FILEPATHS
filepath_stats = "data/pokemon.csv"
filepath_combats = "data/combats.csv"
filepath_prediction = "data/predicted_combats.csv"


def normalize_dataset(dataset):
    result = dataset.copy()
    for feature_name in dataset.columns:
        max_value = dataset[feature_name].max()
        min_value = dataset[feature_name].min()
        result[feature_name] = (dataset[feature_name] - min_value) / (max_value - min_value)

    return result


def load_combats_dataset():
    names = ['Pokemon_A', 'Pokemon_B', 'Winner']
    combats = pd.read_csv(filepath_combats, names=names)

    return combats


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

    # Normalize stats
    stats = normalize_dataset(stats)

    return stats


def format_net_input(combats):
    stats = load_and_format_stats()
    # stats = pd.read_csv('data/formated_stats.csv', index_col='#')

    # Total number of combats:
    total_combats = combats.count()[0]

    X = []
    for i in range(0, total_combats):
        pok_A_id, pok_B_id, _ = combats[combats.index == i].values[0].tolist()
        stats_A = stats[stats.index == pok_A_id].values[0].tolist()
        stats_B = stats[stats.index == pok_B_id].values[0].tolist()
        X.append(stats_A + stats_B)

    return X


# Main Program
if __name__ == '__main__':

    print('SOLOSISNET! ROLL OUT!')

    print('Loading model...')
    solosisnet = load_model("solosisnet.h5")

    print('Loading combats to predict...')
    combats = load_combats_dataset()
    net_input = format_net_input(combats)

    print('Make a prediction...')
    net_output = solosisnet.predict(net_input)

    print('Format the predictions...')
    winner_pokemons = []
    for prediction_index, prediction in enumerate(net_output):
        if prediction == [0]:
            winner = combats['Pokemon_A'][combats.index == prediction_index].values[0]
        else:
            winner = combats['Pokemon_B'][combats.index == prediction_index].values[0]
        winner_pokemons.append(winner)

    print('Write the prediction in a new CSV...')
    pd.DataFrame(winner_pokemons).to_csv(filepath_prediction)
