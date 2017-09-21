# SolosisNET Neural Network Model

###Install:
```
pip install -r requirements.txt
```

### Run Model Training:
For running the model, open a terminal and type:
```sh
$ cd solosisnet-model
$ python3 solosisnet_training.py
```

### Make Combats Predictions:
Make combat predictions for the final delivery.

File with combats to predict: 'data/combats.csv'
Output file:                  'data/predicted_combats.csv'

```sh
$ cd solosisnet-model
$ python3 solosisnet_training.py
```

### Make only ONE combat prediction:
Make one combat prediction.

[...WORK IN PROGRESS...]

```sh
$ cd solosisnet-model
$ python3 solosisnet_one_prediction.py
```

###Start the API:
```
python ./run.py
```

### Example POST API
curl -H "Content-Type: application/json" -X POST -d '{"pokA":100,"pokB":345}' http://localhost:5000/
