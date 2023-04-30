# CMPE-256 Food Recommendation System


The data needed for the Python Notebooks can be found at https://cseweb.ucsd.edu/~jmcauley/datasets.html#foodcom
</br>
Since dataset is huge, it is not uploaded here. Need to be kept in a
folder data for the code to run.
</br>
Given a person’s preferences in past recipes, can we predict other new recipes they might enjoy?

# Running the APIs

## 1. Install the required dependencies
```
pip install -r requirements.txt
```

## 2. Run the API
```
cd backend
flask --app main run --debug
```

### API Endpoints
```
http://127.0.0.1:5000/model/hybrid?recipe_id=192839
http://127.0.0.1:5000/model/nmf?recipe_id=192839
```