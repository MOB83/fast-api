# Library imports
import uvicorn
from fastapi import FastAPI
from Model import IrisModel, IrisSpecies

# Create the client
app = FastAPI()
model = IrisModel()


# Create the routes
@app.post('/predict')
def predict_species(iris: IrisSpecies):
    data = iris.dict()
    prediction, probability = model.predict_species(
        data['sepal_length'], data['sepal_width'], data['petal_length'], data['petal_width']
    )

    return {
        'preciction': prediction,
        'probability': probability
    }

# Run the API
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
