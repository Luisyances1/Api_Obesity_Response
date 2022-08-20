from fastapi import  FastAPI
from typing import  List
from classes import ModelInput, ModelOutput, APIModelBackEnd

#Creamos el objeto app
app = FastAPI(title="API for clasification obesity", version='1.0')

#Con el decorador, ponemos en el endpoint /predict la funcionalidad de la función predict_proba
# response_model=List[ModelOuput] es que puede responder una lista de instancias válidad de ModelOutput
# En la definición, le decimos que los Inputs son una lista de ModelInput.
# Así, la API recibe para hacer multiples predicciones
@app.post("/predict", response_model=List[ModelOutput])
async def predict_(Inputs: List[ModelInput]):
    '''Endpoint de predicion de la api'''
    #lista vacia para las respuestas
    response = []
    #iteracion por todas las entradas que brindamos
    for Input in Inputs:
        #llamamos a nuestra clase en el backend para predecir
        #ponemos Input.nombre_atributo
        Model = APIModelBackEnd(Input.Peso, Input.Altura, Input.Edad, Input.N_comidas, Input.F_actividad_fisica, Input.Consumo_agua, Input.F_consumo_verduras)
        response.append(Model.predict()[0])
        
    #Retornamos la lista con todas las predicciones hechas, si se hizo mas de una request de la api 
    return response
