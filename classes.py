from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field, ValidationError
from typing import Literal
import joblib
import pandas as pd
import datetime as dt
import os
import numpy as np
from fastapi import HTTPException


class ModelInput(PydanticBaseModel):
    '''
    Clase que define las entradas del modelo
    '''
        
    Edad: int = Field(description = 'Edad de la persona', ge = 10 , le= 90)    
    Altura: float = Field(description = 'Altura de la persona', ge = 1 , le= 2)
    Peso: float = Field(description = 'Peso de la persona en KG', ge=10, le=100)
    F_consumo_verduras: float = Field(description = 'Consumo de verduras por dia', ge = 1, le= 10)
    N_comidas: float = Field(description = 'Cantidad de comidas en el dia', ge = 0, le= 10)
    Consumo_agua: float = Field(description = 'Litros de agua consumidos por dia', ge = 1, le= 10)
    F_actividad_fisica: float = Field(description = 'Frecuencia de actividad fisica', ge = 0, le= 10)
    
    
    
    
    class Config:
        schema_extra = {
            "Example": {
                'Peso': 40,
                'Altura': 1.50,
                'Edad': 35,
                'N_comidas': 3,
                'F_actividad_fisica': 1.5,
                'Consumo_agua': 3,
                'F_consumo_verduras': 2.5
            }
        }
class ModelOutput(PydanticBaseModel):
    '''
    Clase que define la salida del modelo
    '''
    
    Tipo_obesidad: Literal['Obesidad_Tipo_I','Obesidad_Tipo_II','Obesidad_Tipo_III','Peso_normal','Sobrepeso_Nivel_I','Sobrepeso_Nivel_II','Peso_insuficiente']
    
    class Config:
        schema_extra = {
            "Example": {
                'Tipo_obesidad': 'Obesidad_Tipo_I'
            }
        }
        
class APIModelBackEnd():
    '''
    Esta clase maneja el back end de nuestro modelo de Machine Learning para la API en FastAPI
    '''
    
    def __init__(self,Edad,Altura,Peso,F_consumo_verduras,N_comidas,Consumo_agua,F_actividad_fisica):
        
        '''
        Este método se usa al instanciar las clases

        Aquí, hacemos que pida los mismos parámetros que tenemos en ModelInput.

        Para más información del __init__ method, pueden leer en línea en sitios cómo 

        https://www.udacity.com/blog/2021/11/__init__-in-python-an-overview.html

        '''
        
        
        self.Edad = Edad
        self.Altura = Altura
        self.Peso = Peso
        self.F_consumo_verduras = F_consumo_verduras
        self.N_comidas = N_comidas
        self.Consumo_agua = Consumo_agua
        self.F_actividad_fisica = F_actividad_fisica
        
        
        
    def _load_model(self, model_name:str = "ModelRF.pkl"):
        self.model = joblib.load(model_name)
        
    def _prepare_data(self):
        '''
        Clase de preparar lo datos.
        Este método convierte las entradas en los datos que tenían en X_train y X_test.

        Miren el orden de las columnas de los datos antes de su modelo.
        Tienen que recrear ese orden, en un dataframe de una fila.

        '''
        
        Edad = self.Edad
        Altura = self.Altura
        Peso = self.Peso
        F_consumo_verduras = self.F_consumo_verduras
        N_comidas = self.N_comidas
        Consumo_agua = self.Consumo_agua
        F_actividad_fisica = self.F_actividad_fisica
        
        df2 = pd.DataFrame(
            columns=[
                "Edad",
                "Altura",
                "Peso",
                "F_consumo_verduras",
                "N_comidas",
                "Consumo_agua",
                "F_actividad_fisica"
            ],
            data=[[Edad,Altura,Peso,F_consumo_verduras,N_comidas,Consumo_agua,F_actividad_fisica]],   
             )
        df2[
            [
                x
                for x in df2.columns
                if((str(F_actividad_fisica)in x) and (x.starswith("F_actividad_fisica")))
            ]
        ] = 1
            
        return df2
    
    def predict(self, y_name: str = 'Tipo_obesidad'):
        '''
        Clase para predecir.
        Carga el modelo, prepara los datos y predice.

        prediction = pd.DataFrame(self.model.predict(X)).rename(columns={0:y_name})

        '''
        
        self._load_model()
        X = self._prepare_data()
        prediction = pd.DataFrame(self.model.predict(X)).rename(columns={0: y_name})
        return prediction.to_dict(orient='records')