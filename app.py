"""
This file only used for 🤗 spaces.
"""

import gradio as gr
import json
import torch
from torch import nn
import numpy as np

title = "# Malaysia KL House Price Prediction"

description = """
<br/>

<img src="https://static.vecteezy.com/system/resources/previews/000/344/591/non_2x/kuala-lumpur-skyline-vector.jpg" style="width:200px;height:150px;">

This machine learning model is to predict the house price 
around the rural areas in Malaysia, Kuala Lumpur. 

There are 11 types of houses in this dataset. 
Apartment, Condominuim, Serviced Residence, Bungalow,
Semi-detached House, and Terrace/Link Houses with different styles.

Note that the [dataset](https://www.kaggle.com/datasets/dragonduck/property-listings-in-kuala-lumpur) that I used for training is a bit outdated (from 2019), 
maybe you need to multiply the inflation rate to have more accurate figure.

<br/>
"""

link = """
|                                 |                                         |
| ------------------------------- | --------------------------------------- |
| 🏡📊 **KL House Price**        | <a style="display:inline-block" href='https://github.com/pehcy/Malaysia-House-Price-prediction'><img src='https://img.shields.io/github/stars/pehcy/Malaysia-House-Price-prediction?style=social' /></a>|
"""

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Feed Forward Neural Network for tabular data
class SparseTabularNN(nn.Module):
    def __init__(self, embedding_dim, n_cont, out_sz, layers, dropout_rate=0.5) -> None:
        super().__init__()
        self.embeds = nn.ModuleList([
            nn.Embedding(inp, out) for inp, out in embedding_dim
        ])
        self.emb_drop = nn.Dropout(dropout_rate)
        self.bn_cont = nn.BatchNorm1d(n_cont)

        layer_list = []
        n_emb = sum(e.embedding_dim for e in self.embeds)
        n_in = n_emb + n_cont

        for i in layers:
            layer_list.append(nn.Linear(n_in, i))
            layer_list.append(nn.ReLU(inplace=True))
            layer_list.append(nn.BatchNorm1d(i))
            layer_list.append(nn.Dropout(dropout_rate))
            n_in = i
        
        layer_list.append(nn.Linear(layers[-1], out_sz))
        
        self.layers = nn.Sequential(*layer_list)
        
    def forward(self, x_cat, x_cont):
        embeddings = [e(torch.clamp(x_cat[:,i], 0, e.num_embeddings - 1)) for i, e in enumerate(self.embeds)]
        x = torch.cat(embeddings, 1)
        x = self.emb_drop(x)

        x2 = self.bn_cont(x_cont)
        x = torch.cat([x, x2], 1)
        x = self.layers(x)

        return x

emb_size = [(77, 39), (9, 5), (6, 3), (9, 5), (85, 43), (3, 2)]
model = SparseTabularNN(emb_size, 1, 1, [240, 70], dropout_rate=0.40)
model.load_state_dict(torch.load('./HouseWeights.pt', map_location=torch.device(DEVICE)))
model.to(DEVICE)

locations_dict = None
property_dict = None

with open('locations.txt', 'r') as f:
   locations_dict = json.loads(f.read())

with open('property-type.txt', 'r') as f:
    property_dict = json.loads(f.read())

def predict_price(
    rooms, 
    bathrooms, 
    car_parks, 
    size, 
    furnished, 
    location,
    house_type,
    extra_info,
    model=model, 
    locations_dict=locations_dict,
    property_dict=property_dict
):
    model.eval()
    location_index = locations_dict[location]

    furnishing_dict = {
        'Unfurnished': 0,
        'Partly Furnished': 1,
        'Fully Furnished': 2,
    }

    furnished_status = furnishing_dict[furnished]
    property_type = f"{house_type} ({extra_info})"


    if extra_info != "Normal" and property_type in property_dict:
        property_index = property_dict[property_type]
    else:
        property_index = property_dict[house_type]

    X_cat = torch.tensor(np.array([[location_index,  rooms,  bathrooms, car_parks, property_index, furnished_status]]), dtype=torch.int64).to(DEVICE)
    X_cont = torch.tensor(np.array([[size]]), dtype=torch.float32).reshape(-1, 1).to(DEVICE)

    y_pred = 0

    with torch.no_grad():
        y_pred = model(X_cat, X_cont)
        y_pred = y_pred.item() * 1e6
        print(y_pred)
        return f"""The expected house price is: MYR {y_pred:.2f}"""

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            gr.Markdown(title)
            gr.Markdown(description)
        with gr.Column():
            gr.Markdown(link)

    with gr.Row():
        with gr.Column():
            rooms_gr = gr.Slider(0, 20, value=3, step=1.0, label="Number of Rooms", info="Choose between 0 and 20")
            bathrooms_gr = gr.Slider(0, 20, value=1, step=1.0, label="Number of Bathrooms", info="Choose between 0 and 20")
            car_parks_gr = gr.Slider(0, 20, value=1, step=1.0, label="Number of Car Parks", info="Choose between 0 and 20")
            size_gr = gr.Slider(0, 10, value=1.5, step=0.01, label="Size (in 100,000 feet²)", info="Choose between 0 and 10")
            furnished_gr = gr.Radio(["Unfurnished", "Partly Furnished", "Fully Furnished"], label="Furnished status", info="Is this house furnished?")
            locations_gr = gr.Dropdown(
                locations_dict.keys(), label="Location in KL", info="So far this is all locations we got!")
            
            house_type_gr = gr.Dropdown([
                "Apartment",
                "Condominuim", 
                "Serviced Residence", 
                "Bungalow",
                "Semi-detached House",
                "1-sty Terrace/Link House",
                "2-sty Terrace/Link House",
                "2.5-sty Terrace/Link House",
                "3-sty Terrace/Link House",
                "3.5-sty Terrace/Link House",
                "4-sty Terrace/Link House"
            ], label="House Type", info="Pick one of the house type")

            extra_rd = gr.Radio([
                "Normal",
                "Corner",
                "Intermediate",
                "EndLot"
            ], label="Extra information for house property", info="Choose one of the properties")

            submit_btn = gr.Button("Submit", elem_id="send_tbn", visible=True)
        
        with gr.Column():
            out = gr.Textbox(label="Prediction result")
        
    submit_btn.click(predict_price, [
        rooms_gr, 
        bathrooms_gr,
        car_parks_gr, 
        size_gr, 
        furnished_gr, 
        locations_gr,
        house_type_gr,
        extra_rd
    ], 
    outputs=out)

if __name__ == "__main__":
    # start gradio app
    demo.queue()
    demo.launch(debug=True, show_api=False)