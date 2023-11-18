"""
This file only used for 🤗 spaces.
"""

import gradio as gr
import json
import torch
from torch import nn
import numpy as np
import locale
locale.setlocale(locale.LC_ALL, 'en_US.utf8')

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
model.load_state_dict(torch.load('./HouseWeights.pt'))
model.cuda()


title = "Malaysia KL House Price Prediction"

description = """

<br/>

<img src=""/>

<br/>

"""

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

    X_cat = torch.tensor(np.array([[location_index,  rooms,  bathrooms, car_parks, property_index, furnished_status]]), dtype=torch.int64).to('cuda')
    X_cont = torch.tensor(np.array([[size]]), dtype=torch.float32).reshape(-1, 1).to('cuda')

    y_pred = 0

    with torch.no_grad():
        y_pred = model(X_cat, X_cont)
        y_pred = y_pred.item() * 1e6
        print(y_pred)
        return f"""The expected house price is: MYR {locale.currency(y_pred, grouping=True)[1:]}"""

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            rooms_gr = gr.Slider(0, 20, value=4, step=1.0, label="Number of Rooms", info="Choose between 0 and 20")
            bathrooms_gr = gr.Slider(0, 20, value=4, step=1.0, label="Number of Bathrooms", info="Choose between 0 and 20")
            car_parks_gr = gr.Slider(0, 20, value=4, step=1.0, label="Number of Car Parks", info="Choose between 0 and 20")
            size_gr = gr.Slider(0, 10, value=1.5, step=0.01, label="Size (in 100,000 meter²)", info="Choose between 0 and 10")
            furnished_gr = gr.Radio(["Unfurnished", "Partly Furnished", "Fully Furnished"], label="Furnished status", info="Is this house furnished?")
                #gr.Slider(0, 20, value=4, step=1.0, label="Number of Bathrooms", info="Choose between 2 and 20"),
            locations_gr = gr.Dropdown(
                locations_dict.keys(), label="Location in KL", info="So far this is all locations we got!")
        
        with gr.Column():
            out = gr.Textbox()
    
    with gr.Row():
        with gr.Column():
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
            ], label="Furnished status", info="House Type")

            extra_rd = gr.Radio([
                "Normal",
                "Corner",
                "Intermediate",
                "EndLot"
            ], label="Extra information for house property", info="Extra Information")
        
        with gr.Column():
            pass
    
    with gr.Row():
        with gr.Column():
            submit_btn = gr.Button("Send", elem_id="send_tbn", visible=True)
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
        
        with gr.Column():
            pass

if __name__ == "__main__":
    # start gradio app
    demo.queue()
    demo.launch(debug=True, show_api=False)