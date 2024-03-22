import numpy as np
import pandas as pd
import traceback
import os

from config import config
 
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from DownloadData import download_data
from SP500Data import  getSP500
from Normalizer import Normalizer
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from PrepareData import  prepare_data_x, prepare_data_y
from TimeSeriesDataset import  TimeSeriesDataset
from ModelTraining import run_epoch
from LSTMModel import LSTMModel

list_of_sp500 = getSP500()


for company in list_of_sp500:
    try:
        config['alpha_vantage']["symbol"] = company
        data_date, data_close_price, num_data_points, display_date_range = download_data(config)

        scaler = Normalizer()
        normalized_data_close_price = scaler.fit_transform(data_close_price)
        
        data_x, data_x_unseen = prepare_data_x(normalized_data_close_price, window_size=config["data"]["window_size"])
        data_y = prepare_data_y(normalized_data_close_price, window_size=config["data"]["window_size"])
        
        # split dataset

        split_index = int(data_y.shape[0]*config["data"]["train_split_size"])
        data_x_train = data_x[:split_index]
        data_x_val = data_x[split_index:]
        data_y_train = data_y[:split_index]
        data_y_val = data_y[split_index:]


        dataset_train = TimeSeriesDataset(data_x_train, data_y_train)
        dataset_val = TimeSeriesDataset(data_x_val, data_y_val)

        print("Train data shape", dataset_train.x.shape, dataset_train.y.shape)
        print("Validation data shape", dataset_val.x.shape, dataset_val.y.shape)

        train_dataloader = DataLoader(dataset_train, batch_size=config["training"]["batch_size"], shuffle=True)
        val_dataloader = DataLoader(dataset_val, batch_size=config["training"]["batch_size"], shuffle=True)

        model = LSTMModel(input_size=config["model"]["input_size"], hidden_layer_size=config["model"]["lstm_size"], num_layers=config["model"]["num_lstm_layers"], output_size=1, dropout=config["model"]["dropout"])
        model = model.to(config["training"]["device"])

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"], betas=(0.9, 0.98), eps=1e-9)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config["training"]["scheduler_step_size"], gamma=0.1)

        for epoch in range(config["training"]["num_epoch"]):
            loss_train, lr_train = run_epoch(config,val_dataloader,model,optimizer,scheduler,criterion, is_training=True)
            loss_val, lr_val = run_epoch(config,val_dataloader,model,optimizer,scheduler,criterion)
            scheduler.step()
    
        print('Epoch[{}/{}] | loss train:{:.6f}, test:{:.6f} | lr:{:.6f}')

        train_dataloader = DataLoader(dataset_train, batch_size=config["training"]["batch_size"], shuffle=False)
        val_dataloader = DataLoader(dataset_val, batch_size=config["training"]["batch_size"], shuffle=False)

        model.eval()

        # predict on the training data, to see how well the model managed to learn and memorize

        predicted_train = np.array([])

        for idx, (x, y) in enumerate(train_dataloader):
            x = x.to(config["training"]["device"])
            out = model(x)
            out = out.cpu().detach().numpy()
            predicted_train = np.concatenate((predicted_train, out))

        # predict on the validation data, to see how the model does

        predicted_val = np.array([])

        for idx, (x, y) in enumerate(val_dataloader):
            x = x.to(config["training"]["device"])
            out = model(x)
            out = out.cpu().detach().numpy()
            predicted_val = np.concatenate((predicted_val, out))

        # predict the closing price of the next trading day

        model.eval()

        x = torch.tensor(data_x_unseen).float().to(config["training"]["device"]).unsqueeze(0).unsqueeze(2) # this is the data type and shape required, [batch, sequence, feature]
        prediction = model(x)
        prediction = prediction.cpu().detach().numpy()


        # prepare plots

        plot_range = 10
        to_plot_data_y_val = np.zeros(plot_range)
        to_plot_data_y_val_pred = np.zeros(plot_range)
        to_plot_data_y_test_pred = np.zeros(plot_range)

        to_plot_data_y_val[:plot_range-1] = scaler.inverse_transform(data_y_val)[-plot_range+1:]
        to_plot_data_y_val_pred[:plot_range-1] = scaler.inverse_transform(predicted_val)[-plot_range+1:]

        to_plot_data_y_test_pred[plot_range-1] = scaler.inverse_transform(prediction)

        to_plot_data_y_val = np.where(to_plot_data_y_val == 0, None, to_plot_data_y_val)
        to_plot_data_y_val_pred = np.where(to_plot_data_y_val_pred == 0, None, to_plot_data_y_val_pred)
        to_plot_data_y_test_pred = np.where(to_plot_data_y_test_pred == 0, None, to_plot_data_y_test_pred)

        # plot

        plot_date_test = data_date[-plot_range+1:]
        plot_date_test.append("tomorrow")

        fig = figure(figsize=(25, 5), dpi=80)
        fig.patch.set_facecolor((1.0, 1.0, 1.0))
        plt.plot(plot_date_test, to_plot_data_y_val, label="Actual prices", marker=".", markersize=10, color=config["plots"]["color_actual"])
        plt.plot(plot_date_test, to_plot_data_y_val_pred, label="Past predicted prices", marker=".", markersize=10, color=config["plots"]["color_pred_val"])
        plt.plot(plot_date_test, to_plot_data_y_test_pred, label="Predicted price for next day", marker=".", markersize=20, color=config["plots"]["color_pred_test"])
        plt.title("Predicted close price of the next trading day")
        plt.grid(which='major', axis='y', linestyle='--')
        plt.legend()

        print("Predicted close price of the next trading day:", round(to_plot_data_y_test_pred[plot_range-1], 2))

        
        save_dir = './predictions'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_path = os.path.join(save_dir, f"{company}.png")
        plt.savefig(save_path)
        plt.close()  # Close the plot to free up memory
        ## plots

    except Exception as e:
        print(f"Could not retrieve data for {company}- ERROR -- {e}")
        traceback.print_exc()
        continue