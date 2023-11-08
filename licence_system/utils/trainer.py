"""
This file defines the functions for training the NN

Authors: Erencan Pelin, Daniel Angeloni, Ben Carroll, Declan Seeto
License: MIT License
Version: 1.0.0
"""
import os
import time

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from licence_system.utils.data_loader import show_imgs
from licence_system.utils.logger import logger
from licence_system.utils.model_class import LPLocalNet, LPR_Training_Dataset_Processed
from licence_system.config import ACCEPTABLE_DISTANCE
from jupyterplot import ProgressPlot


def train(training_dataset: LPR_Training_Dataset_Processed) -> int:
    """Trainer main function

    Args:
        training_dataset (LPR_Training_Dataset_Processed): training dataset object

    Returns:
        int: final epoch value
    """
    # BATCH_SIZE memory requirements:
    #   - 250 approx eq. 6.6 GB (fine on secondary 8GB VRAM GPU)
    #   - 200 approx eq. 5.5 GB
    #   - 100 approx eq. 2.6 GB
    #   - 50  approx eq. 1.3 GB

    # LOSS_ABANDON:
    #   -0.2 , abandons training when loss change % average is higher than this
    #    0   , will disable

    # Constants
    BATCH_SIZE = 300
    EPOCHS = 700
    LEARNING_RATE = 0.001
    PRELOAD_ALL_DATA_TO_GPU = False
    LOSS_ABANDON = 0

    # Helper function to convert seconds to hours, minutes, seconds format
    def seconds_to_hms(seconds):
        h, remainder = divmod(seconds, 3600)
        m, s = divmod(remainder, 60)
        return f"{int(h):02}:{int(m):02}:{int(s):02}"

    # Set device for training
    torch.cuda.empty_cache()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize neural network and transfer to device
    training_dataset.neural_network = LPLocalNet().to(device)
    net = training_dataset.neural_network
    logger.info("Running %s on %s", net.__class__.__name__, device)

    # Set optimizer and loss function
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    loss_function = nn.MSELoss()

    # Preloading data to GPU if required
    if PRELOAD_ALL_DATA_TO_GPU:
        train_X_gpu = training_dataset.train_X.to(device)
        train_Y_gpu = training_dataset.train_Y.to(device)
        dataset = TensorDataset(train_X_gpu, train_Y_gpu)
    else:
        dataset = TensorDataset(training_dataset.train_X, training_dataset.train_Y)

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Training preparation
    net.train()
    # TO-DO: is this to keep or nah cus it uses from jupyterplot import ProgressPlot
    pp = ProgressPlot(
        plot_names=["Training Loss over Epochs"],
        line_names=["loss"],
        x_label="Epochs",
    )
    avg_epoch_duration = None
    start_loss_plot = 2
    last_loss = []
    loss_simple = 1
    loss_pc_history = []
    cumulative_loss = torch.zeros(1, device=device)
    batch_count = 0
    final_epoch = 0

    logger.info(
        "The above loss plot will start from Epoch #%d onwards, to enhance readability.",
        start_loss_plot,
    )
    logger.info("Waiting 2s before starting training...")
    time.sleep(2)
    start_time = time.time()

    # Training loop
    for epoch in range(EPOCHS):
        epoch_start_time = time.time()

        for batch_X, batch_y in loader:  # tqdm(loader, unit="batch"):
            if PRELOAD_ALL_DATA_TO_GPU:
                batch_X = batch_X.view(-1, 1, 416, 416)
            else:
                batch_X = batch_X.to(device).view(-1, 1, 416, 416)
                batch_y = batch_y.to(device)

            # This removes bad data, but should probably try fix it instead!
            if batch_y.shape[0] == 0:
                continue

            optimizer.zero_grad(set_to_none=True)
            outputs = net(batch_X)
            loss = loss_function(outputs, batch_y)
            loss.backward()
            optimizer.step()

            cumulative_loss += loss
            batch_count += 1

        loss_simple = (cumulative_loss / batch_count).item()
        cumulative_loss = torch.zeros(1, device=device)
        batch_count = 0

        if len(loss_pc_history) > start_loss_plot:
            pp.update(loss_simple)

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        avg_epoch_duration = (
            epoch_duration if epoch == 0 else (avg_epoch_duration + epoch_duration) / 2
        )

        if len(last_loss) > 0:
            loss_diff = -(100 - (loss_simple / last_loss[-1] * 100))
        else:
            loss_diff = 0

        last_loss.append(loss_simple)
        loss_pc_history.append(loss_diff)
        avg_loss_trend = (
            sum(loss_pc_history[-15:]) / 15
            if len(loss_pc_history) >= 15
            else sum(loss_pc_history) / len(loss_pc_history)
        )
        completion_time = seconds_to_hms((EPOCHS - epoch - 1) * avg_epoch_duration)

        logger.info(
            "Epoch #%s - Loss: %.3f - Loss Diff: %.3f%% - Loss Trend: %.3f - Complete in: %ss",
            str(epoch).ljust(3),
            loss_simple,
            loss_diff,
            avg_loss_trend,
            completion_time,
        )

        if (
            LOSS_ABANDON != 0
            and avg_loss_trend > LOSS_ABANDON
            and not avg_loss_trend == 0
        ):
            logger.info("--- LOSS TREND TOO HIGH ---")
            logger.info("Abandoning training at Epoch #%s", epoch)
            break

        final_epoch = epoch

    # Training completion
    end_time = time.time()
    total_duration = end_time - start_time
    logger.info("Total training time: %02f seconds", float(total_duration))
    pp.finalize()
    return final_epoch


def get_accuracy(training_dataset: LPR_Training_Dataset_Processed) -> float:
    """Get the accuracy of the model

    Args:
        training_dataset (LPR_Training_Dataset_Processed): training dataset object

    Returns:
        float: accuracy
    """
    correct = 0
    total = 0

    correct_data = []
    og_correctdata = []

    net = training_dataset.neural_network
    net.eval()

    def close_enough(num1, num2):
        a = num1
        b = num2
        return abs(a - b) < ACCEPTABLE_DISTANCE

    demo_arr = []

    # Set device
    torch.cuda.empty_cache()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for i in tqdm(range(len(training_dataset.test_X))):
            # real_bbox = torch.argmax(training_dataset.test_Y[i])
            real_bbox = training_dataset.test_Y[i]
            net_out = net(training_dataset.test_X[i].to(device).view(-1, 1, 416, 416))[
                0
            ]
            # logger.info(net_out)
            # predicted_bbox = torch.argmax(net_out)
            predicted_bbox = net_out
            # logger.info(predicted_bbox[0], real_bbox[0])
            # logger.info("Comparing", predicted_bbox, "with", real_bbox, "- Result: ", end="")

            if (
                close_enough(predicted_bbox[0], real_bbox[0])
                and close_enough(predicted_bbox[1], real_bbox[1])
                and close_enough(predicted_bbox[2], real_bbox[2])
                and close_enough(predicted_bbox[3], real_bbox[3])
            ):
                correct += 1
                if len(demo_arr) < 5:
                    demo_arr.append(
                        [
                            f"Image #{i}",
                            training_dataset.test_X[i].view(-1, 1, 416, 416),
                            real_bbox,
                            predicted_bbox,
                        ]
                    )
                    logger.info(
                        training_dataset.test_Y[i].shape,
                        training_dataset.test_X[i].shape,
                    )

                correct_data.append(
                    [np.asarray(training_dataset.test_Y[i].cpu()), predicted_bbox.cpu()]
                )
                og_correctdata.append(
                    [np.asarray(training_dataset.test_Y[i].cpu()), real_bbox.cpu()]
                )
                # logger.info("Success!")
            # else:
            # logger.info("Fail.")
            total += 1
            # logger.info(real_bbox, net_out)

    accuracy = round((correct / total) * 100, 3)
    logger.info("Accuracy: %s %%", accuracy)
    show_imgs(demo_arr)

    torch.cuda.empty_cache()

    return accuracy


def save_model(
    training_dataset: LPR_Training_Dataset_Processed,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    accuracy: float,
):
    """
    Save the PyTorch model with a filename constructed from training parameters.

    Args:
        training_dataset (LPR_Training_Dataset_Processed): The model object containing the model to save
        final_epoch (int):  Number of epochs for training
        batch_size (int): Batch size used during training
        learning_rate (float): Learning rate used during training
        accuracy (float): Final accuracy of the model
    Returns:
        str: The path to the saved model
    """
    # TO-DO: add this into a try block
    model = training_dataset.neural_network
    network_name = model.__class__.__name__

    # Construct the filename
    filename = f"{network_name}_B{batch_size}_E{epochs}_LR{learning_rate:.4f}_Acc{accuracy:.2f}.pth"

    # Save the model
    torch.save(model.state_dict(), os.path.join("models/checkpoints/", filename))

    return filename
