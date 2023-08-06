from copyreg import pickle
import matplotlib.pyplot as plt
import argparse
import json
import os
import pickle
from PIL import Image


argparser = argparse.ArgumentParser(
    description='Plot training loss and validation loss history.')

argparser.add_argument(
    '-c',
    '--conf',
    default='/src/datafiles/yolo/configs/benchmark_configbees_detection.json',
    help='Path to config file.')


def _plot_history_(args):
    config_path = args.conf

    # Load config file as a dict
    with open(config_path) as config_buffer:    
        config = json.loads(config_buffer.read())
    
    # Get the pickle history file based on config file
    root, ext = os.path.splitext(config['train']['saved_weights_name'])
    saved_pickle_path = config['data']['saved_pickles_path']
    saved_weights_file_name=root.split(os.sep)[-1]
    #pickle_path = f'{saved_pickle_path}/history/history_{root}_bestLoss{ext}.p'
    #pickle_path=f'{saved_pickle_path}/benchmark_weights/best_model_bestLoss{ext}.p'
    pickle_path = f'{saved_pickle_path}history/{saved_weights_file_name}_bestLoss{ext}.p'

    # Load history pickle
    with open(pickle_path, 'rb') as pickle_buffer:
        history = pickle.load(pickle_buffer)
        
    # Extract losses

    loss = history['loss']
    val_loss = history['val_loss']

    for i in range(len(loss)):
        if loss[i] > 100:
            loss[i] = 100
        if val_loss[i] > 100:
            val_loss[i] = 100

    steps = [i for i in range(len(loss))]

    # Plot curves
    plt.plot(steps, loss, label='Training loss')
    plt.plot(steps, val_loss, label='Validation loss')
    
    # Modify figure params
    _, xmax, _, _ = plt.axis()
    xmin = int(0.1 * xmax)
    plt.axis((xmin, xmax, min(loss + val_loss), val_loss[xmin]))
    plt.legend()
    plt.title('Training from {}'.format(pickle_path.split(os.sep)[-1]))
    
    # Save figure
    plot_path=os.path.join('/src/datafiles/yolo/outputs/train/{}.jpg'.format(pickle_path.split(os.sep)[-1].split('.')[0]))
    plt.savefig(plot_path)

    # Displays figure
    im=Image.open(plot_path)  
    im.show()


if __name__ == '__main__':
    _args = argparser.parse_args()
    _plot_history_(_args)
