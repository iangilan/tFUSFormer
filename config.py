from models import FSRCNN_1ch, SESRResNet_1ch, SRGAN_1ch, tFUSFormer_1ch, tFUSFormer_5ch

# Configuration settings
num_epochs = 100
batch_size = 16 # Batch size for DataLoader

# weights for the custom loss function
alpha = 0.999

# Configuration variable for the test dataset mode
# seen:    foreseen test dataset
# unseen1: unseen test dataset 1
# unseen2: unseen test dataset 2
test_data_mode = 'seen'  # 'seen', 'unseen1', or 'unseen2' for the test dataset

# Upsample methods
upsample_methods = {
    "1": "pixelshuffle",
    "2": "pixelshuffledirect",
    "3": "nearest+conv",
    "4": ""
}

# Current model/upsample method
selected_model_key = 'tFUSFormer_5ch'  # Choose a model
upsampler = upsample_methods["2"]

# Configuration for available models
models_config = {
    'FSRCNN_1ch': {
        'class': FSRCNN_1ch,
        'params': {}
    },
    'SESRResNet_1ch': {
        'class': SESRResNet_1ch,
        'params': {}
    },  
    'SRGAN_1ch': {
        'class': SRGAN_1ch,
        'params': {}
    },  
    'tFUSFormer_1ch': {
        'class': tFUSFormer_1ch,
        'params': {'upsampler': upsampler}
    },            
    'tFUSFormer_5ch': {
        'class': tFUSFormer_5ch,
        'params': {'upsampler': upsampler}
    }
}

# Define the directory path
#dir_data = "/path/to/your/data_directory"
dir_data   = "/media/mws/Data/tFUSFormer_data"
#dir_data   = "/home/mws/tFUSFormer_data"
model_path = f"model"
