# Configuration settings
num_epochs = 100
batch_size = 16 # Batch size for DataLoader

# weights for the custom loss function
alpha = 0.95

# Upsample methods
upsample_methods = {
    "1": "pixelshuffle",
    "2": "pixelshuffledirect",
    "3": "nearest+conv",
    "4": ""
}

# Current upsample method
upsampler = upsample_methods["2"]

# Define the directory path
#dir_path = "/path/to/your/directory"
dir_path = ""
dir_data   = "/home/mws/BrainUltrasoundSimulation"
model_path = f"model"
