import pandas as pd
import numpy as np

# Function to compute output shape after a convolution layer
def conv_output_shape(input_size, kernel_size, stride, padding):
    return np.floor((input_size + 2 * padding - kernel_size) / stride + 1).astype(int)

# CNN layer details (input size = 32x32x3)
input_size = 32 
input_channels = 3  
results = []

# (Layer, filter size, kernel, stride, padding)
layers = [
    ("Conv2D", 32, 3, 2, "same"),
    ("BatchNorm", 0, 0, 0, 0),
    
    ("Conv2D", 64, 3, 2, "same"),
    ("BatchNorm", 0, 0, 0, 0),
    
    ("Conv2D", 128, 3, 2, "same"),
    ("BatchNorm", 0, 0, 0, 0),
    
    ("Conv2D", 128, 3, 1, "same"),
    ("BatchNorm", 0, 0, 0, 0),
    ("Conv2D", 128, 3, 1, "same"),
    ("BatchNorm", 0, 0, 0, 0),
    ("Conv2D", 128, 3, 1, "same"),
    ("BatchNorm", 0, 0, 0, 0),
    ("Conv2D", 128, 3, 1, "same"),
    ("BatchNorm", 0, 0, 0, 0),
    
    ("MaxPool", 4, 4, 0, 0),
    ("Flatten", 0, 0, 0, 0),
    ("Dense", 128, 0, 0, 0),
    ("BatchNorm", 0, 0, 0, 0),
    ("Dense", 10, 0, 0, 0),
]

for layer in layers:
    layer_type, filters, kernel_size, stride, padding = layer

    if layer_type == "Conv2D":
        # Compute output size
        output_size = conv_output_shape(
            input_size, kernel_size, stride, 1 if padding == "same" else 0
        )
        num_params = (kernel_size * kernel_size * input_channels * filters) + filters
        macs = num_params * (output_size**2)

        # Update input for next layer
        input_size = output_size
        input_channels = filters

    elif layer_type == "MaxPool":
        output_size = input_size // kernel_size  # Corrected pooling size calculation
        num_params = 0  
        macs = 0

    elif layer_type == "Flatten":
        num_params = 0
        macs = 0
        output_size = input_size * input_size * input_channels  # Flattened feature map size

    elif layer_type == "Dense":
        num_params = input_size * filters + filters  # Fully connected layer parameter count
        macs = num_params
        output_size = filters
        input_size = filters  # Next layer input

    else:  # BatchNorm
        num_params = 4 * input_channels  # Corrected batch norm parameters
        macs = 0
        output_size = input_size

    results.append([layer_type, filters, output_size, num_params, macs])

# Convert results to DataFrame and save
df = pd.DataFrame(
    results, columns=["Layer", "Filters", "Output Size", "Num Params", "MACs"]
)
df.to_csv("cnn_analysis.csv", index=False)
print(df)
