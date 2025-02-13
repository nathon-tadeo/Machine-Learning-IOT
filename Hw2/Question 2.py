import pandas as pd
import numpy as np

# Function to compute output shape after a convolution layer
def conv_output_shape(input_size, kernel_size, stride, padding):
    if padding == "same":
        return np.ceil(input_size / stride).astype(int)
    else:
        return np.floor((input_size + 2 * padding - kernel_size) / stride + 1).astype(int)

# CNN layer details (input size = 32x32x3)
input_size = 32 
input_channels = 3  
results = []

# (Layer, filter size, kernel, stride, padding)
layers1 = [
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

# (Layer, filter size, kernel, stride, padding)
layers2 = [
    ("Conv2D", 32, 3, 2, "same"),
    ("BatchNorm", 0, 0, 0, 0),
    
    ("SeparableConv2D", 64, 3, 2, "same"),
    ("BatchNorm", 0, 0, 0, 0),
    
    ("SeparableConv2D", 128, 3, 2, "same"),
    ("BatchNorm", 0, 0, 0, 0),
    
    ("SeparableConv2D", 128, 3, 1, "same"),
    ("BatchNorm", 0, 0, 0, 0),
    ("SeparableConv2D", 128, 3, 1, "same"),
    ("BatchNorm", 0, 0, 0, 0),
    ("SeparableConv2D", 128, 3, 1, "same"),
    ("BatchNorm", 0, 0, 0, 0),
    ("SeparableConv2D", 128, 3, 1, "same"),
    ("BatchNorm", 0, 0, 0, 0),
    
    ("MaxPool", 4, 4, 0, 0),
    ("Flatten", 0, 0, 0, 0),
    ("Dense", 128, 0, 0, 0),
    ("BatchNorm", 0, 0, 0, 0),
    ("Dense", 10, 0, 0, 0),
]

for layer in layers1:
    layer_type, filters, kernel_size, stride, padding = layer

    if layer_type == "Conv2D":
        # Compute output size
        output_size = conv_output_shape(input_size, kernel_size, stride, padding)
        num_params = (kernel_size * kernel_size * input_channels * filters) + filters
        
        macs = (kernel_size * kernel_size * input_channels) * filters * output_size * output_size

        # Update input for next layer
        input_size = output_size
        input_size2 = filters
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
        num_params = input_size2 * filters + filters  # Fully connected layer parameter count
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
 
 
for layer in layers2:
    layer_type, filters, kernel_size, stride, padding = layer

    if layer_type == "Conv2D":
        # Compute output size
        output_size = conv_output_shape(input_size, kernel_size, stride, padding)
        num_params = (kernel_size * kernel_size * input_channels * filters) + filters
        macs = (kernel_size * kernel_size * input_channels) * filters * output_size * output_size
        input_size = output_size
        input_channels = filters

    elif layer_type == "SeparableConv2D":
        # SeparableConv2D: Depthwise + Pointwise Convolution
        output_size = conv_output_shape(input_size, kernel_size, stride, padding)
        num_params = (kernel_size * kernel_size * input_channels) + (kernel_size * kernel_size * input_channels * filters)
        macs = (kernel_size * kernel_size * input_channels) * filters * output_size * output_size
        input_size = output_size
        input_channels = filters

    elif layer_type == "MaxPool":
        output_size = input_size // kernel_size
        num_params = 0  
        macs = 0

    elif layer_type == "Flatten":
        num_params = 0
        macs = 0
        output_size = input_size * input_size * input_channels  # Flattened feature map size

    elif layer_type == "Dense":
        num_params = input_size * filters + filters  # Fully connected layer parameter count
        macs = input_size * filters  # Each input node has a connection to each output node
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
df.to_csv("cnn_analysis_separable.csv", index=False)
print(df)