Traffic

I began with a small convolutional neural network containing one
convolutional layer, one max-pooling layer, and one dense hidden layer.
This model compiled and trained quickly, but a single convolutional
block had limited capacity to learn both simple edges and more complex
traffic-sign shapes. I therefore added a second convolutional layer
with more filters. The first layer uses 32 filters and the second uses
64 filters, allowing later layers to learn more detailed combinations
of the features found earlier. Each convolutional layer is followed by
max pooling to reduce the spatial size of the data.

The final model rescales pixel values from the 0-255 range to the 0-1
range before convolution. After the convolutional blocks, the output
is flattened and passed to a dense layer with 128 units. I added
dropout at a rate of 0.5 because a dense layer without dropout can fit
the training images too closely. The output layer contains 43 units
with softmax activation, one for each traffic-sign category. The model
uses Adam optimization and categorical cross-entropy loss because this
is a multi-class classification problem.

