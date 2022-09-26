// console.log("Website works!")

// // Method 1 of initializing a tensor
const data = tf.tensor([[4,6],[5,9],[13,25],[1,57]])
data.print()
// // Method 2 of initializing a tensor
// const shape = [4,2]
// const data2 = tf.tensor([4,6,5,9,13,25,1,57],shape)
// data2.print()
// // Tensors are immutables and cannot be changed.
// // However, variables can be changes using the assign() method
// // Length must match in the assign method
// const data3 = tf.variable(tf.zeros([8]))
// data3.print()
// data3.assign(tf.tensor1d([1,2,3,4,5,6,7,8]))

// // Operations (Ops) - methods to manipulate data
// const data4 = tf.tensor1d([4,6,5,9])
// const data5 = tf.tensor1d([2,7,5,6])
// // Add 2 tensors and print the result
// data4.add(data5).print()
// // Multiply 2 tensors and print the results (element wise multlipication - i.e. [2*4, 6*7, 5*5, 9*6])
// data4.mul(data5).print()

// // define a simple model
// function simpleAdd(input1, input2) {
//   // tify is the tool that tensorflow uses to handle memory allocation inside the GPU
//   // tidy is used to free up GPU memory once the funciton returns
//   // inside, we are writing a lambda function that uses the GPU
//   return tf.tidy(() => {
//     const x1 = input1;
//     const x2 = input2;
//     const y = x1.add(x2)
//     return y;
//   });
// }
// const data_m1 = tf.tensor1d([4,6,5,9])
// const data_m2 = tf.tensor1d([2,7,5,6])
// const result = simpleAdd(data_m1,data_m2)
// result.print()

// Sequential model
// This model structure is taked from a project of classifying MNIST dataset
// So its a convolutional Neural Network with 10 outputs of 0-9 numbers
const model = tf.sequential();

model.add(
    tf.layers.conv2d({
        // only for the first layer - input shape of the image
        inputShape:[28,28,1],
        // size of kernel
        kernelSize:5,
        // how many filters to use
        filters:8,
        // stide
        strides:1,
        // activation function
        activation:'relu',
        // how to initialize the weights. there are many options in the docs
        kernelInitializer:'varianceScaling'
    })
)

model.add(tf.layers.maxPooling2d({
    poolSize:[2,2],
    strides:[2,2]
}))

model.add(
    tf.layers.conv2d({
        kernelSize:5,
        filters:16,
        strides:1,
        activation:'relu',
        kernelInitializer:'varianceScaling'
    })
)

model.add(tf.layers.maxPooling2d({
    poolSize:[2,2],
    strides:[2,2]
}))

model.add(tf.layers.flatten());

model.add(tf.layers.dense({
    // how many neurons should be in this layer
    units:10,
    kernelInitializer:'varianceScaling',
    activation:'softmax'
}))

model.compile({
    optimizer:'adam',
    loss:'categoricalCrossentropy',
    metrics:['accuracy']
})