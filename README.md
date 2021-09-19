# lnn
This is a generalized neural network package with clean and transparent API for the Go language. Also available for javascript [github/lovesaroha/lnn.js](https://github.com/lovesaroha/lnn.js) 

## Features
- Lightweight and Fast.
- Native Go implementation.
- Tensor Operations.
- Sequential Models.
- Support loss functions like (Mean Square Error).
- Opitmization algorithms like (Gradient Descent).

## Requirements
- Go 1.9 or higher. We aim to support the 3 latest versions of Go.

## Installation
Simple install the package to your [$GOPATH](https://github.com/golang/go/wiki/GOPATH "GOPATH") with the [go tool](https://golang.org/cmd/go/ "go command") from shell:
```bash
$ go get -u github.com/lovesaroha/lnn
```
Make sure [Git is installed](https://git-scm.com/downloads) on your machine and in your system's `PATH`.

## Tensor Usage

### Create Tensor

```Golang
  // Create tensor of given shape.
  tensor := lnn.Tensor([]int{3, 4})
  // Print values.
  tensor.Print()
```
![image](https://raw.githubusercontent.com/lovesaroha/gimages/main/53.png)

### Random Tensor

```Golang
  // Create tensor of given shape and (minimum, maximum).
  tensor := lnn.Tensor([]int{3, 4} , -1 , 1)
  // Print values.
  tensor.Print()
  // Scalar tensor.
  stensor := lnn.Tensor([]int{} , -1 , 1)
  // Print values.
  stensor.Print()
```
![image](https://raw.githubusercontent.com/lovesaroha/gimages/main/54.png)

### Convert Slice Or Values Into Tensor

```Golang
    // Slice of int to tensor and print values.
    lnn.ToTensor([]int{1, 2, 3}).Print()
    // 2d slice of int to tensor and print values.
    lnn.ToTensor([][]int{[]int{1, 2},[]int{3, 4}}).Print()
    // Value to tensor and print values.
    lnn.ToTensor(5).Print()
```
![image](https://raw.githubusercontent.com/lovesaroha/gimages/main/55.png)

### Tensor Element Wise Operations (Add, Subtract, Multiply, Divide) 

```Golang
  // Create a random tensor.
  tensor := lnn.Tensor([]int{3, 4} , 10 , 20)
  tensorB := lnn.Tensor([]int{3, 4} , 0 , 10)

  // Add and print values.
  tensor.Add(tensorB).Print()
  // Subtract and print values.
  tensor.Sub(tensorB).Print()
  // Multiply and print values.
  tensor.Mul(tensorB).Print()
  // Divide and print values.
  tensor.Sub(tensorB).Print()
```
![image](https://raw.githubusercontent.com/lovesaroha/gimages/main/56.png)

### Tensor Element Wise Operations With Scalar Value (Add, Subtract, Multiply, Divide) 

```Golang
  // Create a random tensor.
  tensor := lnn.Tensor([]int{3, 4} , 10 , 20)
  tensorB := lnn.ToTensor(4)

  // Add and print values.
  tensor.Add(tensorB).Print()
  // Subtract and print values.
  tensor.Sub(tensorB).Print()
  // Multiply and print values.
  tensor.Mul(tensorB).Print()
  // Divide and print values.
  tensor.Sub(tensorB).Print()
```
![image](https://raw.githubusercontent.com/lovesaroha/gimages/main/57.png)

### Tensor Dot Product 

```Golang
  // Create a random tensor.
  tensor := lnn.Tensor([]int{3, 4} , 10 , 20)
  tensorB := lnn.Tensor([]int{4, 3} , 0 , 10)

  // Dot product and print values.
  tensor.Dot(tensorB).Print()
```
![image](https://raw.githubusercontent.com/lovesaroha/gimages/main/58.png)

### Tensor Transpose
```Golang
  // Create a random tensor.
  tensor := lnn.Tensor([]int{3, 1} , 10 , 20)

  // Print values.
  tensor.Print()
  // Transpose and print values.
  tensor.Transpose().Print()
```
![image](https://raw.githubusercontent.com/lovesaroha/gimages/main/59.png)

### Add All Column Values
```Golang
  // Create a random tensor.
  tensor := lnn.Tensor([]int{3, 3} , 10 , 20)

  // Print values.
  tensor.Print()
  // Add columns and print values.
  tensor.AddCols().Print()
```
![image](https://raw.githubusercontent.com/lovesaroha/gimages/main/60.png)

### Change Tensor Values (Map)
```Golang
  // Create a random tensor.
  tensor := lnn.Tensor([]int{3, 3} , 0, 10)

  // Print values.
  tensor.Print()
  // Square and print values.
  tensor.Map(func (value float64) float64 {
    return value * value
  }).Print()
```
![image](https://raw.githubusercontent.com/lovesaroha/gimages/main/61.png)

## Model Usage

### Create Model
```golang
  // Create a model.
  model := lnn.Model()
```

### Add Layers In Model
```golang
  // Add layer to model with 4 units , Input shape (2) and activation function relu.
  model.AddLayer(lnn.LayerConfig{InputShape: []int{2}, Units: 4, Activation: "relu"})
  // Add another layer to model with 1 unit and activation function sigmoid.
  model.AddLayer(lnn.LayerConfig{Units: 1})
```

### Model Configuration
```golang
  // Makes the model with given values of loss function , optimizer and learning rate.
  model.Make(lnn.ModelConfig{Loss: "meanSquareError", Optimizer: "sgd", LearningRate: 0.2})
```

### Train Model
```golang
  // Trains the model with given configuration.
  model.Train(inputs, outputs, lnn.TrainConfig{Epochs: 1000, BatchSize: 4, Shuffle: true})
```

### Predict Output
```golang
  model.Predict(inputs)
```
## Examples

### XOR Gate Model Training

```golang
  // Create a model.
  model := lnn.Model()
  
  // Add layer to model with 4 units , Input shape (2) and activation function relu.
  model.AddLayer(lnn.LayerConfig{InputShape: []int{2}, Units: 4, Activation: "relu"})
  
  // Add another layer to model with 1 unit and activation function sigmoid.
  model.AddLayer(lnn.LayerConfig{Units: 1})
  
  // Makes the model with given values of loss function , optimizer and learning rate.
  model.Make(lnn.ModelConfig{Loss: "meanSquareError", Optimizer: "sgd", LearningRate: 0.2})
  
  // Inputs and outputs as a tensor object.
  inputs := lnn.ToTensor([][]float64{[]float64{1, 1, 0, 0}, []float64{1, 0, 1, 0}})
  outputs := lnn.ToTensor([][]float64{[]float64{0, 1, 1, 0}}) 
  
  // Trains the model with given configuration.
  model.Train(inputs, outputs, lnn.TrainConfig{Epochs: 5000, BatchSize: 4, Shuffle: true})
  
  // Print values.
  model.Predict(inputs).Print() 
```

![image](https://raw.githubusercontent.com/lovesaroha/gimages/main/62.png)

### Logistic Regression (OR Gate)  With Tensor
```Golang 
  // Learning rate.
  learningRate := lnn.ToTensor(0.2)
  size := lnn.ToTensor(4)
  
  // Inputs and outputs as a tensor object.
  inputs := lnn.ToTensor([][]float64{[]float64{1, 1, 0, 0}, []float64{1, 0, 1, 0}})
  outputs := lnn.ToTensor([][]float64{[]float64{1, 1, 1, 0}}) 

  // Weights and bias.
  weights := lnn.Tensor([]int{2, 1} , -1, 1)
  bias := lnn.Tensor([]int{}, -1, 1)

  // Train weights and bias (epochs 1000).
  for i := 0; i < 1000; i++ {
    // Use lmath package for Sigmoid(wx + b).
    prediction := weights.Transpose().Dot(inputs).Add(bias).Map(lmath.Sigmoid)
    dZ := prediction.Sub(outputs)
    weights = weights.Sub(inputs.Dot(dZ.Transpose()).Divide(size).Mul(learningRate))
    bias = bias.Sub(dZ.AddCols().Divide(size).Mul(learningRate))
  }

  // Show prediction.
  weights.Transpose().Dot(inputs).Add(bias).Map(lmath.Sigmoid).Print()

```
![image](https://raw.githubusercontent.com/lovesaroha/gimages/main/63.png)