/*  Love Saroha
    lovesaroha1994@gmail.com (email address)
    https://www.lovesaroha.com (website)
    https://github.com/lovesaroha  (github)
*/
package lnn

import (
	"fmt"
	"log"

	"./lmath"
)

// Tensor structure.
type TensorObject struct {
	Shape  []int
	Values interface{}
}

// Print tensor.
func (ts TensorObject) Print() {
	switch v := ts.Values.(type) {
	case float64:
		fmt.Printf("\n %f \n %s \n\n", v, "Scalar")
		return
	case [][]float64:
		fmt.Printf("\n")
		for i := 0; i < ts.Shape[0]; i++ {
			fmt.Printf(" [ ")
			for j := 0; j < ts.Shape[1]; j++ {
				fmt.Printf(" %f ", v[i][j])
			}
			fmt.Printf(" ]  \n")
		}
		fmt.Printf(" %s %v %s %v %s \n\n", "(", ts.Shape[0], "x", ts.Shape[1], ")")
	}
}

// Copy tensor values.
func (ts TensorObject) Copy() TensorObject {
	var newTensor = TensorObject{}
	// Copy shape.
	for i := 0; i < len(ts.Shape); i++ {
		newTensor.Shape = append(newTensor.Shape, ts.Shape[i])
	}
	// Check shape.
	switch len(ts.Shape) {
	case 0:
		newTensor.Values = ts.Values
		return newTensor
	case 2:
		newTensor.Values = [][]float64{}
		matrix := ts.Values.([][]float64)
		for i := 0; i < ts.Shape[0]; i++ {
			r := make([]float64, ts.Shape[1])
			for j := 0; j < ts.Shape[1]; j++ {
				r[j] = matrix[i][j]
			}
			newTensor.Values = append(newTensor.Values.([][]float64), r)
		}
	}
	return newTensor
}

// Add tensor.
func (ts TensorObject) Add(arg TensorObject) TensorObject {
	return elementWise(ts, arg, 0)
}

// Subtract tensor.
func (ts TensorObject) Sub(arg TensorObject) TensorObject {
	return elementWise(ts, arg, 1)
}

// Multiplication.
func (ts TensorObject) Mul(arg TensorObject) TensorObject {
	return elementWise(ts, arg, 2)
}

// Square.
func (ts TensorObject) Square() TensorObject {
	return elementWise(ts, ts, 2)
}

// Divide.
func (ts TensorObject) Divide(arg TensorObject) TensorObject {
	return elementWise(ts, arg, 3)
}

// Dot product.
func (ts TensorObject) Dot(arg TensorObject) TensorObject {
	var newTensor TensorObject
	if len(ts.Shape) == 0 || len(arg.Shape) == 0 {
		log.Fatal("lnn: Cannot perform dot product on scalar values.")
		return newTensor
	}
	if ts.Shape[1] != arg.Shape[0] {
		log.Fatal("lnn: Number of columns of first matrix must be equal to number of rows in second.")
		return newTensor
	}
	newTensor.Shape = []int{ts.Shape[0], arg.Shape[1]}
	matrix := ts.Values.([][]float64)
	matrixArg := arg.Values.([][]float64)
	values := make([][]float64, ts.Shape[0])
	for i := 0; i < newTensor.Shape[0]; i++ {
		r := make([]float64, newTensor.Shape[1])
		for j := 0; j < newTensor.Shape[1]; j++ {
			var sum float64
			for k := 0; k < ts.Shape[1]; k++ {
				sum += matrix[i][k] * matrixArg[k][j]
			}
			r[j] = sum
		}
		values[i] = r
	}
	newTensor.Values = values
	return newTensor
}

// Transpose.
func (ts TensorObject) Transpose() TensorObject {
	var newTensor TensorObject
	switch len(ts.Shape) {
	case 1:
		newTensor.Values = ts.Values
	case 2:
		newTensor.Shape = []int{ts.Shape[1], ts.Shape[0]}
		values := make([][]float64, newTensor.Shape[0])
		matrix := ts.Values.([][]float64)
		// Matrix or vector.
		for i := 0; i < newTensor.Shape[0]; i++ {
			r := make([]float64, newTensor.Shape[1])
			for j := 0; j < newTensor.Shape[1]; j++ {
				r[j] = matrix[j][i]
			}
			values[i] = r
		}
		newTensor.Values = values
	}
	return newTensor
}

// Map function.
func (ts TensorObject) Map(callback func(value float64) float64) TensorObject {
	var newTensor = ts.Copy()
	switch len(ts.Shape) {
	case 0:
		value := ts.Values.(float64)
		value = callback(value)
		newTensor.Values = value
	case 2:
		values := ts.Values.([][]float64)
		for i := 0; i < newTensor.Shape[0]; i++ {
			for j := 0; j < newTensor.Shape[1]; j++ {
				values[i][j] = callback(values[i][j])
			}
		}
		newTensor.Values = values
	}
	return newTensor
}

// Add all.
func (ts TensorObject) Sum() float64 {
	var sum float64
	switch len(ts.Shape) {
	case 0:
		return ts.Values.(float64)
	case 2:
		values := ts.Values.([][]float64)
		for i := 0; i < ts.Shape[0]; i++ {
			for j := 0; j < ts.Shape[1]; j++ {
				sum += values[i][j]
			}
		}
		return sum
	}
	return sum
}

// Values of a matrix tensor.
func (ts TensorObject) Value() [][]float64 {
	switch len(ts.Shape) {
	case 1:
		return [][]float64{{ts.Values.(float64)}}
	case 2:
		return ts.Values.([][]float64)
	}
	return [][]float64{}
}

// Col extend.
func (ts TensorObject) ColExtend(scale int) TensorObject {
	var newTensor TensorObject
	// Check shape.
	switch len(ts.Shape) {
	case 0:
		return ts
	case 2:
		newTensor.Shape = []int{ts.Shape[0], ts.Shape[1] * scale}
		matrix := make([][]float64, ts.Shape[0])
		values := ts.Values.([][]float64)
		for i := 0; i < ts.Shape[0]; i++ {
			r := make([]float64, newTensor.Shape[1])
			for j := 0; j < newTensor.Shape[1]; j++ {
				r[j] = values[i][j/scale]
			}
			matrix[i] = r
		}
		newTensor.Values = matrix
	}
	return newTensor
}

// Add columns.
func (ts TensorObject) AddCols() TensorObject {
	var newTensor TensorObject
	if len(ts.Shape) == 0 {
		// Scalar.
		return ts
	} else if len(ts.Shape) == 2 {
		newTensor.Shape = []int{ts.Shape[0], 1}
		values := make([][]float64, newTensor.Shape[0])
		matrix := ts.Values.([][]float64)
		for i := 0; i < ts.Shape[0]; i++ {
			sum := 0.0
			for j := 0; j < ts.Shape[1]; j++ {
				sum += matrix[i][j]
			}
			values[i] = []float64{sum}
		}
		newTensor.Values = values
	}
	return newTensor
}

// Make batches.
func (ts TensorObject) MakeBatches(size int) []TensorObject {
	var newTensor []TensorObject
	if len(ts.Shape) == 0 {
		// Scalar.
		return []TensorObject{ts.Copy()}
	} else if len(ts.Shape) == 2 {
		// Matrix.
		totalBatches := ts.Shape[1] / size
		if totalBatches*size != ts.Shape[1] {
			totalBatches += 1
		}
		matrix := ts.Values.([][]float64)
		initial := size
		for t := 0; t < totalBatches; t++ {
			initial = t * size
			limit := initial + size
			if limit > ts.Shape[1] {
				limit = ts.Shape[1]
			}
			var nts TensorObject = TensorObject{Shape: []int{ts.Shape[0], limit - initial}}
			values := [][]float64{}
			for i := 0; i < ts.Shape[0]; i++ {
				r := make([]float64, limit-initial)
				var c = 0
				for j := initial; j < limit; j++ {
					r[c] = matrix[i][j]
					c++
				}
				values = append(values, r)
			}
			nts.Values = values
			newTensor = append(newTensor, nts)
		}
	} else {
		return []TensorObject{ts}
	}
	return newTensor
}

// Export function.
func Tensor(shape []int, args ...float64) TensorObject {
	// Default values.
	var newTensor TensorObject
	if len(shape) == 0 {
		newTensor.Shape = shape
	} else if len(shape) == 1 {
		newTensor.Shape = []int{shape[0], 1}
	} else {
		newTensor.Shape = []int{shape[0], shape[1]}
	}
	var min float64 = 0
	var max float64 = 0
	// Assign arguments.
	for k, arg := range args {
		if k == 0 {
			// Minimum.
			min = arg
		} else if k == 1 {
			// Maximum.
			max = arg
		}
	}
	// Check shape.
	switch len(newTensor.Shape) {
	case 0:
		newTensor.Values = lmath.Random(min, max)
	case 2:
		matrix := [][]float64{}
		for i := 0; i < newTensor.Shape[0]; i++ {
			matrix = append(matrix, []float64{})
			for j := 0; j < newTensor.Shape[1]; j++ {
				matrix[i] = append(matrix[i], lmath.Random(min, max))
			}
		}
		newTensor.Values = matrix
	}
	return newTensor
}

// ToTensor function convert given value to tensor.
func ToTensor(value interface{}) TensorObject {
	switch v := value.(type) {
	case int:
		return TensorObject{Values: float64(v)}
	case float64:
		return TensorObject{Values: v}
	case []int:
		var newTensor = TensorObject{Shape: []int{len(v), 1}}
		newTensor.Values = toMatrix(sliceToF64(v))
		return newTensor
	case []float64:
		var newTensor = TensorObject{Shape: []int{len(v), 1}}
		newTensor.Values = toMatrix(v)
		return newTensor
	case [][]int:
		var newTensor = TensorObject{Shape: []int{len(v), len(v[0])}}
		newTensor.Values = slice2dToF64(v)
		return newTensor
	case [][]float64:
		var newTensor = TensorObject{Shape: []int{len(v), len(v[0])}}
		newTensor.Values = v
		return newTensor
	}
	return TensorObject{}
}

// Element wise operation.
func elementWise(ts TensorObject, arg TensorObject, operation int) TensorObject {
	// Create result tensor.
	if len(ts.Shape) == 0 && len(arg.Shape) == 0 {
		return scalarOperations(ts, arg, operation)
	} else if len(ts.Shape) == 0 && len(arg.Shape) == 2 {
		return elementWiseWithMatrix(arg, ts, operation)
	} else if len(ts.Shape) == 2 && len(arg.Shape) == 0 {
		return elementWiseWithMatrix(ts, arg, operation)
	}
	return elementWiseWithMatrix(ts, arg, operation)
}

// Element wise with matrix.
func elementWiseWithMatrix(ts TensorObject, arg TensorObject, operation int) TensorObject {
	newTensor := ts.Copy()
	matrix := newTensor.Values.([][]float64)
	switch len(arg.Shape) {
	case 0:
		for i := 0; i < newTensor.Shape[0]; i++ {
			for j := 0; j < newTensor.Shape[1]; j++ {
				switch operation {
				case 0:
					matrix[i][j] = matrix[i][j] + arg.Values.(float64)
				case 1:
					matrix[i][j] = matrix[i][j] - arg.Values.(float64)
				case 2:
					matrix[i][j] = matrix[i][j] * arg.Values.(float64)
				case 3:
					matrix[i][j] = matrix[i][j] / arg.Values.(float64)
				}
			}
		}
	case 2:
		if ts.Shape[0] == arg.Shape[0] && arg.Shape[1] == 1 {
			arg = arg.ColExtend(ts.Shape[1])
		}
		matrixArg := arg.Values.([][]float64)
		for i := 0; i < arg.Shape[0]; i++ {
			for j := 0; j < arg.Shape[1]; j++ {
				switch operation {
				case 0:
					matrix[i][j] = matrix[i][j] + matrixArg[i][j]
				case 1:
					matrix[i][j] = matrix[i][j] - matrixArg[i][j]
				case 2:
					matrix[i][j] = matrix[i][j] * matrixArg[i][j]
				case 3:
					matrix[i][j] = matrix[i][j] / matrixArg[i][j]
				}
			}
		}
	}
	return newTensor
}

// Scalar values.
func scalarOperations(ts TensorObject, arg TensorObject, operation int) TensorObject {
	newTensor := ts.Copy()
	switch operation {
	case 0:
		newTensor.Values = newTensor.Values.(float64) + arg.Values.(float64)
	case 1:
		newTensor.Values = newTensor.Values.(float64) - arg.Values.(float64)
	case 2:
		newTensor.Values = newTensor.Values.(float64) * arg.Values.(float64)
	case 3:
		newTensor.Values = newTensor.Values.(float64) / arg.Values.(float64)
	}
	return newTensor
}

// Convert [] to []float64.
func sliceToF64(val interface{}) []float64 {
	var newSlice []float64
	switch v := val.(type) {
	case []int:
		for i := 0; i < len(v); i++ {
			newSlice = append(newSlice, float64(v[i]))
		}
	case []int64:
		for i := 0; i < len(v); i++ {
			newSlice = append(newSlice, float64(v[i]))
		}
	case []float32:
		for i := 0; i < len(v); i++ {
			newSlice = append(newSlice, float64(v[i]))
		}
	}
	return newSlice
}

// Convert [][] to [][]float64.
func slice2dToF64(val interface{}) [][]float64 {
	var newSlice [][]float64
	switch v := val.(type) {
	case [][]int:
		for i := 0; i < len(v); i++ {
			newSlice = append(newSlice, sliceToF64(v[i]))
		}
	case [][]int64:
		for i := 0; i < len(v); i++ {
			newSlice = append(newSlice, sliceToF64(v[i]))
		}
	case [][]float32:
		for i := 0; i < len(v); i++ {
			newSlice = append(newSlice, sliceToF64(v[i]))
		}
	}
	return newSlice
}

// Vector to matrix.
func toMatrix(value []float64) [][]float64 {
	matrix := [][]float64{}
	for i := 0; i < len(value); i++ {
		matrix = append(matrix, []float64{})
		for j := 0; j < 1; j++ {
			matrix[i] = append(matrix[i], value[i])
		}
	}
	return matrix
}
