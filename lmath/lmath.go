/*  Love Saroha
    lovesaroha1994@gmail.com (email address)
    https://www.lovesaroha.com (website)
    https://github.com/lovesaroha  (github)
*/
package lmath

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

var i float64

// Random gives float64 number between range.
func Random(min float64, max float64) float64 {
	i++
	rand.Seed(time.Now().UTC().UnixNano() * int64(i))
	return min + rand.Float64()*(max-min)
}

// Map function to map float value in a range.
func Map(value float64, start1 float64, stop1 float64, start2 float64, stop2 float64) float64 {
	return (value-start1)/(stop1-start1)*(stop2-start2) + start2
}

// Sigmoid function.
func Sigmoid(value float64) float64 {
	return 1 / (1 + math.Exp(-value))
}

// Dsigmoid function.
func Dsigmoid(value float64) float64 {
	return value * (1 - value)
}

// Relu function.
func Relu(value float64) float64 {
	return math.Max(0, value)
}

// Drelu function.
func Drelu(value float64) float64 {
	if value <= 0 {
		return 0
	} else {
		return 1
	}
}

// MatrixObject structure.
type MatrixObject struct {
	Rows   int
	Cols   int
	Values [][]float64
}

// Print function.
func (matrix MatrixObject) Print() {
	// Print matrix values.
	fmt.Printf("\n")
	for i := 0; i < matrix.Rows; i++ {
		fmt.Printf(" [ ")
		for j := 0; j < matrix.Cols; j++ {
			fmt.Printf(" %f ", matrix.Values[i][j])
		}
		fmt.Printf(" ] \n")
	}
	fmt.Printf("\n")
}

// This function print shape of matrix.
func (matrix MatrixObject) Shape() {
	fmt.Println("(", matrix.Rows, "x", matrix.Cols, ")")
}

// Assign random values to matrix.
func (matrix *MatrixObject) Random(min float64, max float64, floor bool) {
	// Assign random values.
	for i := 0; i < matrix.Rows; i++ {
		for j := 0; j < matrix.Cols; j++ {
			// If random is true.
			matrix.Values[i][j] = Random(min, max)
			if floor {
				matrix.Values[i][j] = float64(int64(matrix.Values[i][j]))
			}
		}
	}
}

// Copy function create a copy of matrix object.
func (matrix MatrixObject) Copy() MatrixObject {
	newMatrix := MatrixObject{Rows: matrix.Rows, Cols: matrix.Cols}
	// Copy values.
	for i := 0; i < matrix.Rows; i++ {
		r := make([]float64, matrix.Cols)
		for j := 0; j < matrix.Cols; j++ {
			r[j] = matrix.Values[i][j]
		}
		newMatrix.Values = append(newMatrix.Values, r)
	}
	return newMatrix
}

// Transpose function return transpose of matrix.
func (matrix MatrixObject) Transpose() MatrixObject {
	transpose := Matrix(matrix.Cols, matrix.Rows)
	// Assign values.
	for i := 0; i < transpose.Rows; i++ {
		for j := 0; j < transpose.Cols; j++ {
			transpose.Values[i][j] = matrix.Values[j][i]
		}
	}
	return transpose
}

// Add matrix function.
func (matrix MatrixObject) Add(value interface{}) MatrixObject {
	newMatrix := matrix.Copy()
	elementWise(&newMatrix, value, 1)
	return newMatrix
}

// Subtract matrix function.
func (matrix MatrixObject) Sub(value interface{}) MatrixObject {
	newMatrix := matrix.Copy()
	elementWise(&newMatrix, value, 2)
	return newMatrix
}

// Multiply matrix function.
func (matrix MatrixObject) Mul(value interface{}) MatrixObject {
	newMatrix := matrix.Copy()
	elementWise(&newMatrix, value, 3)
	return newMatrix
}

// Divide matrix function.
func (matrix MatrixObject) Divide(value interface{}) MatrixObject {
	newMatrix := matrix.Copy()
	elementWise(&newMatrix, value, 4)
	return newMatrix
}

// Dot function.
func (matrix MatrixObject) Dot(v MatrixObject) MatrixObject {
	var newMatrix MatrixObject
	// Matrix.
	if matrix.Cols != v.Rows {
		fmt.Println("lmath.go: Number of columns of first matrix must be equal to number of rows in second!!")
		return newMatrix
	}
	newMatrix = Matrix(matrix.Rows, v.Cols)
	// Multiply.
	for i := 0; i < newMatrix.Rows; i++ {
		for j := 0; j < newMatrix.Cols; j++ {
			var sum float64
			for k := 0; k < matrix.Cols; k++ {
				sum += matrix.Values[i][k] * v.Values[k][j]
			}
			newMatrix.Values[i][j] = sum
		}
	}
	return newMatrix
}

// This function return sum of all values.
func (matrix MatrixObject) Sum() float64 {
	var sum float64
	for i := 0; i < matrix.Rows; i++ {
		for j := 0; j < matrix.Cols; j++ {
			sum += matrix.Values[i][j]
		}
	}
	return sum
}

// This function add all columns values.
func (matrix MatrixObject) AddCols() MatrixObject {
	newMatrix := MatrixObject{Rows: matrix.Rows, Cols: 1}
	// Add values.
	for i := 0; i < matrix.Rows; i++ {
		var sum float64
		for j := 0; j < matrix.Cols; j++ {
			sum += matrix.Values[i][j]
		}
		newMatrix.Values = append(newMatrix.Values, []float64{sum})
	}
	return newMatrix
}

// Map function.
func (matrix MatrixObject) Map(callback func(value float64) float64) MatrixObject {
	newMatrix := matrix.Copy()
	// Assign values.
	for i := 0; i < newMatrix.Rows; i++ {
		for j := 0; j < newMatrix.Cols; j++ {
			newMatrix.Values[i][j] = callback(newMatrix.Values[i][j])
		}
	}
	return newMatrix
}

// Element wise operation on matrix.
func elementWise(matrix *MatrixObject, value interface{}, operation int8) {
	var fvalue float64
	switch v := value.(type) {
	case int:
		// If argument is a int.
		fvalue = float64(v)
	case float64:
		fvalue = v
	case MatrixObject:
		// If argument is a matrix.
		if matrix.Rows != v.Rows || matrix.Cols != v.Cols {
			fmt.Println("lmath.go: Shape of two matrix must be same in element wise operations!")
			return
		}
		for i := 0; i < matrix.Rows; i++ {
			for j := 0; j < matrix.Cols; j++ {
				if operation == 1 {
					matrix.Values[i][j] += v.Values[i][j]
				} else if operation == 2 {
					matrix.Values[i][j] -= v.Values[i][j]
				} else if operation == 3 {
					matrix.Values[i][j] *= v.Values[i][j]
				} else {
					matrix.Values[i][j] /= v.Values[i][j]
				}
			}
		}
		return
	default:
		fmt.Println("lmath.go: Argument to element wise operation is not valid!")
		return
	}
	// Perform element wise operation.
	for i := 0; i < matrix.Rows; i++ {
		for j := 0; j < matrix.Cols; j++ {
			if operation == 1 {
				matrix.Values[i][j] += fvalue
			} else if operation == 2 {
				matrix.Values[i][j] -= fvalue
			} else if operation == 3 {
				matrix.Values[i][j] *= fvalue
			} else {
				matrix.Values[i][j] /= fvalue
			}
		}
	}
}

// ToMatrix function convert given value to matrix.
func ToMatrix(arg interface{}) MatrixObject {
	var newMatrix MatrixObject
	// Check argument.
	switch v := arg.(type) {
	case []int:
		newMatrix = Matrix(len(v), 1)
		// Assign values.
		for i := 0; i < newMatrix.Rows; i++ {
			for j := 0; j < newMatrix.Cols; j++ {
				newMatrix.Values[i][j] = float64(v[i])
			}
		}
	case [][]int:
		newMatrix = Matrix(len(v), len(v[0]))
		// Assign values.
		for m := 0; m < newMatrix.Rows; m++ {
			for n := 0; n < newMatrix.Cols; n++ {
				if len(v[m]) != newMatrix.Cols {
					fmt.Println("lmath.go: Argument must be a two-dimensional array of right shape!")
					return newMatrix
				}
				newMatrix.Values[m][n] = float64(v[m][n])
			}
		}
	case []float64:
		newMatrix = Matrix(len(v), 1)
		// Assign values.
		for k := 0; k < newMatrix.Rows; k++ {
			for l := 0; l < newMatrix.Cols; l++ {
				newMatrix.Values[k][l] = v[k]
			}
		}
	case [][]float64:
		newMatrix = Matrix(len(v), len(v[0]))
		// Assign values.
		for r := 0; r < newMatrix.Rows; r++ {
			for c := 0; c < newMatrix.Cols; c++ {
				if len(v[r]) != newMatrix.Cols {
					fmt.Println("lmath.go: Argument must be a two-dimensional array of right shape!")
					return newMatrix
				}
				newMatrix.Values[r][c] = float64(v[r][c])
			}
		}
	}
	return newMatrix
}

// Matrix function create new matrix.
func Matrix(rows int, cols int, args ...float64) MatrixObject {
	// Create a new matrix object.
	newMatrix := MatrixObject{Rows: rows, Cols: cols}
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
	// Generate matrix values.
	for i := 0; i < rows; i++ {
		r := make([]float64, cols)
		for j := 0; j < cols; j++ {
			// If random is true.
			if min != 0 || max != 0 {
				r[j] = Random(min, max)
			} else {
				r[j] = 0.0
			}
		}
		newMatrix.Values = append(newMatrix.Values, r)
	}
	return newMatrix
}
