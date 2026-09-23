//go:build mlx && linux && amd64 && !cuda

package mlx

/*
#cgo CFLAGS: -I${SRCDIR}/../../../third_party/mlx/install/linux_amd64_cpu/include
#cgo LDFLAGS: -L${SRCDIR}/../../../third_party/mlx/install/linux_amd64_cpu/lib -lmlxc -lmlx -lopenblas -llapack -lpthread -lstdc++ -lm
*/
import "C"
