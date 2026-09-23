//go:build mlx && linux && arm64 && cuda

package mlx

/*
#cgo CFLAGS: -I${SRCDIR}/../../../third_party/mlx/install/linux_arm64_cuda/include
#cgo LDFLAGS: -L${SRCDIR}/../../../third_party/mlx/install/linux_arm64_cuda/lib -L/usr/local/cuda/lib64 -lmlxc -lmlx -lcudart_static -lcublas -lcublasLt -lnvrtc -lcudnn -lcuda -ldl -lrt -lpthread -lstdc++ -lm
*/
import "C"
