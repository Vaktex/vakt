//go:build mlx && linux && amd64 && cuda

package mlx

/*
#cgo CFLAGS: -I${SRCDIR}/../../../third_party/mlx/install/linux_amd64_cuda/include
#cgo LDFLAGS: -L${SRCDIR}/../../../third_party/mlx/install/linux_amd64_cuda/lib -L/usr/local/cuda/lib64 -lmlxc -lmlx -Wl,--as-needed -lcudart_static -lcublas -lcublasLt -lnvrtc -lcudnn -lcuda -Wl,--no-as-needed -lopenblas -llapack -ldl -lrt -lpthread -lstdc++ -lm
*/
import "C"
