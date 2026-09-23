//go:build mlx && darwin

package mlx

/*
#cgo CFLAGS: -I${SRCDIR}/../../../third_party/mlx/install/darwin_arm64_metal/include
#cgo LDFLAGS: -L${SRCDIR}/../../../third_party/mlx/install/darwin_arm64_metal/lib -lmlxc -lmlx -framework Metal -framework Foundation -framework QuartzCore -framework Accelerate -lc++
*/
import "C"
