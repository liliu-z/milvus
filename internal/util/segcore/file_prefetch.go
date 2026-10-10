package segcore

/*
#cgo pkg-config: milvus_core
#include <stdlib.h>
#include "storage/storage_c.h"
*/
import "C"

import (
	"sync"
	"unsafe"
)

// BeginLoadFilePrefetch starts optional read-only work. End must run on every
// exit from the shared load operation. No Go pointers survive this C call.
func BeginLoadFilePrefetch(manifest string, fields []int64) func() (uint64, uint64) {
	if manifest == "" || len(fields) == 0 {
		return func() (uint64, uint64) { return 0, 0 }
	}
	path := C.CString(manifest)
	defer C.free(unsafe.Pointer(path))
	scope := C.BeginAutoLoadFilePrefetch(path, (*C.int64_t)(unsafe.Pointer(&fields[0])), C.int64_t(len(fields)))
	var once sync.Once
	var reserved, consumed uint64
	return func() (uint64, uint64) {
		once.Do(func() {
			stats := C.EndAutoLoadFilePrefetch(scope)
			reserved, consumed = uint64(stats.files), uint64(stats.hits)
		})
		return reserved, consumed
	}
}
