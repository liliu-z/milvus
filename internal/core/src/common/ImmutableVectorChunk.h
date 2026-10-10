// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once

#include <arrow/array/array_binary.h>
#include <arrow/buffer.h>
#include <cstdint>
#include <memory>

#include "common/Chunk.h"
#include "common/FieldMeta.h"

namespace milvus {

class ImmutableFloatVectorChunk final : public FixedWidthChunk {
 public:
    ImmutableFloatVectorChunk(int32_t rows,
                              int32_t dim,
                              std::shared_ptr<arrow::Buffer> values)
        : FixedWidthChunk(
              rows,
              dim,
              reinterpret_cast<char*>(const_cast<uint8_t*>(values->data())),
              values->capacity(),
              sizeof(float),
              false,
              nullptr),
          values_(std::move(values)) {
    }

 private:
    std::shared_ptr<arrow::Buffer> values_;
};

// The caller must keep the input values immutable for the returned chunk's
// lifetime. This is for fresh decoded storage data, not the copying ChunkWriter
// API. Adjacent slices may be joined only when they cover one whole aligned
// buffer. Partial slices cannot retain a much larger parent allocation or
// undercount cache memory.
inline std::shared_ptr<Chunk>
TryAdoptImmutableFloatVector(const FieldMeta& field,
                             const arrow::ArrayVector& arrays) {
    constexpr int64_t max_bytes = 4 * 1024 * 1024;
    if (field.get_data_type() != DataType::VECTOR_FLOAT ||
        field.is_nullable() || arrays.empty()) {
        return nullptr;
    }
    int64_t rows = 0;
    std::shared_ptr<arrow::Buffer> values;
    const int64_t width = field.get_dim() * sizeof(float);
    if (width <= 0 || width > max_bytes)
        return nullptr;
    for (const auto& input : arrays) {
        auto array =
            std::dynamic_pointer_cast<arrow::FixedSizeBinaryArray>(input);
        if (!array || !array->values() || array->offset() != rows ||
            array->null_count() != 0 || array->length() <= 0 ||
            array->byte_width() != width ||
            array->length() > max_bytes / width - rows) {
            return nullptr;
        }
        if (!values) {
            values = array->values();
        } else if (array->values() != values) {
            return nullptr;
        }
        rows += array->length();
    }
    if (!values || values->parent() || values->size() != rows * width ||
        values->capacity() > max_bytes ||
        reinterpret_cast<uintptr_t>(values->data()) % 64 != 0) {
        return nullptr;
    }
    return std::make_shared<ImmutableFloatVectorChunk>(
        rows, field.get_dim(), values);
}

}  // namespace milvus
