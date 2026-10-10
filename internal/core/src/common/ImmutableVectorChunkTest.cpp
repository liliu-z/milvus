// Copyright (C) 2026 Zilliz. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <arrow/io/memory.h>
#include <gtest/gtest.h>
#include <parquet/arrow/reader.h>
#include <cstring>

#include "common/ImmutableVectorChunk.h"
#include "common/ChunkWriter.h"

using namespace milvus;

namespace {
std::shared_ptr<arrow::Array>
MakeDecodedVectorArray(int64_t rows, int32_t dim, bool with_null = false) {
    arrow::FixedSizeBinaryBuilder builder(arrow::fixed_size_binary(dim * 4));
    std::vector<float> row(dim);
    for (int64_t i = 0; i < rows; ++i) {
        std::fill(row.begin(), row.end(), static_cast<float>(i));
        auto status =
            with_null && i == 0
                ? builder.AppendNull()
                : builder.Append(reinterpret_cast<uint8_t*>(row.data()));
        EXPECT_TRUE(status.ok());
    }
    std::shared_ptr<arrow::Array> array;
    EXPECT_TRUE(builder.Finish(&array).ok());
    return array;
}
FieldMeta
VectorMeta(int64_t dim = 768, bool nullable = false) {
    return FieldMeta(FieldName("vector"),
                     FieldId(101),
                     DataType::VECTOR_FLOAT,
                     dim,
                     std::nullopt,
                     nullable,
                     std::nullopt);
}
}  // namespace

TEST(ImmutableVectorChunk, OwnsValuesAfterArrowTableDies) {
    auto array = MakeDecodedVectorArray(1000, 768);
    auto values =
        std::static_pointer_cast<arrow::FixedSizeBinaryArray>(array)->values();
    std::weak_ptr<arrow::Buffer> weak = values;
    auto chunk = TryAdoptImmutableFloatVector(VectorMeta(), {array});
    ASSERT_NE(chunk, nullptr);
    EXPECT_EQ(chunk->Data(), reinterpret_cast<const char*>(values->data()));
    EXPECT_EQ(chunk->Size(), values->capacity());
    EXPECT_EQ(chunk->CellByteSize().memory_bytes, values->capacity());
    EXPECT_EQ(chunk->CellByteSize().file_bytes, 0);
    array.reset();
    values.reset();
    EXPECT_FALSE(weak.expired());
    EXPECT_EQ(chunk->RowNums(), 1000);
    for (int64_t i = 0; i < 1000; ++i) {
        auto row = reinterpret_cast<const float*>(chunk->ValueAt(i));
        for (int d = 0; d < 768; ++d) EXPECT_EQ(row[d], i);
    }
    chunk.reset();
    EXPECT_TRUE(weak.expired());
}

TEST(ImmutableVectorChunk, FallsBackForUnsupportedLayouts) {
    auto array = MakeDecodedVectorArray(1000, 768);
    EXPECT_EQ(
        TryAdoptImmutableFloatVector(VectorMeta(), {array->Slice(1, 100)}),
        nullptr);
    EXPECT_EQ(
        TryAdoptImmutableFloatVector(VectorMeta(), {array->Slice(0, 100)}),
        nullptr);
    EXPECT_EQ(TryAdoptImmutableFloatVector(VectorMeta(), {array, array}),
              nullptr);
    EXPECT_EQ(TryAdoptImmutableFloatVector(VectorMeta(384), {array}), nullptr);
    EXPECT_EQ(TryAdoptImmutableFloatVector(VectorMeta(768, true), {array}),
              nullptr);
    EXPECT_EQ(TryAdoptImmutableFloatVector(
                  VectorMeta(), {MakeDecodedVectorArray(1000, 768, true)}),
              nullptr);
    EXPECT_EQ(TryAdoptImmutableFloatVector(VectorMeta(),
                                           {MakeDecodedVectorArray(0, 768)}),
              nullptr);
    EXPECT_EQ(TryAdoptImmutableFloatVector(VectorMeta(),
                                           {MakeDecodedVectorArray(2000, 768)}),
              nullptr);
}

TEST(ImmutableVectorChunk, ExistingCopyingFactoryStillCopies) {
    auto array = MakeDecodedVectorArray(1000, 768);
    auto values =
        std::static_pointer_cast<arrow::FixedSizeBinaryArray>(array)->values();
    auto chunks = create_group_chunk({FieldId(101)}, {VectorMeta()}, {{array}});
    ASSERT_EQ(chunks.size(), 1);
    EXPECT_NE(chunks.at(FieldId(101))->Data(),
              reinterpret_cast<const char*>(values->data()));
}

TEST(ImmutableVectorChunk, JoinsOnlyContiguousSlicesOfTheSameWholeBuffer) {
    auto array = MakeDecodedVectorArray(1000, 768);
    auto first = array->Slice(0, 342);
    auto middle = array->Slice(342, 342);
    auto last = array->Slice(684, 316);
    auto chunk =
        TryAdoptImmutableFloatVector(VectorMeta(), {first, middle, last});
    ASSERT_NE(chunk, nullptr);
    EXPECT_EQ(chunk->RowNums(), 1000);
    EXPECT_EQ(chunk->Data(),
              reinterpret_cast<const char*>(
                  std::static_pointer_cast<arrow::FixedSizeBinaryArray>(array)
                      ->values()
                      ->data()));
    EXPECT_EQ(TryAdoptImmutableFloatVector(VectorMeta(), {first, last, middle}),
              nullptr);
    EXPECT_EQ(TryAdoptImmutableFloatVector(VectorMeta(), {first, last}),
              nullptr);
    EXPECT_EQ(TryAdoptImmutableFloatVector(
                  VectorMeta(),
                  {first,
                   middle,
                   MakeDecodedVectorArray(1000, 768)->Slice(684, 316)}),
              nullptr);
    auto missing = std::make_shared<arrow::FixedSizeBinaryArray>(
        arrow::fixed_size_binary(768 * 4), 342, nullptr);
    EXPECT_EQ(
        TryAdoptImmutableFloatVector(VectorMeta(), {missing, middle, last}),
        nullptr);
    array.reset();
    first.reset();
    middle.reset();
    last.reset();
    for (int64_t i = 0; i < 1000; ++i) {
        auto row = reinterpret_cast<const float*>(chunk->ValueAt(i));
        for (int d = 0; d < 768; ++d) EXPECT_EQ(row[d], i);
    }
}

TEST(ImmutableVectorChunk, RealParquetBufferIsAdopted) {
    const char* path = std::getenv("COLD_VECTOR_PARQUET");
    if (!path)
        GTEST_SKIP();
    auto file = arrow::io::ReadableFile::Open(path).ValueOrDie();
    auto bytes = file->ReadAt(0, file->GetSize().ValueOrDie()).ValueOrDie();
    auto memory_file = std::make_shared<arrow::io::BufferReader>(bytes);
    parquet::arrow::FileReaderBuilder builder;
    ASSERT_TRUE(builder.Open(memory_file).ok());
    auto props = parquet::default_arrow_reader_properties();
    props.set_batch_size(INT64_MAX);
    props.set_pre_buffer(true);
    std::unique_ptr<parquet::arrow::FileReader> reader;
    ASSERT_TRUE(builder.properties(props)->Build(&reader).ok());
    std::shared_ptr<arrow::Table> table;
    ASSERT_TRUE(reader->ReadTable(&table).ok());
    ASSERT_EQ(table->num_columns(), 1);
    ASSERT_EQ(table->column(0)->num_chunks(), 1);
    auto array = std::dynamic_pointer_cast<arrow::FixedSizeBinaryArray>(
        table->column(0)->chunk(0));
    ASSERT_NE(array, nullptr);
    auto values = array->values();
    auto chunk = TryAdoptImmutableFloatVector(VectorMeta(), {array});
    ASSERT_NE(chunk, nullptr)
        << "rows=" << array->length() << " offset=" << array->offset()
        << " nulls=" << array->null_count() << " width=" << array->byte_width()
        << " size=" << values->size() << " capacity=" << values->capacity()
        << " parent=" << static_cast<bool>(values->parent())
        << " alignment=" << reinterpret_cast<uintptr_t>(values->data()) % 64;
    auto sliced = TryAdoptImmutableFloatVector(
        VectorMeta(),
        {array->Slice(0, 342), array->Slice(342, 342), array->Slice(684, 316)});
    ASSERT_NE(sliced, nullptr);
    EXPECT_EQ(sliced->Data(), chunk->Data());
}
