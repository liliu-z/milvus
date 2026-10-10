// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <boost/filesystem.hpp>
#include <folly/CancellationToken.h>
#include <folly/ScopeGuard.h>
#include <gtest/gtest.h>

#include "common/OpContext.h"
#include "common/Schema.h"
#include "common/SystemProperty.h"
#include "futures/Future.h"
#include "query/PlanImpl.h"
#include "segcore/ChunkedSegmentSealedImpl.h"
#include "segcore/SegmentSealed.h"
#include "segcore/SegcoreConfig.h"
#include "segcore/storagev2translator/SystemIndexTranslator.h"
#include "test_utils/Constants.h"
#include "test_utils/DataGen.h"
#include "test_utils/ManifestTestUtil.h"

using namespace milvus;
using namespace milvus::segcore;

namespace {
std::string
TestPath() {
    return (boost::filesystem::path(TestLocalPath) /
            boost::filesystem::unique_path("int64-pk-%%%%-%%%%-%%%%"))
        .string();
}

void
LoadManifest(SegmentSealed& segment,
             const test::V3SegmentTestData& data,
             int64_t rows,
             int64_t id) {
    proto::segcore::SegmentLoadInfo info;
    info.set_segmentid(id);
    info.set_collectionid(1);
    info.set_partitionid(1);
    info.set_num_of_rows(rows);
    info.set_storageversion(STORAGE_V3);
    info.set_manifest_path(data.ManifestPathJson());
    segment.SetLoadInfo(info);
    tracer::TraceContext trace;
    segment.Load(trace, nullptr);
}
}  // namespace

TEST(Int64PrimaryKeyFill, ManifestColumnUnsortedRepeatedOffsetsAndCancellation) {
    auto& config = SegcoreConfig::default_config();
    const bool old_preference = config.get_prefer_field_data_when_index_has_raw_data();
    auto restore = folly::makeGuard([&] {
        config.set_prefer_field_data_when_index_has_raw_data(old_preference);
    });
    config.set_prefer_field_data_when_index_has_raw_data(true);
    auto schema = std::make_shared<Schema>();
    const auto pk = schema->AddDebugField("pk", DataType::INT64);
    const auto other = schema->AddDebugField("other", DataType::INT64);
    schema->AddField(FieldName("ts"), TimestampFieldID, DataType::INT64,
                     false, std::nullopt);
    schema->set_primary_field_id(pk);
    const auto path = TestPath();
    auto cleanup = folly::makeGuard([&] { boost::filesystem::remove_all(path); });
    constexpr int64_t batches = 3, batch_rows = 37, rows = batches * batch_rows;
    test::V3SegmentTestData data(schema, batches, batch_rows, 1,
                                  TestLocalPath, path);
    auto segment = CreateSealedSegment(schema, nullptr, 731);
    LoadManifest(*segment, data, rows, 731);
    auto* sealed = dynamic_cast<ChunkedSegmentSealedImpl*>(segment.get());
    ASSERT_NE(sealed, nullptr);
    ASSERT_NE(sealed->GetChunkedColumn(pk), nullptr);
    auto pk_index = sealed->TestGetPublishedStateSnapshot()->runtime->pk_index_slot;
    ASSERT_NE(pk_index, nullptr);
    ASSERT_FALSE(pk_index->IsCached(0));
    const int64_t other_offsets[] = {38, 0, 110};
    auto other_data = segment->bulk_subscript(nullptr, other, other_offsets, 3);
    ASSERT_EQ(other_data->scalars().long_data().data_size(), 3);
    EXPECT_FALSE(pk_index->IsCached(0)) << "non-PK output must not build the PK index";
    query::Plan plan(schema);

    for (const std::vector<int64_t> offsets :
         {std::vector<int64_t>{110, 0, 38, 74, 38, 36, 73},
          std::vector<int64_t>{}}) {
        SearchResult result;
        result.read_lease_ = sealed->AcquireReadLease(folly::CancellationToken());
        result.seg_offsets_.assign(offsets.begin(), offsets.end());
        result.distances_.resize(offsets.size());
        segment->FillPrimaryKeys(&plan, result);
        ASSERT_EQ(result.pk_type_, DataType::INT64);
        ASSERT_EQ(result.primary_keys_.size(), offsets.size());
        if (!offsets.empty()) {
            EXPECT_FALSE(pk_index->IsCached(0)) << "PK fill should use the loaded column";
            auto raw_output = segment->bulk_subscript(nullptr, pk, offsets.data(),
                                                      offsets.size());
            EXPECT_EQ(raw_output->scalars().long_data().data_size(), offsets.size());
            EXPECT_FALSE(pk_index->IsCached(0)) << "late PK output must also avoid index construction";
        }
        config.set_prefer_field_data_when_index_has_raw_data(false);
        auto reference = segment->bulk_subscript(nullptr, pk, offsets.data(),
                                                 offsets.size());
        config.set_prefer_field_data_when_index_has_raw_data(true);
        if (!offsets.empty()) {
            EXPECT_TRUE(pk_index->IsCached(0)) << "disabled preference must retain indexed lookup";
        }
        for (size_t i = 0; i < offsets.size(); ++i) {
            auto expected = DataGen(schema, batch_rows, 42 + offsets[i] / batch_rows)
                                .get_col<int64_t>(pk);
            EXPECT_EQ(std::get<int64_t>(result.primary_keys_[i]),
                      expected[offsets[i] % batch_rows]);
            EXPECT_EQ(std::get<int64_t>(result.primary_keys_[i]),
                      reference->scalars().long_data().data(i));
        }
    }

    folly::CancellationSource cancel;
    cancel.requestCancellation();
    OpContext ctx(cancel.getToken());
    SearchResult canceled;
    canceled.seg_offsets_.push_back(0);
    canceled.distances_.push_back(1);
    try {
        segment->FillPrimaryKeys(&plan, canceled, &ctx);
        FAIL() << "canceled primary-key fill must fail";
    } catch (const SegcoreError& error) {
        EXPECT_EQ(error.get_error_code(), ErrorCode::FollyCancel);
    }
}

TEST(Int64PrimaryKeyFill, ExternalVirtualPrimaryKeyUsesFallback) {
    auto& config = SegcoreConfig::default_config();
    const bool old_preference = config.get_prefer_field_data_when_index_has_raw_data();
    auto restore = folly::makeGuard([&] {
        config.set_prefer_field_data_when_index_has_raw_data(old_preference);
    });
    config.set_prefer_field_data_when_index_has_raw_data(true);
    auto source_schema = std::make_shared<Schema>();
    const FieldId field(100), pk(101);
    source_schema->AddField(FieldMeta(FieldName("value"), field,
                                      DataType::INT64, false,
                                      std::nullopt, "source_value"));
    source_schema->set_external_source("s3://test-bucket/data");
    source_schema->set_external_spec(R"({"format":"parquet"})");
    const auto path = TestPath();
    auto cleanup = folly::makeGuard([&] { boost::filesystem::remove_all(path); });
    constexpr int64_t rows = 8, segment_id = 732;
    test::V3SegmentTestData data(source_schema, 1, rows, 1,
                                  TestLocalPath, path);
    auto schema = std::make_shared<Schema>(*source_schema);
    schema->AddField(FieldMeta(FieldName("pk"), pk, DataType::INT64,
                               false, std::nullopt));
    schema->AddField(FieldName("RowID"), RowFieldID, DataType::INT64,
                     false, std::nullopt);
    schema->AddField(FieldName("Timestamp"), TimestampFieldID, DataType::INT64,
                     false, std::nullopt);
    schema->set_primary_field_id(pk);
    auto segment = CreateSealedSegment(schema, nullptr, segment_id);
    LoadManifest(*segment, data, rows, segment_id);
    auto* sealed = dynamic_cast<ChunkedSegmentSealedImpl*>(segment.get());
    ASSERT_NE(sealed, nullptr);
    ASSERT_NE(sealed->GetChunkedColumn(pk), nullptr);
    query::Plan plan(schema);
    SearchResult result;
    result.read_lease_ = sealed->AcquireReadLease(folly::CancellationToken());
    result.seg_offsets_ = {7, 0, 3, 7};
    result.distances_.resize(4);
    segment->FillPrimaryKeys(&plan, result);
    ASSERT_EQ(result.pk_type_, DataType::INT64);
    ASSERT_EQ(result.primary_keys_.size(), 4);
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(std::get<int64_t>(result.primary_keys_[i]),
                  (segment_id << 32) | result.seg_offsets_[i]);
    }
}
