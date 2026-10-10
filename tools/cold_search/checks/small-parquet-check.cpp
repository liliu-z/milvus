#include <atomic>
#include <iostream>
#include <stdexcept>
#include <arrow/api.h>
#include <arrow/filesystem/localfs.h>
#include <arrow/io/file.h>
#include <parquet/arrow/writer.h>
#include "milvus-storage/column_groups.h"
#include "milvus-storage/common/extend_status.h"
#include "milvus-storage/format/parquet/parquet_format_reader.h"
#include "milvus-storage/properties.h"

using milvus_storage::parquet::ParquetFormatReader;
using namespace milvus_storage;

static void check(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
static void ok(arrow::Status status) {
  if (!status.ok()) throw std::runtime_error(status.ToString());
}
template <class T> static T take(arrow::Result<T> result) {
  ok(result.status());
  return std::move(result).ValueOrDie();
}
struct Stats {
  std::atomic<int> reads{0};
  arrow::Status fault = arrow::Status::OK();
};
class CountedFile : public arrow::io::RandomAccessFile {
 public:
  CountedFile(std::shared_ptr<arrow::io::RandomAccessFile> file, std::shared_ptr<Stats> stats)
      : file_(std::move(file)), stats_(std::move(stats)) {}
  arrow::Status Close() override { return file_->Close(); }
  bool closed() const override { return file_->closed(); }
  arrow::Result<int64_t> Tell() const override { return file_->Tell(); }
  arrow::Status Seek(int64_t p) override { return file_->Seek(p); }
  arrow::Result<int64_t> GetSize() override { return file_->GetSize(); }
  arrow::Result<int64_t> Read(int64_t n, void* out) override { return file_->Read(n, out); }
  arrow::Result<std::shared_ptr<arrow::Buffer>> Read(int64_t n) override { return file_->Read(n); }
  arrow::Result<int64_t> ReadAt(int64_t p, int64_t n, void* out) override {
    ++stats_->reads;
    ARROW_RETURN_NOT_OK(stats_->fault);
    return file_->ReadAt(p, n, out);
  }
  arrow::Result<std::shared_ptr<arrow::Buffer>> ReadAt(int64_t p, int64_t n) override {
    ++stats_->reads;
    ARROW_RETURN_NOT_OK(stats_->fault);
    return file_->ReadAt(p, n);
  }
 private:
  std::shared_ptr<arrow::io::RandomAccessFile> file_;
  std::shared_ptr<Stats> stats_;
};
class CountedFS : public arrow::fs::SubTreeFileSystem {
 public:
  explicit CountedFS(std::shared_ptr<Stats> stats)
      : SubTreeFileSystem("/", std::make_shared<arrow::fs::LocalFileSystem>()), stats_(stats) {}
  arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile>> OpenInputFile(const std::string& path) override {
    ARROW_ASSIGN_OR_RAISE(auto file, SubTreeFileSystem::OpenInputFile(path));
    return std::make_shared<CountedFile>(std::move(file), stats_);
  }
  arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile>> OpenInputFile(const arrow::fs::FileInfo& info) override {
    return OpenInputFile(info.path());
  }
 private:
  std::shared_ptr<Stats> stats_;
};

int main(int argc, char** argv) {
  check(argc == 2 && argv[1][0] == '/', "pass an absolute temporary Parquet path");
  const std::string path = argv[1];
  auto local = std::make_shared<arrow::fs::LocalFileSystem>();
  arrow::Int64Builder builder;
  for (int i = 0; i < 1000; ++i) ok(builder.Append(i));
  auto data = take(builder.Finish());
  auto schema = arrow::schema({arrow::field("pk", arrow::int64())});
  auto table = arrow::Table::Make(schema, {data});
  auto output = take(local->OpenOutputStream(path));
  ok(::parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), output, 1000));
  ok(output->Close());
  const auto size = take(local->GetFileInfo(path)).size();
  auto relative_path = path.substr(1);
  api::Properties props;
  api::SetValue(props, PROPERTY_READER_PARQUET_WHOLE_FILE_PREFETCH_LIMIT, "4194304");
  auto stats = std::make_shared<Stats>();
  auto fs = std::make_shared<CountedFS>(stats);
  auto make_reader = [&](const api::Properties& p, int64_t file_size) {
    return std::make_shared<ParquetFormatReader>(fs, relative_path, p, std::vector<std::string>{"pk"}, nullptr, file_size);
  };
  auto reader = make_reader(props, size);
  ok(reader->open());
  check(stats->reads == 1, "prefetch must read the object once");
  auto batches = take(reader->get_chunks({0}));
  check(batches.size() == 1 && batches[0]->num_rows() == 1000, "incorrect decoded rows");
  check(batches[0]->column(0)->Equals(data), "incorrect values");
  auto clone = take(reader->clone_reader());
  check(take(clone->get_chunks({0}))[0]->column(0)->Equals(data), "incorrect cloned values");
  check(stats->reads == 1, "clone must reuse immutable bytes");
  reader.reset(); clone.reset();
  reader = make_reader(props, size);
  ok(reader->open());
  check(stats->reads == 2, "new reader must fetch again after release");
  reader.reset();

  for (const char* limit : {"0", "1"}) {
    api::SetValue(props, PROPERTY_READER_PARQUET_WHOLE_FILE_PREFETCH_LIMIT, limit);
    stats->reads = 0;
    reader = make_reader(props, size);
    ok(reader->open());
    check(take(reader->get_chunks({0}))[0]->column(0)->Equals(data), "fallback values mismatch");
    check(stats->reads >= 2, "disabled/oversized file must retain range reads");
    reader.reset();
  }

  api::SetValue(props, PROPERTY_READER_PARQUET_WHOLE_FILE_PREFETCH_LIMIT, "4194304");
  auto short_status = make_reader(props, size + 1)->open();
  auto detail = ExtendStatusDetail::UnwrapStatus(short_status);
  check(detail && detail->code() == ExtendStatusCode::PackedFileCorrupted && !detail->retryable(), "short read must be permanent corruption");
  check(ToSegcoreErrorCode(detail->code()) == milvus::DataFormatBroken, "short read lost segcore corruption code");

  for (auto code : {ExtendStatusCode::StorageTransientThrottling, ExtendStatusCode::StorageTransientTimeout,
                    ExtendStatusCode::AwsErrorNotFound, ExtendStatusCode::AwsErrorAccessDenied}) {
    stats->fault = MakeExtendError(code, "injected storage error");
    auto status = make_reader(props, size)->open();
    check(status.Equals(stats->fault), "prefetch must preserve original storage error and detail");
  }
  for (auto status : {arrow::Status::Cancelled("injected cancellation"),
                      arrow::Status::OutOfMemory("injected allocation failure")}) {
    stats->fault = status;
    check(make_reader(props, size)->open().Equals(status), "prefetch must preserve cancellation/OOM status");
  }
  stats->fault = arrow::Status::OK();

  api::SetValue(props, PROPERTY_FS_STORAGE_TYPE, "local");
  api::SetValue(props, PROPERTY_FS_ROOT_PATH, "/");
  api::ColumnGroupFile file{path, 0, 1000, {}};
  file.Set(api::kPropertyFileSize, size);
  auto metadata = take(ParquetFormatReader::MetaTrait::load_metadata(file, props, nullptr));
  check(metadata->payload.whole_file && metadata->payload.whole_file->size() == size, "metadata must retain bytes");
  check(metadata->cache_size >= size, "metadata must account for retained bytes");
  ok(local->DeleteFile(path));
  reader = take(ParquetFormatReader::MetaTrait::create_from_metadata(metadata, file, schema, {"pk"}, ""));
  check(take(reader->get_chunks({0}))[0]->column(0)->Equals(data), "metadata reconstruction failed to reuse bytes");
  std::cout << "PASS: one read, clone, release, disabled/threshold fallback, short read, four storage errors, metadata reuse/accounting\n";
}
