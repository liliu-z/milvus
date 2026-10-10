#include <atomic>
#include <future>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include "milvus-storage/format/parquet/scoped_file_prefetch.h"
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
  std::mutex mutex;
  std::condition_variable cv;
  bool blocked = false, entered = false;
  void BeforeRead() {
    std::unique_lock lock(mutex);
    entered = true; cv.notify_all();
    cv.wait(lock, [&] { return !blocked; });
  }
  void Release() { std::lock_guard lock(mutex); blocked = false; cv.notify_all(); }

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
    stats_->BeforeRead();
    ++stats_->reads;
    ARROW_RETURN_NOT_OK(stats_->fault);
    return file_->ReadAt(p, n, out);
  }
  arrow::Result<std::shared_ptr<arrow::Buffer>> ReadAt(int64_t p, int64_t n) override {
    stats_->BeforeRead();
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


  using Scope = milvus_storage::parquet::ScopedFilePrefetch;
  check(Scope::ReservedBytes() == 0, "initial reservation");
  {
    auto scope = Scope::Reserve(fs, relative_path, size);
    check(bool(scope), "scope reservation failed");
    check(!Scope::Reserve(fs, relative_path, size), "duplicate admitted");
    scope->Run();
    int reads = stats->reads;
    auto prepared = make_reader(props, size);
    ok(prepared->open());
    check(take(prepared->get_chunks({0}))[0]->column(0)->Equals(data), "handoff decode mismatch");
    check(stats->reads == reads && scope->Hits() == 1, "reader must consume actual prefetched bytes");
    check(!Scope::Take(fs, relative_path, size), "handoff consumed twice");
    scope->Cancel();
    prepared.reset();
    auto cold = make_reader(props, size); ok(cold->open());
    check(stats->reads == reads + 1, "release must require fresh IO");
  }
  check(Scope::ReservedBytes() == 0, "reservation leaked after consumption");
  {
    auto other_fs = std::make_shared<CountedFS>(stats);
    auto scope = Scope::Reserve(fs, relative_path, size);
    check(!Scope::Take(other_fs, relative_path, size), "cross-filesystem reuse");
    check(!Scope::Take(fs, relative_path + "x", size), "wrong object key reuse");
    check(!Scope::Take(fs, relative_path, size + 1), "wrong size reuse");
    int reads = stats->reads;
    check(bool(Scope::Take(fs, relative_path, size)), "foreground must claim a queued read");
    scope->Run();
    check(stats->reads == reads + 1, "queued work duplicated the foreground read");
  }
  {
    std::vector<std::shared_ptr<Scope>> scopes;
    for (int i = 0; i < 4; ++i) {
      auto scope = Scope::Reserve(fs, "budget" + std::to_string(i), 4 * 1024 * 1024);
      check(bool(scope), "budget reservation unexpectedly denied"); scopes.push_back(scope);
    }
    check(!Scope::Reserve(fs, "overflow", 1), "global byte budget exceeded");
    check(!Scope::Reserve(fs, "large", 4 * 1024 * 1024 + 1), "per-file byte budget exceeded");
  }
  check(Scope::ReservedBytes() == 0, "budget not returned");
  {
    std::vector<std::shared_ptr<Scope>> scopes;
    for (int i = 0; i < 16; ++i) scopes.push_back(Scope::Reserve(fs, "count" + std::to_string(i), 1));
    check(!Scope::Reserve(fs, "count-overflow", 1), "entry budget exceeded");
  }
  {
    auto old = Scope::Reserve(fs, relative_path, size);
    old->Cancel();
    auto current = Scope::Reserve(fs, relative_path, size);
    old.reset();
    check(bool(Scope::Take(fs, relative_path, size)), "old scope erased replacement generation");
  }
  {
    auto scope = Scope::Reserve(fs, relative_path, size);
    { std::lock_guard lock(stats->mutex); stats->blocked = true; stats->entered = false; }
    auto io = std::async(std::launch::async, [scope] { scope->Run(); });
    bool entered;
    { std::unique_lock lock(stats->mutex);
      entered = stats->cv.wait_for(lock, std::chrono::seconds(2), [&] { return stats->entered; }); }
    if (!entered) { stats->Release(); io.get(); check(false, "IO did not start"); }
    auto waiter = std::async(std::launch::async, [&] { return Scope::Take(fs, relative_path, size); });
    scope->Cancel();
    bool woke = waiter.wait_for(std::chrono::seconds(2)) == std::future_status::ready;
    stats->Release(); io.get();
    check(!waiter.get() && woke, "cancel failed to release foreground waiter");
    check(!Scope::Take(fs, relative_path, size), "canceled bytes remained reusable");
  }
  check(Scope::ReservedBytes() == 0, "canceled IO reservation leaked");
  for (auto code : {ExtendStatusCode::StorageTransientThrottling, ExtendStatusCode::StorageTransientTimeout,
                    ExtendStatusCode::AwsErrorNotFound, ExtendStatusCode::AwsErrorAccessDenied}) {
    auto scope = Scope::Reserve(fs, relative_path, size);
    stats->fault = MakeExtendError(code, "injected prefetch and fallback failure");
    scope->Run();
    check(make_reader(props, size)->open().Equals(stats->fault), "fallback destroyed storage status");
    check(scope->Hits() == 0, "failed IO was consumed");
  }
  for (auto fault : {arrow::Status::Cancelled("cancel"), arrow::Status::OutOfMemory("oom")}) {
    auto scope = Scope::Reserve(fs, relative_path, size); stats->fault = fault; scope->Run();
    check(make_reader(props, size)->open().Equals(fault), "fallback destroyed cancellation/OOM");
  }
  {
    auto scope = Scope::Reserve(fs, relative_path, size);
    stats->fault = arrow::Status::IOError("one speculative error"); scope->Run();
    stats->fault = arrow::Status::OK();
    ok(make_reader(props, size)->open());
    check(scope->Hits() == 0, "fallback recovery counted as a cache hit");
  }
  {
    auto scope = Scope::Reserve(fs, relative_path, size + 1); scope->Run();
    auto status = make_reader(props, size + 1)->open();
    auto detail = ExtendStatusDetail::UnwrapStatus(status);
    check(detail && detail->code() == ExtendStatusCode::PackedFileCorrupted, "short read must reach corruption path");
  }
  check(Scope::ReservedBytes() == 0, "final reservation leaked");

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
  std::cout << "PASS: scoped handoff, consume-once, real Parquet decode, fresh IO after release, identity isolation, byte/entry budgets, generation-safe cancel, queued foreground ownership, cancellation, failure/recovery/corruption status preservation, metadata reuse/accounting\n";
}
