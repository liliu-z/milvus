#include <aws/core/Aws.h>
#include <aws/core/auth/AWSCredentialsProvider.h>
#include <aws/core/client/DefaultRetryStrategy.h>
#include <aws/s3/S3Client.h>
#include <aws/s3/model/GetObjectRequest.h>
#include <iostream>
#include <iterator>
#include <stdexcept>

int main() {
  Aws::SDKOptions options;
  Aws::InitAPI(options);
  int status = 0;
  {
    Aws::Client::ClientConfiguration config;
    config.endpointOverride = "127.0.0.1:18099";
    config.scheme = Aws::Http::Scheme::HTTP;
    config.region = "us-east-1";
    config.retryStrategy = Aws::MakeShared<Aws::Client::DefaultRetryStrategy>("check", 0);
    Aws::S3::S3Client client(Aws::Auth::AWSCredentials("test", "test"), config,
        Aws::Client::AWSAuthV4Signer::PayloadSigningPolicy::Never, false);
    for (const char* name : {"no-checksum", "crc-ok", "sha-ok", "crc-bad", "sha-bad"}) {
      Aws::S3::Model::GetObjectRequest request;
      request.SetBucket("bucket");
      request.SetKey(name);
      request.SetChecksumMode(Aws::S3::Model::ChecksumMode::ENABLED);
      auto result = client.GetObject(request);
      const bool bad = std::string(name).find("bad") != std::string::npos;
      if (bad) {
        if (result.IsSuccess() || result.GetError().GetMessage() != "Response checksums mismatch") {
          std::cerr << "FAIL " << name << ": expected checksum rejection; success=" << result.IsSuccess();
          if (!result.IsSuccess()) std::cerr << ", code=" << result.GetError().GetExceptionName()
                                            << ", message=" << result.GetError().GetMessage();
          std::cerr << '\n';
          status = 1;
        } else {
          std::cout << "PASS " << name << ": " << result.GetError().GetMessage() << '\n';
        }
      } else if (!result.IsSuccess()) {
        std::cerr << "FAIL " << name << ": " << result.GetError().GetMessage() << '\n';
        status = 1;
      } else {
        auto outcome = result.GetResultWithOwnership();
        std::string body{std::istreambuf_iterator<char>(outcome.GetBody()), {}};
        if (body.size() != 3 * 1024 * 1024) {
          status = 1;
        } else {
          for (size_t i = 0; i < body.size(); ++i) {
            if (static_cast<unsigned char>(body[i]) != i % 251) {
              status = 1;
              break;
            }
          }
        }
        std::cout << (status ? "FAIL " : "PASS ") << name << ": exact 3MiB body\n";
      }
    }
  }
  Aws::ShutdownAPI(options);
  return status;
}
