[settings]
os=Linux
arch=armv8
compiler=gcc
compiler.version=12
compiler.libcxx=libstdc++11
compiler.cppstd=20
build_type=Release
[conf]
tools.build:compiler_executables={"c": "/usr/bin/gcc-12", "cpp": "/usr/bin/g++-12"}
tools.build:jobs=8
opentelemetry-cpp/*:tools.cmake.cmaketoolchain:extra_variables={"WITH_STL": {"value": "CXX17", "cache": True, "force": True, "type": "STRING"}}
opentelemetry-cpp/*:tools.info.package_id:confs=["tools.cmake.cmaketoolchain:extra_variables"]
