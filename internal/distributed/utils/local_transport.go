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

package utils

import (
	"sync"

	"github.com/prometheus/client_golang/prometheus"
	"google.golang.org/grpc"

	"github.com/milvus-io/milvus/pkg/v3/metrics"
	"github.com/milvus-io/milvus/pkg/v3/util/netutil"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
)

var localTransportMetricsOnce sync.Once

// ServeGRPC retains TCP serving and optionally provides an in-process byte
// transport at the same advertised address. The complete gRPC stack still runs.
func ServeGRPC(server *grpc.Server, listener *netutil.NetListener) error {
	if !paramtable.Get().CommonCfg.InProcessGRPCEnabled.GetAsBool() {
		return server.Serve(listener)
	}
	localTransportMetricsOnce.Do(func() {
		metrics.GetRegisterer().MustRegister(prometheus.NewCounterFunc(prometheus.CounterOpts{
			Name: "milvus_internal_in_process_grpc_connections_total",
			Help: "Successful internal gRPC connections using the experimental in-process byte transport.",
		}, func() float64 { return float64(grpc.InProcessDialCount()) }))
	})
	stop, err := server.ServeInProcess(listener.Address(), 256*1024)
	if err != nil {
		return err
	}
	defer stop()
	return server.Serve(listener)
}
