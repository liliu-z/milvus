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

package segments

import (
	"fmt"

	"github.com/milvus-io/milvus/pkg/v2/proto/querypb"
	"github.com/milvus-io/milvus/pkg/v2/proto/schemapb"
	"github.com/milvus-io/milvus/pkg/v2/paramtable"
)

// SegmentInfoFromLoadInfo creates a SegmentInfo from querypb.SegmentLoadInfo
func SegmentInfoFromLoadInfo(schema *schemapb.CollectionSchema, loadInfo *querypb.SegmentLoadInfo) (*SegmentInfo, error) {
	segmentID := loadInfo.GetSegmentID()
	
	// Estimate metadata size (simplified)
	metaSize := estimateMetadataSize(loadInfo)
	
	// Create cells for different data types
	cells, err := createCellsFromLoadInfo(schema, loadInfo)
	if err != nil {
		return nil, fmt.Errorf("failed to create cells for segment %d: %w", segmentID, err)
	}
	
	return NewSegmentInfo(segmentID, metaSize, cells), nil
}

// estimateMetadataSize estimates the metadata size for a segment
func estimateMetadataSize(loadInfo *querypb.SegmentLoadInfo) uint64 {
	// Simplified metadata size estimation
	// This includes segment schema, field metadata, etc.
	baseMetaSize := uint64(1024 * 1024) // 1MB base metadata
	
	// Add size based on number of fields
	fieldCount := len(loadInfo.GetBinlogPaths())
	fieldMetaSize := uint64(fieldCount) * 1024 // 1KB per field
	
	return baseMetaSize + fieldMetaSize
}

// createCellsFromLoadInfo creates cells from segment load info
func createCellsFromLoadInfo(schema *schemapb.CollectionSchema, loadInfo *querypb.SegmentLoadInfo) ([]*Cell, error) {
	var cells []*Cell
	
	// Create cells for field data
	fieldCells, err := createFieldDataCells(schema, loadInfo)
	if err != nil {
		return nil, err
	}
	cells = append(cells, fieldCells...)
	
	// Create cells for index data
	indexCells, err := createIndexDataCells(schema, loadInfo)
	if err != nil {
		return nil, err
	}
	cells = append(cells, indexCells...)
	
	// Create cells for stats data
	statsCells := createStatsDataCells(loadInfo)
	cells = append(cells, statsCells...)
	
	// Create cells for delta data
	deltaCells := createDeltaDataCells(loadInfo)
	cells = append(cells, deltaCells...)
	
	return cells, nil
}

// createFieldDataCells creates cells for field binlog data
func createFieldDataCells(schema *schemapb.CollectionSchema, loadInfo *querypb.SegmentLoadInfo) ([]*Cell, error) {
	var cells []*Cell
	segmentID := loadInfo.GetSegmentID()
	
	for _, fieldBinlog := range loadInfo.GetBinlogPaths() {
		fieldID := fieldBinlog.GetFieldID()
		fieldSchema := getFieldSchema(schema, fieldID)
		if fieldSchema == nil {
			continue
		}
		
		// Determine if field data is evictable
		evictable := isFieldDataEvictable(fieldSchema, loadInfo)
		
		// Calculate field data size
		fieldSize := uint64(getBinlogDataMemorySize(fieldBinlog))
		
		if fieldSize > 0 {
			cellID := fmt.Sprintf("segment_%d_field_%d_data", segmentID, fieldID)
			cell := NewCell(cellID, fieldSize, evictable)
			cells = append(cells, cell)
		}
	}
	
	return cells, nil
}

// createIndexDataCells creates cells for index data
func createIndexDataCells(schema *schemapb.CollectionSchema, loadInfo *querypb.SegmentLoadInfo) ([]*Cell, error) {
	var cells []*Cell
	segmentID := loadInfo.GetSegmentID()
	
	for _, indexInfo := range loadInfo.GetIndexInfos() {
		fieldID := indexInfo.GetFieldID()
		fieldSchema := getFieldSchema(schema, fieldID)
		if fieldSchema == nil {
			continue
		}
		
		// Index data is usually evictable in tiered storage
		evictable := isIndexDataEvictable(indexInfo, fieldSchema)
		
		// Estimate index size (simplified)
		indexSize := estimateIndexSize(indexInfo)
		
		if indexSize > 0 {
			cellID := fmt.Sprintf("segment_%d_field_%d_index_%d", segmentID, fieldID, indexInfo.GetBuildID())
			cell := NewCell(cellID, indexSize, evictable)
			cells = append(cells, cell)
		}
	}
	
	return cells, nil
}

// createStatsDataCells creates cells for stats data
func createStatsDataCells(loadInfo *querypb.SegmentLoadInfo) []*Cell {
	var cells []*Cell
	segmentID := loadInfo.GetSegmentID()
	
	// Create cells for bloom filter stats
	for _, statsBinlog := range loadInfo.GetStatslogs() {
		fieldID := statsBinlog.GetFieldID()
		statsSize := uint64(getBinlogDataMemorySize(statsBinlog))
		
		if statsSize > 0 {
			cellID := fmt.Sprintf("segment_%d_field_%d_stats", segmentID, fieldID)
			// Stats data is typically not evictable as it's needed for query processing
			cell := NewCell(cellID, statsSize, false)
			cells = append(cells, cell)
		}
	}
	
	// Create cells for BM25 stats
	for _, bm25Log := range loadInfo.GetBm25Logs() {
		fieldID := bm25Log.GetFieldID()
		bm25Size := uint64(getBinlogDataMemorySize(bm25Log))
		
		if bm25Size > 0 {
			cellID := fmt.Sprintf("segment_%d_field_%d_bm25", segmentID, fieldID)
			// BM25 stats might be evictable depending on configuration
			evictable := isBM25StatsEvictable()
			cell := NewCell(cellID, bm25Size, evictable)
			cells = append(cells, cell)
		}
	}
	
	return cells
}

// createDeltaDataCells creates cells for delta data
func createDeltaDataCells(loadInfo *querypb.SegmentLoadInfo) []*Cell {
	var cells []*Cell
	segmentID := loadInfo.GetSegmentID()
	
	for _, deltaBinlog := range loadInfo.GetDeltalogs() {
		fieldID := deltaBinlog.GetFieldID()
		deltaSize := uint64(getBinlogDataMemorySize(deltaBinlog))
		
		// Apply delta data expansion factor
		expansionFactor := paramtable.Get().QueryNodeCfg.DeltaDataExpansionRate.GetAsFloat()
		deltaSize = uint64(float64(deltaSize) * expansionFactor)
		
		if deltaSize > 0 {
			cellID := fmt.Sprintf("segment_%d_field_%d_delta", segmentID, fieldID)
			// Delta data is typically not evictable as it's needed for query correctness
			cell := NewCell(cellID, deltaSize, false)
			cells = append(cells, cell)
		}
	}
	
	return cells
}

// getFieldSchema returns the field schema for a given field ID
func getFieldSchema(schema *schemapb.CollectionSchema, fieldID int64) *schemapb.FieldSchema {
	for _, field := range schema.GetFields() {
		if field.GetFieldID() == fieldID {
			return field
		}
	}
	return nil
}

// isFieldDataEvictable determines if field data can be evicted
func isFieldDataEvictable(fieldSchema *schemapb.FieldSchema, loadInfo *querypb.SegmentLoadInfo) bool {
	// System fields are typically not evictable
	if isSystemField(fieldSchema.GetFieldID()) {
		return false
	}
	
	// Primary key fields are typically not evictable
	if fieldSchema.GetIsPrimaryKey() {
		return false
	}
	
	// Check if there's an index that includes the raw data
	// If so, the raw data might be evictable
	for _, indexInfo := range loadInfo.GetIndexInfos() {
		if indexInfo.GetFieldID() == fieldSchema.GetFieldID() {
			// If index contains raw data, then raw data is evictable
			return indexContainsRawData(indexInfo)
		}
	}
	
	// Default: scalar fields might be evictable, vector fields typically not
	return !isVectorField(fieldSchema)
}

// isIndexDataEvictable determines if index data can be evicted
func isIndexDataEvictable(indexInfo *querypb.FieldIndexInfo, fieldSchema *schemapb.FieldSchema) bool {
	// Vector indexes are typically evictable in tiered storage
	if isVectorField(fieldSchema) {
		return true
	}
	
	// Scalar indexes might be evictable depending on type
	return isScalarIndexEvictable(indexInfo)
}

// isSystemField checks if a field is a system field
func isSystemField(fieldID int64) bool {
	// Common system field IDs in Milvus
	systemFields := []int64{0, 1, 100, 101} // adjust based on actual system field IDs
	for _, id := range systemFields {
		if fieldID == id {
			return true
		}
	}
	return false
}

// isVectorField checks if a field is a vector field
func isVectorField(fieldSchema *schemapb.FieldSchema) bool {
	dataType := fieldSchema.GetDataType()
	return dataType == schemapb.DataType_FloatVector ||
		dataType == schemapb.DataType_BinaryVector ||
		dataType == schemapb.DataType_Float16Vector ||
		dataType == schemapb.DataType_BFloat16Vector ||
		dataType == schemapb.DataType_SparseFloatVector
}

// indexContainsRawData checks if an index contains raw data
func indexContainsRawData(indexInfo *querypb.FieldIndexInfo) bool {
	// This is a simplified check - in reality, you'd need to check the specific index type
	// and its configuration to determine if it includes raw data
	return false // Conservative default
}

// isScalarIndexEvictable determines if a scalar index is evictable
func isScalarIndexEvictable(indexInfo *querypb.FieldIndexInfo) bool {
	// This depends on the specific index type and usage patterns
	// For now, assume scalar indexes are not evictable
	return false
}

// isBM25StatsEvictable determines if BM25 stats are evictable
func isBM25StatsEvictable() bool {
	// BM25 stats might be evictable depending on configuration
	// For now, assume they are evictable
	return true
}

// estimateIndexSize estimates the size of an index
func estimateIndexSize(indexInfo *querypb.FieldIndexInfo) uint64 {
	// Use the provided index size if available
	if size := indexInfo.GetIndexSize(); size > 0 {
		return uint64(size)
	}
	
	// Fallback to a simple estimation based on number of files
	fileCount := len(indexInfo.GetIndexFilePaths())
	if fileCount == 0 {
		return 0
	}
	
	// Simple estimation: assume each file is around 100MB
	return uint64(fileCount) * 100 * 1024 * 1024
}

// ConvertResourceUsageToSegmentInfo converts legacy ResourceUsage to new SegmentInfo
func ConvertResourceUsageToSegmentInfo(segmentID int64, usage ResourceUsage) *SegmentInfo {
	var cells []*Cell
	
	// Create a single cell for the entire segment data
	// This is a simplified conversion - in reality, you'd want to break this down further
	totalDataSize := usage.MemorySize
	if totalDataSize > 0 {
		cellID := fmt.Sprintf("segment_%d_legacy_data", segmentID)
		// Assume most data is evictable for now
		cell := NewCell(cellID, totalDataSize, true)
		cells = append(cells, cell)
	}
	
	// Use a small metadata size estimate
	metaSize := uint64(1024 * 1024) // 1MB
	
	return NewSegmentInfo(segmentID, metaSize, cells)
} 