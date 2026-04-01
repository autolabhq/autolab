package main

type OpKind uint8

const (
	opPut OpKind = iota
	opGet
	opDelete
	opCount
)

type Operation struct {
	Kind   OpKind
	Key    string
	Tag    string
	Value  []byte
	Prefix string
}

type Phase struct {
	Name    string
	Workers [][]Operation
}

type Workload struct {
	WorkerCount int
	Phases      []Phase
	TotalOps    int64
}

type Result struct {
	PutOps       int64
	GetHits      int64
	GetMisses    int64
	DeleteHits   int64
	DeleteMisses int64
	ScanOps      int64
	ScanTotal    int64
	FinalKeys    int64
	FinalBytes   int64
	IndexKeys    int64
	WALRecords   int64
	WALBytes     int64
	WALChecksum  uint64
	Checksum     uint64
}

type workerResult struct {
	putOps       int64
	getHits      int64
	getMisses    int64
	deleteHits   int64
	deleteMisses int64
	scanOps      int64
	scanTotal    int64
	checksum     uint64
}

type StoreStats struct {
	LiveKeys    int64
	LiveBytes   int64
	IndexKeys   int64
	WALRecords  int64
	WALBytes    int64
	WALChecksum uint64
}
