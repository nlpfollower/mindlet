module mindlet

go 1.23.1

replace github.com/nlpfollower/deltamind/orchestration => ./../orchestration

require (
	github.com/nlpfollower/deltamind/orchestration v0.0.0-00010101000000-000000000000
	github.com/spf13/cobra v1.8.1
	github.com/stretchr/testify v1.10.0
	google.golang.org/grpc v1.65.0
)

require (
	github.com/davecgh/go-spew v1.1.1 // indirect
	github.com/inconshreveable/mousetrap v1.1.0 // indirect
	github.com/pmezard/go-difflib v1.0.0 // indirect
	github.com/spf13/pflag v1.0.5 // indirect
	golang.org/x/net v0.25.0 // indirect
	golang.org/x/sys v0.20.0 // indirect
	golang.org/x/text v0.15.0 // indirect
	google.golang.org/genproto/googleapis/rpc v0.0.0-20240528184218-531527333157 // indirect
	google.golang.org/protobuf v1.34.1 // indirect
	gopkg.in/yaml.v3 v3.0.1 // indirect
)
