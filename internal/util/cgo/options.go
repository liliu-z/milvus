package cgo

func getDefaultOpt() *options {
	return &options{
		name: "unknown",
	}
}

type options struct {
	name        string
	skipManager bool
}

// Opt is the option type for future.
type Opt func(*options)

// WithName sets the name of the future.
// Only used for metrics.
func WithName(name string) Opt {
	return func(o *options) {
		o.name = name
	}
}

// WithSkipManager skips futureManager registration and context.WithCancel.
// Use when the caller guarantees the context won't be canceled before the future completes.
func WithSkipManager() Opt {
	return func(o *options) {
		o.skipManager = true
	}
}
