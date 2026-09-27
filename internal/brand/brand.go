// Package brand is the single place the product and binary names live.
package brand

const (
	Product   = "Vaktex OSS"
	Binary    = "vakt"
	Tagline   = "on watch for vulnerable code"
	ModelRepo = "vaktex/dom-oss-0.8b"
	// ModelCommit is the Hub commit of the verified launch weights
	// (model.safetensors sha256 a396d4f4601b8be4...).
	ModelCommit = "fe1e7a9f728992cb20f0bcc782022ab55898ec11"
	ModelName   = "DOM-0.8B"
	ModelFile   = "model.safetensors"
)

// Set at link time by the build (-X).
var (
	Version = "dev"
	Commit  = "none"
	Backend = "none" // metal | cuda | cpu | fake | none (no native engine linked)
)
