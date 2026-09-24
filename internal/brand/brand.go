// Package brand is the single place the product and binary names live.
package brand

const (
	Product   = "Vaktex OSS"
	Binary    = "vakt"
	Tagline   = "on watch for vulnerable code"
	ModelRepo = "vaktex/dom-oss-0.8b"
	ModelName = "DOM-0.8B"
	ModelFile = "model.safetensors"
)

// Set at link time by the build (-X).
var (
	Version = "dev"
	Commit  = "none"
	Backend = "none" // metal | cuda | cpu | fake | none (no native engine linked)
)
