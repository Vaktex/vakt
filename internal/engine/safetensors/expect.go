package safetensors

import (
	"errors"
	"fmt"
	"slices"
)

// Expect declares a tensor that must be present with an exact shape and one
// of the allowed dtypes.
type Expect struct {
	Name   string
	Shape  []int64
	DTypes []DType // allowed
}

// Require checks that every expectation is satisfied. Tensors not named in
// expect are ignored. All violations are reported (joined), each naming the
// tensor and the got/want values.
func (h *Header) Require(expect []Expect) error {
	var errs []error
	for _, e := range expect {
		t, ok := h.Tensors[e.Name]
		if !ok {
			errs = append(errs, fmt.Errorf("%w: %q (want %v %v)", ErrMissingTensor, e.Name, e.DTypes, e.Shape))
			continue
		}
		if !slices.Equal(t.Shape, e.Shape) {
			errs = append(errs, fmt.Errorf("%w: %q: got %v, want %v", ErrShapeMismatch, e.Name, t.Shape, e.Shape))
		}
		if len(e.DTypes) > 0 && !slices.Contains(e.DTypes, t.DType) {
			errs = append(errs, fmt.Errorf("%w: %q: got %s, want one of %v", ErrDTypeNotAllowed, e.Name, t.DType, e.DTypes))
		}
	}
	return errors.Join(errs...)
}

// DOM-0.8B (text tower + pooling + two heads) architecture constants.
const (
	domLayers        = 24
	domFullAttnEvery = 4 // layers 3, 7, ..., 23 are full attention
	domHidden        = 1024
	domVocab         = 248320
	domIntermediate  = 3584
	domAuxClasses    = 18
	domPoolHeads     = 4
	domHeadInner     = 2 * domHidden
	// BackbonePrefix is the tensor-name prefix of the published DOM model.
	BackbonePrefix = "backbone."
	// BaseModelPrefix is the tensor-name prefix of the base checkpoint DOM-0.8B was fine-tuned from.
	BaseModelPrefix = "model.language_model."
)

// DOMExpectations returns the full expected tensor list for DOM-0.8B with
// backbone tensor names prefixed by prefix. The classification heads are
// unprefixed and only expected when prefix == BackbonePrefix. Every tensor
// accepts F32, BF16 or F16.
func DOMExpectations(prefix string) []Expect {
	dt := []DType{F32, BF16, F16}
	var out []Expect
	add := func(name string, shape ...int64) {
		out = append(out, Expect{Name: name, Shape: shape, DTypes: slices.Clone(dt)})
	}

	add(prefix+"embed_tokens.weight", domVocab, domHidden)
	add(prefix+"norm.weight", domHidden)
	for i := range domLayers {
		l := fmt.Sprintf("%slayers.%d.", prefix, i)
		add(l+"input_layernorm.weight", domHidden)
		add(l+"post_attention_layernorm.weight", domHidden)
		add(l+"mlp.gate_proj.weight", domIntermediate, domHidden)
		add(l+"mlp.up_proj.weight", domIntermediate, domHidden)
		add(l+"mlp.down_proj.weight", domHidden, domIntermediate)
		if i%domFullAttnEvery == domFullAttnEvery-1 {
			a := l + "self_attn."
			add(a+"q_proj.weight", 4096, domHidden)
			add(a+"k_proj.weight", 512, domHidden)
			add(a+"v_proj.weight", 512, domHidden)
			add(a+"o_proj.weight", domHidden, 2048)
			add(a+"q_norm.weight", 256)
			add(a+"k_norm.weight", 256)
		} else {
			a := l + "linear_attn."
			add(a+"A_log", 16)
			add(a+"conv1d.weight", 6144, 1, 4)
			add(a+"dt_bias", 16)
			add(a+"in_proj_a.weight", 16, domHidden)
			add(a+"in_proj_b.weight", 16, domHidden)
			add(a+"in_proj_qkv.weight", 6144, domHidden)
			add(a+"in_proj_z.weight", 2048, domHidden)
			add(a+"norm.weight", 128)
			add(a+"out_proj.weight", domHidden, 2048)
		}
	}
	if prefix == BackbonePrefix {
		// Attention pooling: 4 learned queries of head_dim 256.
		add("pool.query", domPoolHeads, domHidden/domPoolHeads)
		add("pool.key.weight", domHidden, domHidden)
		add("pool.value.weight", domHidden, domHidden)
		add("pool.project.weight", domHidden, domHidden)
		add("pool.project.bias", domHidden)
		add("pool.norm.weight", domHidden)
		add("pool.norm.bias", domHidden)
		// MLP heads: LayerNorm (net.0) -> Linear (net.1) -> GELU -> Dropout
		// -> Linear (net.4).
		for _, h := range []struct {
			name string
			out  int64
		}{{"binary_head", 1}, {"auxiliary_head", domAuxClasses}} {
			add(h.name+".net.0.weight", domHidden)
			add(h.name+".net.0.bias", domHidden)
			add(h.name+".net.1.weight", domHeadInner, domHidden)
			add(h.name+".net.1.bias", domHeadInner)
			add(h.name+".net.4.weight", h.out, domHeadInner)
			add(h.name+".net.4.bias", h.out)
		}
	}
	return out
}
