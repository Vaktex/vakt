package main

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"strings"

	"github.com/spf13/cobra"

	"github.com/vaktex/vakt/internal/brand"
	"github.com/vaktex/vakt/internal/hub"
)

func newSummonCmd() *cobra.Command {
	var (
		model, revision string
		tokenStdin      bool
		offline         bool
	)
	cmd := &cobra.Command{
		Use:   "summon",
		Short: "Download " + brand.ModelName + " into the local cache",
		Long: "summon downloads " + brand.ModelFile + " from Hugging Face (" + brand.ModelRepo + ") into\n" +
			"the cache (" + hub.CacheDir() + "; set $VAKT_CACHE to move all of vakt's caches), verifies its sha256\n" +
			"and prints the path. The model is gated: request access on its Hugging Face page, then set HF_TOKEN, run `hf auth login`,\n" +
			"or pipe a token with --token-stdin.",
		Args: cobra.NoArgs,
		RunE: func(cmd *cobra.Command, _ []string) error {
			rev := ""
			if cmd.Flags().Changed("revision") {
				rev = revision
			}
			spec, err := parseModelSpec(model, rev)
			if err != nil {
				return err
			}
			if spec.Path != "" {
				return errors.New("summon downloads hf: models; a local --model needs no download")
			}
			var token string
			if tokenStdin {
				token, err = readToken(cmd.InOrStdin())
				if err != nil {
					return err
				}
			}
			stderr := cmd.ErrOrStderr()
			type got struct{ path, sha string }
			res, err := withSignIn(cmd.Context(), spec.Repo, cmd.InOrStdin(), stderr, func() (got, error) {
				bar := newDownloadBar(stderr, false)
				defer bar.done()
				p, s, err := hub.Resolve(cmd.Context(), hub.Options{
					Repo: spec.Repo, Revision: spec.Revision, Token: token, Offline: offline, Progress: bar.update,
				})
				return got{p, s}, err
			})
			if err != nil {
				return err
			}
			path, sha := res.path, res.sha
			out := cmd.OutOrStdout()
			fmt.Fprintf(out, "%s@%s ready\n", spec.Repo, spec.Revision)
			fmt.Fprintf(out, "path    %s\n", path)
			fmt.Fprintf(out, "sha256  %s\n", sha)
			return nil
		},
	}
	f := cmd.Flags()
	f.StringVar(&model, "model", defaultModel, "hf:owner/name[@revision]")
	f.StringVar(&revision, "revision", "", "branch, tag or commit (default main)")
	f.BoolVar(&tokenStdin, "token-stdin", false, "read a Hugging Face token from stdin (never stored)")
	f.BoolVar(&offline, "offline", false, "only check the local cache")
	return cmd
}

// readToken reads one line from r. The token is only held in memory.
func readToken(r io.Reader) (string, error) {
	line, err := bufio.NewReader(io.LimitReader(r, 4096)).ReadString('\n')
	if err != nil && !errors.Is(err, io.EOF) {
		return "", errors.New("could not read a token from stdin")
	}
	t := strings.TrimSpace(line)
	if t == "" {
		return "", errors.New("--token-stdin: no token on stdin")
	}
	if strings.ContainsFunc(t, func(r rune) bool { return r < 0x21 || r > 0x7e }) {
		return "", errors.New("--token-stdin: the token contains invalid characters")
	}
	return t, nil
}
