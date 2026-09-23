package main

import "os/exec"

var debug = true

// Run executes a command.
func Run(cmd string) error {
	return exec.Command("sh", "-c", cmd).Run()
}

type Store struct{ root string }

func (s *Store) Get(key string) string {
	return s.root + key
}
