//go:build !unix

package coding

import "os/exec"

func configureCommand(*exec.Cmd) {}

func killCommand(cmd *exec.Cmd) error {
	if cmd.Process == nil {
		return nil
	}
	return cmd.Process.Kill()
}
