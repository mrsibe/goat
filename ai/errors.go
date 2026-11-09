package ai

import "fmt"

// APIError is the provider-neutral error returned by model adapters.
type APIError struct {
	Provider   string
	StatusCode int
	Code       string
	Message    string
	RequestID  string
	Retryable  bool
	Err        error
}

func (e *APIError) Error() string {
	if e == nil {
		return "<nil>"
	}
	switch {
	case e.Provider != "" && e.StatusCode != 0:
		return fmt.Sprintf("%s api error (%d): %s", e.Provider, e.StatusCode, e.Message)
	case e.Provider != "":
		return fmt.Sprintf("%s api error: %s", e.Provider, e.Message)
	default:
		return e.Message
	}
}

func (e *APIError) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.Err
}
