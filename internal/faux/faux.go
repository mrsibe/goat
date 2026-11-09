// Package faux provides a deterministic scripted model for tests.
package faux

import (
	"context"
	"errors"
	"io"

	"github.com/mrsibe/goat/ai"
)

// Turn is one scripted model response.
type Turn struct {
	Events    []ai.Event
	StartErr  error
	StreamErr error
}

// Model returns pre-scripted turns and records requests.
type Model struct {
	Turns    []Turn
	Requests []ai.Request
}

// New creates a scripted model.
func New(turns ...Turn) *Model {
	return &Model{Turns: turns}
}

// Stream implements ai.Model.
func (m *Model) Stream(_ context.Context, request ai.Request) (ai.Stream, error) {
	m.Requests = append(m.Requests, request)
	if len(m.Turns) == 0 {
		return nil, errors.New("faux: unexpected model call")
	}
	turn := m.Turns[0]
	m.Turns = m.Turns[1:]
	if turn.StartErr != nil {
		return nil, turn.StartErr
	}
	return &stream{turn: turn}, nil
}

type stream struct {
	turn    Turn
	current int
	sentErr bool
}

func (s *stream) Next() (ai.Event, error) {
	if s.current < len(s.turn.Events) {
		event := s.turn.Events[s.current]
		s.current++
		return event, nil
	}
	if s.turn.StreamErr != nil && !s.sentErr {
		s.sentErr = true
		return ai.Event{}, s.turn.StreamErr
	}
	return ai.Event{}, io.EOF
}

func (s *stream) Close() error {
	return nil
}
