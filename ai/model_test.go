package ai_test

import (
	"math"
	"testing"

	"github.com/mrsibe/goat/ai"
)

func TestModelPricingCalculate(t *testing.T) {
	pricing := ai.ModelPricing{Input: 3, Output: 15, CacheRead: 0.3, CacheWrite: 3.75}
	usage := ai.Usage{
		Input:        1_000_000,
		Output:       500_000,
		CacheRead:    200_000,
		CacheWrite:   100_000,
		CacheWrite1H: 50_000,
	}
	cost := pricing.Calculate(usage)
	if !closeEnough(cost.Input, 3) || !closeEnough(cost.Output, 7.5) ||
		!closeEnough(cost.CacheRead, 0.06) || !closeEnough(cost.CacheWrite, 0.4875) ||
		!closeEnough(cost.Total, 11.0475) {
		t.Fatalf("cost = %+v", cost)
	}
	if cost.TotalFromProvider {
		t.Fatal("catalog cost must not claim provider billing")
	}
	pricing.ApplyCost(&usage)
	if !closeEnough(usage.Cost.Total, cost.Total) {
		t.Fatalf("usage cost = %+v", usage.Cost)
	}
}

func TestModelInfoThinkingLevels(t *testing.T) {
	plain := ai.ModelInfo{}
	if plain.SupportsThinkingLevel(ai.ThinkingHigh) || !plain.SupportsThinkingLevel(ai.ThinkingOff) {
		t.Fatal("models without reasoning only support off")
	}

	reasoning := ai.ModelInfo{Reasoning: true}
	if !reasoning.SupportsThinkingLevel(ai.ThinkingHigh) ||
		reasoning.SupportsThinkingLevel(ai.ThinkingXHigh) {
		t.Fatal("default reasoning models support off..high only")
	}
	if got := reasoning.ClampThinkingLevel(ai.ThinkingMax); got != ai.ThinkingHigh {
		t.Fatalf("clamped = %q", got)
	}

	extended := ai.ModelInfo{Reasoning: true, ThinkingLevels: []ai.ThinkingLevel{ai.ThinkingOff, ai.ThinkingMax}}
	if !extended.SupportsThinkingLevel(ai.ThinkingMax) || extended.SupportsThinkingLevel(ai.ThinkingHigh) {
		t.Fatal("explicit levels must win")
	}
	if got := extended.ClampThinkingLevel(ai.ThinkingMedium); got != ai.ThinkingOff {
		t.Fatalf("clamped = %q", got)
	}
}

func closeEnough(left, right float64) bool {
	return math.Abs(left-right) < 1e-9
}
