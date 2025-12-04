// Shared
export type PitStop = {
  lap: number; // >=1
  durationMs: number; // 1000..100000
};

export type PredictRequest = {
  circuitId: number | string;
  gridPosition: number; // 1..20
  pitPlan: PitStop[];
  carPerformanceIndex?: number; // 0..1
  round?: number; // 1..25
  driverId?: string;
};

// /predict
export type PredictResponse = {
  prediction: {
    finishP50: number;
    finishP10: number;
    finishP90: number;
  };
  top3: {
    probability: number;
    source: string; // "distribution"
  };
  positionProbs: Record<`P${1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20}`, number>;
  perPitEffects: null; // not used currently
  explanation: null; // not used currently
  modelVersion: string;
};

// /compare
export type CompareScenario = {
  id: string;
  circuitId: number | string;
  gridPosition: number;
  pitPlan: PitStop[];
  carPerformanceIndex?: number;
  round?: number;
};
export type CompareRequest = { scenarios: CompareScenario[] };
export type CompareResult = {
  scenarioId: string;
  finishP50: number;
  intervalWidth: number;
  top3Probability: number;
  robustnessScore: number;
};
export type CompareResponse = {
  results: CompareResult[];
  recommendedScenarioId: string;
  modelVersion: string;
};

// /healthz
export type HealthzResponse = {
  status: "ok" | "error";
  modelVersion: string;
};

// /metadata
export type CircuitMeta = {
  circuitId: number;
  name: string;
  country: string;
  avgLaps: number | null;
  overtakeDifficulty: number; // 0..1-ish
};
export type MetadataResponse = {
  circuits: CircuitMeta[];
  modelVersion: string;
};

// /introspect
export type IntrospectResponse = {
  modelVersion: string;
  gridFeatureName: string;
  feature_names_in_: string[] | null;
};

// /whatif
export type WhatIfResponse = {
  modelVersion: string;
  gridFeatureName: string;
  series: { grid: number; finishP50: number }[];
};
