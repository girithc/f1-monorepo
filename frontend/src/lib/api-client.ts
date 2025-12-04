import type {
  HealthzResponse,
  MetadataResponse,
  PredictRequest,
  PredictResponse,
  CompareRequest,
  CompareResponse,
  IntrospectResponse,
  WhatIfResponse,
} from './types';

const BASE = "https://captivating-emotion-production.up.railway.app";

export async function getHealth(): Promise<HealthzResponse> {
  const r = await fetch(`${BASE}/healthz`);
  if (!r.ok) {
    const errorText = await r.text();
    console.error("Health check failed:", errorText);
    throw new Error(errorText);
  }
  return r.json();
}

export async function getMetadata(): Promise<MetadataResponse> {
  const r = await fetch(`${BASE}/metadata`);
  if (!r.ok) {
    const errorText = await r.text();
    console.error("Get metadata failed:", errorText);
    throw new Error(errorText);
  }
  return r.json();
}

export async function predict(body: PredictRequest): Promise<PredictResponse> {
  const r = await fetch(`${BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    const errorText = await r.text();
    console.error("Predict failed:", errorText);
    throw new Error(errorText);
  }
  return r.json();
}

export async function compare(body: CompareRequest): Promise<CompareResponse> {
  const r = await fetch(`${BASE}/compare`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    const errorText = await r.text();
    console.error("Compare failed:", errorText);
    throw new Error(errorText);
  }
  return r.json();
}

export async function introspect(): Promise<IntrospectResponse> {
  const r = await fetch(`${BASE}/introspect`);
  if (!r.ok) {
    const errorText = await r.text();
    console.error("Introspect failed:", errorText);
    throw new Error(errorText);
  }
  return r.json();
}

export async function whatIf(body: PredictRequest): Promise<WhatIfResponse> {
  const r = await fetch(`${BASE}/whatif`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    const errorText = await r.text();
    console.error("What-if failed:", errorText);
    throw new Error(errorText);
  }
  return r.json();
}
