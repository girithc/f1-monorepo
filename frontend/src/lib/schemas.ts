import { z } from 'zod';

export const pitStopSchema = z.object({
  lap: z.coerce.number().min(1, 'Lap must be at least 1'),
  durationMs: z.coerce.number().min(1000, 'Duration must be at least 1000ms').max(100000, 'Duration must be at most 100000ms'),
});

export const predictRequestSchema = z.object({
  circuitId: z.union([z.string().min(1, 'Circuit is required'), z.number()]),
  gridPosition: z.coerce.number().min(1).max(20),
  pitPlan: z.array(pitStopSchema),
  carPerformanceIndex: z.coerce.number().min(0).max(1),
  round: z.coerce.number().min(1).max(25),
});

export const suggestPitStrategySchema = z.object({
  circuitId: z.coerce.number(),
  gridPosition: z.coerce.number().min(1).max(20),
  carPerformanceIndex: z.coerce.number().min(0).max(1),
  round: z.coerce.number().min(1).max(25),
});
