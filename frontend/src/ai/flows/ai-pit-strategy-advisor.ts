// src/ai/flows/ai-pit-strategy-advisor.ts
'use server';

/**
 * @fileOverview An AI agent to suggest pit stop strategies.
 *
 * - suggestPitStrategy - A function that suggests pit stop strategies.
 * - SuggestPitStrategyInput - The input type for the suggestPitStrategy function.
 * - SuggestPitStrategyOutput - The return type for the suggestPitStrategy function.
 */

import {ai} from '@/ai/genkit';
import {z} from 'genkit';

const SuggestPitStrategyInputSchema = z.object({
  circuitId: z.number().describe('The ID of the circuit.'),
  carPerformanceIndex: z.number().describe('The performance index of the car (0.0 - 1.0).'),
  avgTireScore: z.number().describe('The average tire score (0.0 - 3.0).'),
  round: z.number().describe('The round number (1 - 25).'),
  gridPosition: z.number().describe('The starting grid position (1-20)'),
});
export type SuggestPitStrategyInput = z.infer<typeof SuggestPitStrategyInputSchema>;

const SuggestPitStrategyOutputSchema = z.object({
  pitPlan: z.array(
    z.object({
      lap: z.number().describe('The lap number for the pit stop.'),
      durationMs: z.number().describe('The duration of the pit stop in milliseconds.'),
    })
  ).describe('An array of suggested pit stops.'),
  explanation: z.string().describe('Explanation of why the pit stops were suggested.'),
});
export type SuggestPitStrategyOutput = z.infer<typeof SuggestPitStrategyOutputSchema>;

export async function suggestPitStrategy(input: SuggestPitStrategyInput): Promise<SuggestPitStrategyOutput> {
  return suggestPitStrategyFlow(input);
}

const prompt = ai.definePrompt({
  name: 'suggestPitStrategyPrompt',
  input: {schema: SuggestPitStrategyInputSchema},
  output: {schema: SuggestPitStrategyOutputSchema},
  prompt: `You are an expert race strategist. Given the following race conditions, suggest an optimal pit stop strategy to maximize the driver's finishing position.

Circuit ID: {{{circuitId}}}
Car Performance Index: {{{carPerformanceIndex}}}
Tire Score: {{{avgTireScore}}}
Round: {{{round}}}
Grid Position: {{{gridPosition}}}

Consider factors such as track layout, weather conditions (assume dry), fuel consumption, and tire wear. Provide a pit stop strategy with specific lap numbers and pit stop durations.

Return the pit plan and a brief explanation of the strategy.

Output pit stops as an array of lap number / duration pairs.
`,
});

const suggestPitStrategyFlow = ai.defineFlow(
  {
    name: 'suggestPitStrategyFlow',
    inputSchema: SuggestPitStrategyInputSchema,
    outputSchema: SuggestPitStrategyOutputSchema,
  },
  async input => {
    const {output} = await prompt(input);
    return output!;
  }
);
