'use server';

import { suggestPitStrategy, type SuggestPitStrategyInput } from '@/ai/flows/ai-pit-strategy-advisor';

export async function getAIAdvice(input: SuggestPitStrategyInput) {
  try {
    const result = await suggestPitStrategy(input);
    return { success: true, data: result };
  } catch (error) {
    console.error('AI strategy suggestion failed:', error);
    return { success: false, error: 'Failed to get AI-powered advice. Please try again.' };
  }
}
